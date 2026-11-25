"""Train folding-only scales and deployed-form clips with frozen DuQuant bases."""
from qwen3_models import *
from comparison_models import grouped_weights
from duquant_cuda.backend import transform
from duquant_cuda.lwc import qdq_ste,ste_round
from duquant_cuda.calibration import apply_smoothing
from search_ptq import cpu_tree
import torch.nn.functional as F

FOLDER=BASE/'duquant_reconstruct'
TRIAL_REPORT=REPORT/'duquant_reconstruct'
NAMES=('qkv','o','gate_up','down')


def logit_clip(x,minimum):
    t=((torch.as_tensor(x).float()-minimum)/(1-minimum)).clamp(1e-4,1-1e-4)
    return torch.logit(t)


def clip_value(x,minimum):return minimum+(1-minimum)*x.sigmoid()


class Reconstruction(torch.nn.Module):
    def __init__(self,state,train_scales=True):
        super().__init__();self.train_scales=train_scales
        for key,w in state['weights'].items():self.register_buffer(key.replace('.','__'),w.detach().clone())
        for name,rot in state['rotations'].items():
            for j,r in enumerate(rot):self.register_buffer(f'{name}_rot{j}',r)
        self.scale_logs=torch.nn.ParameterDict({n:torch.nn.Parameter(torch.zeros(k,device='cuda'),requires_grad=train_scales) for n,k in [('qkv',2048),('gate_up',2048),('o',1024),('down',6144)]})
        self.wclips=torch.nn.ParameterDict();self.aclips=torch.nn.ParameterDict()
        for n,w in grouped_weights(state['weights']).items():
            bound=transform(w,*self.rotation(n)).float().abs().amax(1)
            ratio=(state['plan']['weights'][n]['scale'].float()*7/bound).clamp(.2,1)
            self.wclips[n]=torch.nn.Parameter(logit_clip(ratio,.2))
            self.aclips[n]=torch.nn.Parameter(logit_clip(torch.tensor(state['plan']['act_clips'][n],device='cuda'),.4))
        self.kvclips=torch.nn.Parameter(logit_clip(torch.tensor(state['plan']['kv_clips'],device='cuda'),.4))
        b=QwenBlock(state['weights']);self.register_buffer('cos',b.cos);self.register_buffer('sin',b.sin);self.register_buffer('had',b.hadamard)
        self.quantize=True
    def rotation(self,n):return tuple(getattr(self,f'{n}_rot{j}') for j in range(3))
    def original(self):return {n.replace('__','.'):v for n,v in self.named_buffers() if '__' in n}
    def scales(self):
        values={n:p.clamp(-2,2).exp().half() for n,p in self.scale_logs.items()}
        values['o']=values['o'].reshape(8,1,128).expand(8,2,128).reshape(-1)
        return values
    def scaled(self):
        w=self.original();s=self.scales();out=dict(w)
        # Retain FP16 folding boundaries used by the actual export.
        out['input_layernorm.weight']=(w['input_layernorm.weight'].float()/s['qkv']).half()
        out['post_attention_layernorm.weight']=(w['post_attention_layernorm.weight'].float()/s['gate_up']).half()
        for n in ('q','k','v'):
            z=(w[f'self_attn.{n}_proj.weight'].float()*s['qkv'][None]).half()
            if n=='v':z=z/s['o'].reshape(8,2,128)[:,0].reshape(-1,1)
            out[f'self_attn.{n}_proj.weight']=z.half()
        out['self_attn.o_proj.weight']=(w['self_attn.o_proj.weight'].float()*s['o'][None]).half()
        for n in ('gate','up'):
            z=(w[f'mlp.{n}_proj.weight'].float()*s['gate_up'][None]).half()
            if n=='up':z=z/s['down'][:,None]
            out[f'mlp.{n}_proj.weight']=z.half()
        out['mlp.down_proj.weight']=(w['mlp.down_proj.weight'].float()*s['down'][None]).half()
        return out
    def norm(self,x,w):return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+CFG['rms_norm_eps'])).half()*w
    def project(self,x,w,n):
        z=transform(x,*self.rotation(n));wt=transform(w,*self.rotation(n))
        if self.quantize:
            z=qdq_ste(z,clip_value(self.aclips[n],.4).half())
            scale=(wt.float().abs().amax(1)*clip_value(self.wclips[n],.2)/7).clamp(1e-5,1e4).half()
            codes=ste_round(wt.float()/scale[:,None]).clamp(-8,7)
            wt=(codes*scale[:,None]).half()
        return F.linear(z,wt)
    def forward(self,x):
        w=self.scaled();ws=grouped_weights(w);b,t,h=x.shape
        packed=self.project(self.norm(x,w['input_layernorm.weight']),ws['qkv'],'qkv')
        q,k,v=[z.reshape(b,t,heads,128).transpose(1,2) for z,heads in zip(packed.split((2048,1024,1024),-1),(16,8,8))]
        q,k=self.norm(q,w['self_attn.q_norm.weight']),self.norm(k,w['self_attn.k_norm.weight'])
        co,si=self.cos[:t],self.sin[:t]
        def rope(z):return z*co+torch.cat((-z[...,64:],z[...,:64]),-1)*si
        q,k=rope(q)@self.had,rope(k)@self.had
        if self.quantize:
            clips=clip_value(self.kvclips,.4).half();k=qdq_ste(k,clips[0]);v=qdq_ste(v,clips[1])
        a=F.scaled_dot_product_attention(q,k.repeat_interleave(2,1),v.repeat_interleave(2,1),is_causal=t>1)
        a=a.transpose(1,2).reshape(b,t,h).contiguous()
        r=x+self.project(a,ws['o'],'o')
        gu=self.project(self.norm(r,w['post_attention_layernorm.weight']),ws['gate_up'],'gate_up')
        a=F.silu(gu[...,:6144])*gu[...,6144:]
        return r+self.project(a,ws['down'],'down')
    @torch.no_grad()
    def export(self):
        return cpu_tree(dict(scales=self.scales(),weight_clips={n:clip_value(p,.2) for n,p in self.wclips.items()},
            act_clips={n:float(clip_value(p,.4).half()) for n,p in self.aclips.items()},kv_clips=clip_value(self.kvclips,.4).half().tolist()))


@torch.no_grad()
def folded_base(index,params,device='cuda'):
    state=torch.load(BASE/'duquant'/f'layer_{index:02d}.pt',weights_only=True,map_location=device)
    w=state['weights'];apply_smoothing(w,{n:s.to(device).half() for n,s in params['scales'].items()})
    return w,state['rotations']


@torch.inference_mode()
def load_trial(mode='cuda'):
    meta=json.loads((FOLDER/'manifest.json').read_text());assert meta['status']=='complete'
    blocks=[]
    for i in range(28):
        payload=torch.load(FOLDER/f'layer_{i:02d}.pt',weights_only=True,map_location='cuda')
        w,rot=folded_base(i,payload['parameters']);blocks.append(QwenBlock(w,rot,mode,payload['plan'],4))
    return make_model(blocks)
