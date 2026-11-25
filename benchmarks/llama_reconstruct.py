"""Same folding/clipping reconstruction experiment on LLaMA-2 7B."""
from pathlib import Path
import sys
from duquant_reconstruct import Reconstruction,clip_value,logit_clip,activation_ste
from comparison_models import ROOT,MODEL,BASE as LLAMA_BASE,grouped_weights
from duquant_cuda.decoder import LlamaDecoderBlock,load_layer_weights
from duquant_cuda.backend import transform
from duquant_cuda.calibration import apply_smoothing
from duquant_cuda.model import LlamaInferenceModel
from search_ptq import cpu_tree
from safetensors import safe_open
import torch,json
import torch.nn.functional as F

PARENT=ROOT/'benchmarks/cache/ptq_llama2_final/sym_gptq'
BASE=ROOT/'benchmarks/cache/llama_reconstruct'
REPORT=ROOT/'benchmarks/llama_reconstruct'
NAMES=('qkv','o','gate_up','down')
CFG=json.loads((MODEL/'config.json').read_text())


def base_state(i):
    payload=torch.load(PARENT/f'layer_{i:02d}.pt',weights_only=True,map_location='cuda')
    w,_=load_layer_weights(MODEL,i);apply_smoothing(w,payload['scales'])
    return dict(weights=w,rotations=payload['rotations'],plan=payload['quantization'])


def extras():
    index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map'];extra={}
    for dst,key in [('embedding','model.embed_tokens.weight'),('final_norm','model.norm.weight'),('lm_head','lm_head.weight')]:
        with safe_open(str(MODEL/index[key]),framework='pt') as f:extra[dst]=f.get_tensor(key).half()
    return extra


class LlamaReconstruction(Reconstruction):
    def __init__(self,state):
        torch.nn.Module.__init__(self)
        for key,w in state['weights'].items():self.register_buffer(key.replace('.','__'),w.detach().clone())
        for n,rot in state['rotations'].items():
            for j,r in enumerate(rot):self.register_buffer(f'{n}_rot{j}',r)
        self.scale_logs=torch.nn.ParameterDict({n:torch.nn.Parameter(torch.zeros(k,device='cuda')) for n,k in [('qkv',4096),('gate_up',4096),('o',4096),('down',11008)]})
        self.wclips=torch.nn.ParameterDict();self.aclips=torch.nn.ParameterDict()
        for n,w in grouped_weights(state['weights']).items():
            bound=transform(w,*self.rotation(n)).float().abs().amax(1);ratio=(state['plan']['weights'][n]['scale'].float()*7/bound).clamp(.2,1)
            self.wclips[n]=torch.nn.Parameter(logit_clip(ratio,.2));self.aclips[n]=torch.nn.Parameter(logit_clip(torch.tensor(state['plan']['act_clips'][n],device='cuda'),.4))
        self.kvclips=torch.nn.Parameter(logit_clip(torch.tensor(state['plan']['kv_clips'],device='cuda'),.4))
        block=LlamaDecoderBlock(state['weights'],{},'fp16',eps=CFG['rms_norm_eps'],max_position=4096,rope_theta=CFG.get('rope_theta',10000.),kv_bits=16)
        for n in ('cos','sin'):self.register_buffer(n,getattr(block,n))
        self.register_buffer('had',block.hadamard);self.quantize=True
    def scales(self):return {n:p.clamp(-2,2).exp().half() for n,p in self.scale_logs.items()}
    def scaled(self):
        w=self.original();s=self.scales();out=dict(w)
        out['input_layernorm.weight']=(w['input_layernorm.weight'].float()/s['qkv']).half()
        out['post_attention_layernorm.weight']=(w['post_attention_layernorm.weight'].float()/s['gate_up']).half()
        for n in ('q','k','v'):
            z=(w[f'self_attn.{n}_proj.weight'].float()*s['qkv'][None]).half()
            if n=='v':z=z/s['o'][:,None]
            out[f'self_attn.{n}_proj.weight']=z.half()
        out['self_attn.o_proj.weight']=(w['self_attn.o_proj.weight'].float()*s['o'][None]).half()
        for n in ('gate','up'):
            z=(w[f'mlp.{n}_proj.weight'].float()*s['gate_up'][None]).half()
            if n=='up':z=z/s['down'][:,None]
            out[f'mlp.{n}_proj.weight']=z.half()
        out['mlp.down_proj.weight']=(w['mlp.down_proj.weight'].float()*s['down'][None]).half()
        return out
    def norm(self,x,w):return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+CFG['rms_norm_eps'])).half()*w
    def forward(self,x):
        w=self.scaled();ws=grouped_weights(w);b,t,h=x.shape
        packed=self.project(self.norm(x,w['input_layernorm.weight']),ws['qkv'],'qkv')
        q,k,v=[z.reshape(b,t,32,128).transpose(1,2) for z in packed.split(4096,-1)]
        co,si=self.cos[:t],self.sin[:t]
        def rope(z):return z*co+torch.cat((-z[...,64:],z[...,:64]),-1)*si
        q,k=rope(q)@self.had,rope(k)@self.had
        if self.quantize:
            c=clip_value(self.kvclips,.4);k=activation_ste(k,c[0]);v=activation_ste(v,c[1])
        a=F.scaled_dot_product_attention(q,k,v,is_causal=t>1).transpose(1,2).reshape(b,t,h).contiguous()
        r=x+self.project(a,ws['o'],'o');gu=self.project(self.norm(r,w['post_attention_layernorm.weight']),ws['gate_up'],'gate_up')
        a=F.silu(gu[...,:11008])*gu[...,11008:]
        return r+self.project(a,ws['down'],'down')


@torch.no_grad()
def folded_base(index,params):
    state=base_state(index);w=state['weights'];apply_smoothing(w,{n:s.cuda().half() for n,s in params['scales'].items()});return w,state['rotations']


@torch.inference_mode()
def load_trial(mode='cuda',variant='gptq'):
    folder=BASE/variant;m=json.loads((folder/'manifest.json').read_text());assert m['status']=='complete';blocks=[]
    for i in range(32):
        payload=torch.load(folder/f'layer_{i:02d}.pt',weights_only=True,map_location='cuda');w,rot=folded_base(i,payload['parameters'])
        block=LlamaDecoderBlock(w,rot,mode,eps=CFG['rms_norm_eps'],max_position=4096,kv_bits=4,quantization=payload['plan'])
        if blocks:block.cos,block.sin,block.hadamard=blocks[0].cos,blocks[0].sin,blocks[0].hadamard
        blocks.append(block)
    extra={n:t.cuda() for n,t in extras().items()}
    return LlamaInferenceModel(blocks,**extra,eps=CFG['rms_norm_eps']).eval()
