"""Layerwise FlatQuant reconstruction with the official transform/quantizer code.

Local Qwen3 adaptation. Uses 15 epochs, batch4, 128x2048 train, official SVD
transforms with diagonals plus LWC/LAC; learned clipping is replaced at final
common-format export, as in the preceding LLaMA FlatQuant experiment.
"""
from qwen3_flat_models import *
import types,time,math,random,numpy as np
import torch.nn.functional as F
sys.path.insert(0,str(ROOT/'benchmarks/cache/FlatQuant'))
from flatquant.trans_utils import SVDDecomposeTransMatrix,SVDSingleTransMatrix
from flatquant.flat_linear import FlatQuantizedLinear
from flatquant.quant_utils import ActivationQuantizer
from flatquant.function_utils import get_decompose_dim
from search_ptq import cpu_tree,sha


class TrainingBlock(torch.nn.Module):
    def __init__(self,w):
        super().__init__()
        args=types.SimpleNamespace(w_bits=4,w_asym=False,a_bits=4,a_asym=True,a_groupsize=-1,lwc=True,lac=True)
        self.linears=torch.nn.ModuleDict()
        for n in ('q','k','v','o','gate','up','down'):
            key=('self_attn.' if n in ('q','k','v','o') else 'mlp.')+n+'_proj.weight'
            weight=w[key];linear=torch.nn.Linear(weight.shape[1],weight.shape[0],bias=False,device='cuda',dtype=torch.float32)
            linear.weight.data.copy_(weight);linear.weight.requires_grad=False
            self.linears[n]=FlatQuantizedLinear(args,linear).cuda()
        self.trans=torch.nn.ModuleDict({n:SVDDecomposeTransMatrix(*get_decompose_dim(k),add_diag=True).cuda() for n,k in [('qkv',2048),('gate_up',2048),('down',6144)]})
        self.otrans=SVDSingleTransMatrix(16).cuda();self.ktrans=SVDSingleTransMatrix(128).cuda();self.vtrans=SVDSingleTransMatrix(128).cuda()
        self.kquant=ActivationQuantizer(4,sym=False,lac=True).cuda();self.vquant=ActivationQuantizer(4,sym=False,lac=True).cuda()
        for n,key in [('norm1','input_layernorm.weight'),('norm2','post_attention_layernorm.weight'),('qnorm','self_attn.q_norm.weight'),('knorm','self_attn.k_norm.weight')]:self.register_buffer(n,w[key])
        block=QwenBlock(w);self.register_buffer('cos',block.cos);self.register_buffer('sin',block.sin)
    def norm(self,x,w):return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+CFG['rms_norm_eps'])).to(x.dtype)*w
    def forward(self,x):
        b,t,h=x.shape
        z=self.trans['qkv'](self.norm(x,self.norm1))
        q=self.linears['q'](z,qa_trans=self.trans['qkv']).reshape(b,t,16,128).transpose(1,2)
        k=self.linears['k'](z,qa_trans=self.trans['qkv']).reshape(b,t,8,128).transpose(1,2)
        v=self.linears['v'](z,qa_trans=self.trans['qkv'],out_trans=self.vtrans).reshape(b,t,8,128).transpose(1,2)
        q,k=self.norm(q,self.qnorm),self.norm(k,self.knorm)
        co,si=self.cos[:t],self.sin[:t]
        def rope(z):return z*co+torch.cat((-z[...,64:],z[...,:64]),-1)*si
        q=self.ktrans(rope(q),inv_t=True);k=self.kquant(self.ktrans(rope(k)));v=self.vquant(v)
        a=F.scaled_dot_product_attention(q.half(),k.half().repeat_interleave(2,1),v.half().repeat_interleave(2,1),is_causal=t>1)
        a=a.transpose(1,2).reshape(-1,16,128)
        a=(self.otrans.get_matrix().T.to(a)@a).reshape(b,t,h)
        o=self.linears['o'](a,qa_trans=[self.otrans.get_matrix(inv_t=True),self.vtrans.get_matrix(inv_t=True)])
        r=x+o
        z=self.trans['gate_up'](self.norm(r,self.norm2))
        gate=self.linears['gate'](z,qa_trans=self.trans['gate_up']);up=self.linears['up'](z,qa_trans=self.trans['gate_up'])
        a=F.silu(gate)*up
        return r+self.linears['down'](self.trans['down'](a),qa_trans=self.trans['down'])
    def export(self):
        state={}
        for n,prefix in PREFIXES.items():
            trans=self.trans[n];trans.to_eval_mode()
            for key in ('matrix_left','matrix_right','matrix_left_inv','matrix_right_inv','diag_scale'):state[prefix+'.'+key]=getattr(trans,key).detach().clone()
        for name,trans in [('self_attn.o_trans',self.otrans),('self_attn.kcache_trans',self.ktrans),('self_attn.vcache_trans',self.vtrans)]:
            trans.to_eval_mode()
            for key in ('matrix','matrix_inv_t'):state[name+'.'+key]=getattr(trans,key).detach().clone()
        return state


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1);args=parser.parse_args()
    target=BASE/'flatquant_training';target.mkdir(exist_ok=True)
    manifest_path=target/f'manifest_shard{args.shard}.json'
    if manifest_path.exists():raise FileExistsError('Training output exists')
    torch.manual_seed(0);np.random.seed(0);random.seed(0);torch.backends.cuda.matmul.allow_tf32=False
    tokens=torch.load(BASE/'train.pt',weights_only=True).cuda()
    with torch.no_grad():hidden=F.embedding(tokens,extras()['embedding'].cuda())
    logs=[];manifest=dict(status='running',model=str(MODEL),method='flatquant',epochs=15,batch_size=4,learning_rate=.005,
        optimizer='AdamW; cosine eta_min=5e-6; learned clipping lr=0.05',training=json.loads((BASE/'train.json').read_text()),
        implementation='Official FlatQuant SVD transforms/quantizers with local Qwen3 QK-norm/GQA reconstruction forward',
        lwc=True,lac=True,add_diag=True,source_sha256={str(p):sha(p) for p in [Path(__file__),Path('benchmarks/qwen3_flat_models.py'),ROOT/'benchmarks/cache/FlatQuant/flatquant/trans_utils.py',ROOT/'benchmarks/cache/FlatQuant/flatquant/quant_utils.py']},layers=[])
    for layer in range(28):
        print('FlatQuant train layer',layer+1,flush=True);start=time.perf_counter();w=load_weights(layer)
        with torch.no_grad():
            ref=QwenBlock(w);cache=ref.cache(1,2048);expected=torch.empty_like(hidden)
            for j in range(128):expected[j:j+1]=ref(hidden[j:j+1],cache)
            del ref,cache
        if layer%args.shards!=args.shard:
            hidden=expected
            continue
        torch.manual_seed(layer);np.random.seed(layer);random.seed(layer)
        block=TrainingBlock(w)
        normal=[];clipping=[]
        for name,p in block.named_parameters():
            if 'linear.weight' in name:p.requires_grad=False
            elif 'clip_factor' in name:clipping.append(p)
            else:normal.append(p)
        opt=torch.optim.AdamW([dict(params=normal,lr=.005),dict(params=clipping,lr=.05)])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=15*32,eta_min=5e-6)
        epochs=[]
        for epoch in range(15):
            total=0
            for j in range(32):
                opt.zero_grad()
                with torch.autocast('cuda',dtype=torch.float16):
                    actual=block(hidden[j*4:j*4+4]);loss=(actual.float()-expected[j*4:j*4+4].float()).square().mean()
                    objective=loss/loss.detach()
                if not torch.isfinite(loss):raise FloatingPointError((layer,epoch,j))
                objective.backward();opt.step();scheduler.step();total+=loss.item()
            epochs.append(total/32);print(' epoch',epoch+1,'mse',epochs[-1],'seconds',time.perf_counter()-start,flush=True)
        with torch.no_grad():state=cpu_tree(block.export())
        path=target/f'layer_{layer:02d}.pt';torch.save(state,path)
        logs.append(dict(layer=layer,seconds=time.perf_counter()-start,epochs_mse=epochs))
        manifest['layers'].append(dict(index=layer,file=path.name,sha256=sha(path)))
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n');(FLAT_REPORT/f'training_shard{args.shard}.json').write_text(json.dumps(logs,indent=2)+'\n')
        hidden=expected
        del block,w,opt,scheduler,actual,loss,objective;torch.cuda.empty_cache()
    manifest['status']='complete';manifest['shard']=args.shard;manifest['shards']=args.shards
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
