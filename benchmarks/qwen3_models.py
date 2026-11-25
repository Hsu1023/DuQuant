"""Qwen3-1.7B adaptation; existing LLaMA experiments are left intact."""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks/cache/qwen3_deps'))
sys.path.insert(0,str(ROOT))
import json
import torch
from safetensors import safe_open
from duquant_cuda.decoder import LlamaDecoderBlock, qdq
from duquant_cuda.model import LlamaInferenceModel
from flatquant_models import FlatKVCache
from comparison_models import had

MODEL=Path(os.environ.get('DUQUANT_QWEN3','/mnt/public_models/qwen3-1.7b-base'))
BASE=ROOT/'benchmarks/cache/qwen3_compare'
REPORT=ROOT/'benchmarks/qwen3_compare'
CFG=json.loads((MODEL/'config.json').read_text())


class Basis(torch.nn.Module):
    def __init__(self,inner,name,method):
        super().__init__();self.inner,self.name,self.method=inner,name,method
    def forward(self,x):
        if self.name=='down':
            hk,k=had.get_hadK(x.shape[-1]);x=had.matmul_hadU_cuda(x.float(),hk,k).half()
        elif self.name=='o' and self.method=='quarot':
            from fast_hadamard_transform import hadamard_transform
            shape=x.shape
            x=hadamard_transform(x.float().reshape(-1,16,128).transpose(1,2).contiguous(),scale=16**-.5).transpose(1,2).reshape(shape).half()
        return self.inner(x.contiguous())


def wrap_basis(block,method):
    if method in ('quarot','spinquant'):
        for name in ('qkv','o','gate_up','down'):
            setattr(block,name,Basis(getattr(block,name),name,method))
    return block


def load_weights(i,device='cuda'):
    prefix=f'model.layers.{i}.'
    with safe_open(str(MODEL/'model.safetensors'),framework='pt') as f:
        return {k[len(prefix):]:f.get_tensor(k).half().to(device) for k in f.keys() if k.startswith(prefix)}


def extras():
    with safe_open(str(MODEL/'model.safetensors'),framework='pt') as f:
        emb=f.get_tensor('model.embed_tokens.weight').half()
        head=f.get_tensor('lm_head.weight').half() if 'lm_head.weight' in f.keys() else emb
        return dict(embedding=emb,final_norm=f.get_tensor('model.norm.weight').half(),lm_head=head)


def prepare(block,packed,pos):
    b,t,_=packed.shape
    q,k,v=[z.reshape(b,t,heads,128).transpose(1,2) for z,heads in
           zip(packed.split((block.h,block.kv_h,block.kv_h),-1),(block.heads,block.kv_heads,block.kv_heads))]
    def norm(z,w):return (z.float()*torch.rsqrt(z.float().square().mean(-1,keepdim=True)+block.eps)).half()*w
    q,k=norm(q,block.qnorm),norm(k,block.knorm)
    co,si=block.cos[pos:pos+t],block.sin[pos:pos+t]
    def rope(z):return z*co+torch.cat((-z[...,64:],z[...,:64]),-1)*si
    q,k=rope(q),rope(k)
    if block.kv_bits!=16:q,k=q@block.hadamard,k@block.hadamard
    return q.contiguous(),k.contiguous(),v.contiguous()


class QwenCache(FlatKVCache):
    def prepare(self,packed,cos,sin,position):
        from flatquant_models import _pack_transformed
        import triton
        q,k,v=prepare(self.block,packed,position)
        kd,vd=torch.empty_like(k),torch.empty_like(v)
        rows=k.numel()//128
        _pack_transformed[(triton.cdiv(rows,16),)](k,v,self.k,self.v,self.k_params,self.v_params,kd,vd,
            position,k.shape[2],rows,self.capacity,self.k_clip,self.v_clip,16,enable_fp_fusion=False)
        return q,kd,vd


class QwenBlock(LlamaDecoderBlock):
    def __init__(self,w,rotations=None,mode='fp16',plan=None,kv_bits=16):
        super().__init__(w,rotations or {},mode,eps=CFG['rms_norm_eps'],max_position=4096,
            rope_theta=CFG['rope_theta'],quantization=plan,kv_bits=kv_bits)
        self.register_buffer('qnorm',w['self_attn.q_norm.weight'])
        self.register_buffer('knorm',w['self_attn.k_norm.weight'])
        # Q/K normalization requires the reference preparation even for FP16.
        if mode=='fp16':self.mode='reference'
    def cache(self,batch,capacity):
        if self.mode=='cuda' and self.kv_bits==4:
            c=QwenCache(batch,self.kv_heads,self.heads,capacity,self.norm1.device,self.k_clip,self.v_clip)
            c.block=self
            return c
        return super().cache(batch,capacity)
    def _qkv_reference(self,packed,cache,pos):
        q,k,v=prepare(self,packed,pos)
        if hasattr(self,'k_samples'):
            from search_ptq import sample_rows
            self.k_samples.append(sample_rows(k));self.v_samples.append(sample_rows(v))
        if self.kv_bits!=16:k,v=qdq(k,self.k_clip),qdq(v,self.v_clip)
        cache.k[:,:,pos:pos+k.shape[2]].copy_(k)
        cache.v[:,:,pos:pos+v.shape[2]].copy_(v)
        return q


def make_model(blocks,extra=None):
    extra=extras() if extra is None else extra
    for k,v in extra.items():extra[k]=v.cuda()
    return LlamaInferenceModel(blocks,**extra,eps=CFG['rms_norm_eps']).eval()


@torch.inference_mode()
def load_model(method,mode='cuda'):
    if method=='fp16':return make_model([QwenBlock(load_weights(i)) for i in range(CFG['num_hidden_layers'])])
    folder=BASE/method
    meta=json.loads((folder/'manifest.json').read_text());assert meta['status']=='complete'
    blocks=[]
    for i in range(CFG['num_hidden_layers']):
        state=torch.load(folder/f'layer_{i:02d}.pt',map_location='cuda',weights_only=True)
        block=QwenBlock(state['weights'],state['rotations'],mode,state['plan'],4)
        blocks.append(wrap_basis(block,method))
    return make_model(blocks,torch.load(folder/'extras.pt',map_location='cpu',weights_only=True))
