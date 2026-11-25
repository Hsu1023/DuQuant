"""Common decoder/quantizer for QuaRot, SpinQuant and DuQuant accuracy tests.

Method-specific basis changes follow the pinned official implementations.
The evaluation arithmetic, affine quantizer and GPTQ implementation are shared.
This is a controlled-format comparison, not a reproduction of paper defaults.
"""
import os
import importlib.util
import json
import math
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks/cache/qa_deps'))
from fast_hadamard_transform import hadamard_transform
from duquant_cuda.backend import DuQuantLinearCUDA
from duquant_cuda.decoder import LlamaDecoderBlock, qdq, load_layer_weights
from duquant_cuda.model import LlamaInferenceModel

QUAROT=Path(os.environ.get('DUQUANT_QUAROT', '/mnt/yichen/wyc/QDLM/QuaRot'))
spec=importlib.util.spec_from_file_location('comparison_hadamard',QUAROT/'fake_quant/hadamard_utils.py')
had=importlib.util.module_from_spec(spec)
spec.loader.exec_module(had)
MODEL=Path(os.environ.get('DUQUANT_LLAMA2', '/mnt/public_models/llama2-7b-hf'))
BASE=ROOT/'benchmarks/cache/qa_compare'
NAMES=('qkv','o','gate_up','down')


def identity_rotations(k,device):
    eye=torch.eye(128,device=device,dtype=torch.float16)
    return eye,torch.arange(k,device=device),eye


def basis(x,name,method):
    dtype=x.dtype
    if name=='down':
        h,k=had.get_hadK(x.shape[-1])
        return had.matmul_hadU_cuda(x.float(),h,k).to(dtype)
    if name=='o' and method=='quarot':
        shape=x.shape
        y=x.float().reshape(-1,32,128).transpose(1,2).contiguous()
        y=hadamard_transform(y,scale=1/math.sqrt(32)).transpose(1,2)
        return y.reshape(shape).to(dtype)
    return x


class BasisLinear(torch.nn.Module):
    def __init__(self,w,name,method,state=None,act_clip=None,mode='reference',collect=False,hessian=False):
        super().__init__()
        self.name,self.method,self.act_clip,self.mode=name,method,act_clip,mode
        self.samples=[] if collect else None
        self.hessian=torch.zeros((w.shape[1],w.shape[1]),device=w.device) if hessian else None
        self.tokens=0
        self.register_buffer('weight',w if state is None else
                             (state['codes'].half()-state['zero'][:,None])*state['scale'][:,None])
        if mode=='cuda' and state is not None:
            self.inner=DuQuantLinearCUDA(w,*identity_rotations(w.shape[1],w.device),
                                        quantized_weight=state,act_clip=act_clip)
            self.weight=None
    def forward(self,x):
        z=basis(x,self.name,self.method)
        if self.samples is not None:
            from search_ptq import sample_rows
            self.samples.append(sample_rows(z))
        if hasattr(self,'inner'):
            return self.inner(z.contiguous())
        if self.act_clip is not None:
            z=qdq(z,self.act_clip)
        if self.hessian is not None:
            flat=z.reshape(-1,z.shape[-1]).float()
            self.hessian.addmm_(flat.T,flat)
            self.tokens+=len(flat)
        return F.linear(z,self.weight)


def grouped_weights(w):
    return dict(qkv=torch.cat([w[f'self_attn.{n}_proj.weight'] for n in ('q','k','v')]),
                o=w['self_attn.o_proj.weight'],
                gate_up=torch.cat([w[f'mlp.{n}_proj.weight'] for n in ('gate','up')]),
                down=w['mlp.down_proj.weight'])


def make_block(w,method,cfg,plan=None,mode='reference',quantize_kv=True):
    block=LlamaDecoderBlock(w,{},'fp16',eps=cfg['rms_norm_eps'],
        max_position=cfg['max_position_embeddings'],rope_theta=cfg.get('rope_theta',10000.),kv_bits=16)
    block.mode=mode
    block.kv_bits=4 if quantize_kv else 16
    block.k_clip,block.v_clip=plan['kv_clips'] if plan else (.9,1.)
    for name,weight in grouped_weights(w).items():
        setattr(block,name,BasisLinear(weight,name,method,
            state=plan['weights'][name] if plan else None,
            act_clip=plan['act_clips'][name] if plan else None,mode=mode))
    return block


@torch.inference_mode()
def rotate_layer(w,r1,r2,method):
    # Match official FP16 boundaries: fold norm, rotate residual basis, then
    # rotate V/O and down-projection. FP64 is used for offline dense rotations.
    for name in ('q','k','v'):
        key=f'self_attn.{name}_proj.weight'
        fused=(w[key].double()*w['input_layernorm.weight'].double()).half()
        w[key]=(fused.double()@r1).half()
    for name in ('gate','up'):
        key=f'mlp.{name}_proj.weight'
        fused=(w[key].double()*w['post_attention_layernorm.weight'].double()).half()
        w[key]=(fused.double()@r1).half()
    for key in ('self_attn.o_proj.weight','mlp.down_proj.weight'):
        w[key]=(r1.T@w[key].double()).half()
    v=w['self_attn.v_proj.weight']
    w['self_attn.v_proj.weight']=(v.T.float().double().reshape(-1,32,128)@r2).reshape(v.T.shape).T.half().contiguous()
    o=w['self_attn.o_proj.weight']
    if method=='quarot':
        w['self_attn.o_proj.weight']=had.matmul_hadU_cuda(o.float(),None,1).half()
    else:
        w['self_attn.o_proj.weight']=(o.float().double().reshape(-1,32,128)@r2).reshape(o.shape).half()
    d=w['mlp.down_proj.weight']; hk,k=had.get_hadK(d.shape[1])
    w['mlp.down_proj.weight']=had.matmul_hadU_cuda(d.float(),hk,k).half()
    for key in ('input_layernorm.weight','post_attention_layernorm.weight'):
        w[key]=torch.ones_like(w[key])
    return w


@torch.inference_mode()
def rotated_extras(r1):
    index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    tensors={}
    for key in ('model.embed_tokens.weight','model.norm.weight','lm_head.weight'):
        with safe_open(str(MODEL/index[key]),framework='pt') as f:
            tensors[key]=f.get_tensor(key).half().cuda()
    emb=tensors['model.embed_tokens.weight'].double()
    emb=(emb-emb.mean(-1,keepdim=True)).half()
    head=(tensors['lm_head.weight'].double()*tensors['model.norm.weight'].double()).half()
    return dict(embedding=(emb.double()@r1).half().cpu(),
                final_norm=torch.ones_like(tensors['model.norm.weight']).cpu(),
                lm_head=(head.double()@r1).half().cpu())


@torch.inference_mode()
def load_model(method,mode='reference',unquantized=False):
    if method=='duquant':
        return LlamaInferenceModel.from_pretrained(MODEL,mode,
            calibration=ROOT/'benchmarks/cache/ptq_llama2_final/sym_gptq',kv_bits=4)
    if method=='fp16':
        model=LlamaInferenceModel.from_pretrained(MODEL,'fp16',kv_bits=16)
        for block in model.blocks:
            block.mode='reference'
        return model
    folder=BASE/method
    metadata=json.loads((folder/'manifest.json').read_text())
    if metadata['status']!='complete':
        raise ValueError('Incomplete comparator export')
    cfg=json.loads((MODEL/'config.json').read_text())
    blocks=[]
    for i in range(32):
        w=torch.load(folder/'rotated'/f'layer_{i:02d}.pt',map_location='cuda',weights_only=True)
        plan=None if unquantized else torch.load(folder/'gptq'/f'layer_{i:02d}.pt',map_location='cuda',weights_only=True)
        block=make_block(w,method,cfg,plan,mode,quantize_kv=not unquantized)
        if blocks:
            block.cos,block.sin,block.hadamard=blocks[0].cos,blocks[0].sin,blocks[0].hadamard
        blocks.append(block)
        del w,plan
    extra=torch.load(folder/'extras.pt',map_location='cuda',weights_only=True)
    return LlamaInferenceModel(blocks,**extra,eps=cfg['rms_norm_eps']).eval()
