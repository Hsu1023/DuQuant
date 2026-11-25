"""FlatQuant transforms adapted to Qwen3 Q/K norm and GQA."""
from qwen3_models import *
from flatquant_models import (FlatLinear,FlatKVCache,PREFIXES,kron,transforms,transform_weights,_pack_transformed)
from comparison_models import grouped_weights
from duquant_cuda.decoder import qdq
from search_ptq import sample_rows
import triton
FOLDER=BASE/'flatquant'
FLAT_REPORT=REPORT/'flatquant'


def flat_prepare(block,packed,pos):
    b,t,_=packed.shape
    q,k,v=[z.reshape(b,t,heads,128).transpose(1,2) for z,heads in
           zip(packed.split((2048,1024,1024),-1),(16,8,8))]
    def norm(z,w):return (z.float()*torch.rsqrt(z.float().square().mean(-1,keepdim=True)+block.eps)).half()*w
    q,k=norm(q,block.qnorm),norm(k,block.knorm)
    co,si=block.cos[pos:pos+t],block.sin[pos:pos+t]
    def rope(z):return z*co+torch.cat((-z[...,64:],z[...,:64]),-1)*si
    return (rope(q)@block.qmat).contiguous(),(rope(k)@block.kmat).contiguous(),v.contiguous()


class Cache(FlatKVCache):
    def prepare(self,packed,cos,sin,position):
        q,k,v=flat_prepare(self.block,packed,position)
        kd,vd=torch.empty_like(k),torch.empty_like(v);rows=k.numel()//128
        _pack_transformed[(triton.cdiv(rows,16),)](k,v,self.k,self.v,self.k_params,self.v_params,kd,vd,
            position,k.shape[2],rows,self.capacity,self.k_clip,self.v_clip,16,enable_fp_fusion=False)
        return q,kd,vd


class Block(QwenBlock):
    def cache(self,batch,capacity):
        if self.mode=='cuda' and self.kv_bits==4:
            c=Cache(batch,8,16,capacity,self.norm1.device,self.k_clip,self.v_clip);c.block=self;return c
        return super().cache(batch,capacity)
    def _qkv_reference(self,packed,cache,pos):
        q,k,v=flat_prepare(self,packed,pos)
        if hasattr(self,'k_samples'):
            self.k_samples.append(sample_rows(k));self.v_samples.append(sample_rows(v))
        if self.kv_bits==4:k,v=qdq(k,self.k_clip),qdq(v,self.v_clip)
        cache.k[:,:,pos:pos+k.shape[2]].copy_(k);cache.v[:,:,pos:pos+v.shape[2]].copy_(v)
        return q


def make_block(w,trans,plan=None,mode='reference',quantize_kv=True):
    block=Block(w)
    block.mode,block.kv_bits=mode,4 if quantize_kv else 16
    block.k_clip,block.v_clip=plan['kv_clips'] if plan else (.9,1.)
    block.register_buffer('qmat',trans['q']);block.register_buffer('kmat',trans['k'])
    for n,weight in grouped_weights(w).items():
        setattr(block,n,FlatLinear(weight,trans[n],plan['weights'][n] if plan else None,
                                  plan['act_clips'][n] if plan else None,mode))
    return block


@torch.inference_mode()
def load_flat_model(mode='cuda'):
    m=json.loads((FOLDER/'manifest.json').read_text());assert m['status']=='complete' and len(m['layers'])==28
    blocks=[]
    for i in range(28):
        state=torch.load(FOLDER/'rotated'/f'layer_{i:02d}.pt',weights_only=True,map_location='cuda')
        plan=torch.load(FOLDER/'gptq'/f'layer_{i:02d}.pt',weights_only=True,map_location='cuda')
        blocks.append(make_block(state['weights'],state['transforms'],plan,mode))
    return make_model(blocks,torch.load(FOLDER/'extras.pt',weights_only=True,map_location='cpu'))
