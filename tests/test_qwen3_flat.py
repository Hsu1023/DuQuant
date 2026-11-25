import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from qwen3_flat_models import *
from duquant_cuda.ptq import symmetric_rtn
import unittest
from torch.nn.attention import SDPBackend,sdpa_kernel

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class QwenFlatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_grad_enabled(False);torch.manual_seed(5);torch.backends.cuda.matmul.allow_tf32=False
        cls.original=load_weights(0)
        state=torch.load(BASE/'flatquant_training/layer_00.pt',weights_only=True,map_location='cuda')
        cls.weights=transform_weights(cls.original,state);cls.trans=transforms(state)
        cls.plan=dict(weights={},act_clips={},kv_clips=[.925,.95])
        for n,w in grouped_weights(cls.weights).items():
            cls.plan['weights'][n],_=symmetric_rtn(w,[.8]);cls.plan['act_clips'][n]=.85
    def test_float_transform_equivalence(self):
        ref=QwenBlock(self.original);flat=make_block(self.weights,self.trans,quantize_kv=False)
        for b,t in [(1,1),(2,17)]:
            x=torch.randn(b,t,2048,device='cuda',dtype=torch.float16)*.1
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):a,bv=ref(x,ref.cache(b,t)),flat(x,flat.cache(b,t))
            err=((a.float()-bv.float()).norm()/a.float().norm()).item();print('Flat Qwen relative L2:',err)
            self.assertLess(err,.01)
    def test_full_integer_block(self):
        fast=make_block(self.weights,self.trans,self.plan,'cuda');ref=make_block(self.weights,self.trans,self.plan,'integer_reference')
        x=torch.randn(2,17,2048,device='cuda',dtype=torch.float16)*.1
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):a,b=fast(x,fast.cache(2,17)),ref(x,ref.cache(2,17))
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    def test_gqa_cache_and_decode_append(self):
        block=make_block(self.weights,self.trans,self.plan,'cuda');cache=block.cache(2,20)
        for pos,t in [(0,17),(17,1)]:
            packed=torch.randn(2,t,4096,device='cuda',dtype=torch.float16)
            q,k,v=flat_prepare(block,packed,pos);actual,kd,vd=cache.prepare(packed,block.cos,block.sin,pos)
            torch.testing.assert_close(q,actual,atol=0,rtol=0)
            torch.testing.assert_close(kd,qdq(k,block.k_clip),atol=0,rtol=0)
            torch.testing.assert_close(vd,qdq(v,block.v_clip),atol=0,rtol=0)
            ku,vu=cache.unpack(pos+t)
            torch.testing.assert_close(ku[:,:,pos:],kd,atol=0,rtol=0)
            torch.testing.assert_close(vu[:,:,pos:],vd,atol=0,rtol=0)

if __name__=='__main__':unittest.main()
