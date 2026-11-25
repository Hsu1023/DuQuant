"""FlatQuant transform equivalence and actual integer/packed-cache checks."""
import sys
from pathlib import Path
import unittest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
from flatquant_models import (MODEL, CHECKPOINT, transforms, transform_weights,
    make_block, FlatLinear, prepare_qkv, grouped_weights, kron)
from duquant_cuda.decoder import LlamaDecoderBlock, load_layer_weights, qdq
from duquant_cuda.ptq import symmetric_rtn


@unittest.skipUnless(torch.cuda.is_available() and CHECKPOINT.is_file() and
                     (MODEL/'config.json').is_file(),
                     'CUDA and the local LLaMA-2/FlatQuant fixtures are required')
class FlatQuantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(73)
        cls.original, cls.cfg = load_layer_weights(MODEL, 0)
        cls.state = torch.load(CHECKPOINT, weights_only=True, map_location='cuda')[0]
        cls.trans = transforms(cls.state)
        cls.weights = transform_weights(cls.original, cls.state)
        cls.plan = dict(weights={}, act_clips={}, kv_clips=[.925, .95])
        for name, w in grouped_weights(cls.weights).items():
            cls.plan['weights'][name], _ = symmetric_rtn(w, [.8])
            cls.plan['act_clips'][name] = .8

    def test_float_layer_invariance(self):
        ref = LlamaDecoderBlock(self.original, {}, 'fp16', kv_bits=16)
        ref.mode = 'reference'
        flat = make_block(self.weights, self.trans, self.cfg, quantize_kv=False)
        x = torch.randn(2, 17, 4096, device='cuda', dtype=torch.float16)*.1
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            a, b = ref(x, ref.cache(2,17)), flat(x, flat.cache(2,17))
        rel = ((a.float()-b.float()).norm()/a.float().norm()).item()
        print('Float layer relative L2:', rel)
        self.assertLess(rel, .01)

    def test_linear_integer_oracle(self):
        for name, w in grouped_weights(self.weights).items():
            # A subset of output rows keeps the independent dense oracle small.
            state = {k:v[:128].contiguous() for k,v in self.plan['weights'][name].items()}
            fast = FlatLinear(w[:128], self.trans[name], state, .8, 'cuda')
            ref = FlatLinear(w[:128], self.trans[name], state, .8, 'integer_reference')
            for t in (1, 17):
                x = torch.randn(2,t,w.shape[1],device='cuda',dtype=torch.float16)
                torch.testing.assert_close(fast(x),ref(x),atol=0,rtol=0)

    def test_nonorthogonal_packed_cache(self):
        block = make_block(self.weights, self.trans, self.cfg, self.plan, 'cuda')
        for t in (1, 17, 33):
            packed = torch.randn(2,t,3*4096,device='cuda',dtype=torch.float16)
            cache = block.cache(2, t+1)
            q, kd, vd = cache.prepare(packed,block.cos,block.sin,1)
            qr,k,v = prepare_qkv(packed,block.cos,block.sin,1,block.qmat,block.kmat)
            torch.testing.assert_close(q,qr,atol=0,rtol=0)
            torch.testing.assert_close(kd,qdq(k,block.k_clip),atol=0,rtol=0)
            torch.testing.assert_close(vd,qdq(v,block.v_clip),atol=0,rtol=0)
            ku,vu = cache.unpack(t+1)
            torch.testing.assert_close(ku[:,:,1:],kd,atol=0,rtol=0)
            torch.testing.assert_close(vu[:,:,1:],vd,atol=0,rtol=0)

    def test_integer_block_matches_independent_reference(self):
        fast = make_block(self.weights,self.trans,self.cfg,self.plan,'cuda')
        ref = make_block(self.weights,self.trans,self.cfg,self.plan,'integer_reference')
        x = torch.randn(2,17,4096,device='cuda',dtype=torch.float16)*.1
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            a,b = fast(x,fast.cache(2,17)), ref(x,ref.cache(2,17))
        torch.testing.assert_close(a,b,atol=0,rtol=0)


if __name__ == '__main__':
    unittest.main()
