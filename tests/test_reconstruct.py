import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from duquant_reconstruct import *
import unittest
from torch.nn.attention import SDPBackend,sdpa_kernel

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class ReconstructTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9);torch.backends.cuda.matmul.allow_tf32=False
        self.state=torch.load(BASE/'duquant/layer_00.pt',weights_only=True,map_location='cuda')
    def test_quantizer_forward_and_degenerate_gradients(self):
        from duquant_cuda.decoder import qdq
        x=torch.cat([torch.randn(8,128,device='cuda'),torch.zeros(1,128,device='cuda'),torch.randn(1,128,device='cuda')*1e-6]).half().requires_grad_()
        clip=torch.tensor(.825,device='cuda',requires_grad=True)
        y=activation_ste(x,clip)
        torch.testing.assert_close(y,qdq(x.detach(),float(clip.detach().half())),atol=0,rtol=0)
        y.float().square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all());self.assertTrue(torch.isfinite(clip.grad))
    def test_gradients_and_float_folding(self):
        block=Reconstruction(self.state)
        x=torch.randn(1,17,2048,device='cuda',dtype=torch.float16)*.1
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            y=block(x);loss=y.float().square().mean();loss.backward()
        for n,p in block.named_parameters():
            self.assertIsNotNone(p.grad,n);self.assertTrue(torch.isfinite(p.grad).all(),n)
        with torch.no_grad():
            # Isolate diagonal folding from the existing finite-precision
            # greedy rotation matrices' orthogonality error.
            for n in NAMES:
                getattr(block,f'{n}_rot0').copy_(torch.eye(128,device='cuda'))
                getattr(block,f'{n}_rot1').copy_(torch.arange(getattr(block,f'{n}_rot1').numel(),device='cuda'))
                getattr(block,f'{n}_rot2').copy_(torch.eye(128,device='cuda'))
            block.had.copy_(torch.eye(128,device='cuda'))
            for p in block.scale_logs.values():p.uniform_(-.15,.15)
            block.quantize=False
            w=block.scaled();s=block.export();ref=QwenBlock(self.state['weights'])
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):a=block(x);b=ref(x,ref.cache(1,17))
            rel=((a.float()-b.float()).norm()/b.float().norm()).item()
            print('Floating folding relative L2',rel);self.assertLess(rel,.01)
            # Export folding has intentionally rounded scales; check the actual
            # exported tensors rather than assuming identical training arithmetic.
            folded,rot=folded_base(0,s)
            for key in w:torch.testing.assert_close(folded[key],w[key],atol=0,rtol=0)

if __name__=='__main__':unittest.main()
