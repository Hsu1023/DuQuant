import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from llama_reconstruct import *
import unittest
from torch.nn.attention import SDPBackend,sdpa_kernel

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class LlamaReconstructTests(unittest.TestCase):
    def test_gradients_and_export_fold(self):
        torch.manual_seed(6);torch.backends.cuda.matmul.allow_tf32=False
        state=base_state(0);block=LlamaReconstruction(state);x=torch.randn(1,17,4096,device='cuda',dtype=torch.float16)*.1
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):y=block(x);y.float().square().mean().backward()
        for n,p in block.named_parameters():self.assertIsNotNone(p.grad,n);self.assertTrue(torch.isfinite(p.grad).all(),n)
        with torch.no_grad():
            for p in block.scale_logs.values():p.uniform_(-.1,.1)
            w,rot=folded_base(0,block.export());expected=block.scaled()
            for n in w:torch.testing.assert_close(w[n],expected[n],atol=0,rtol=0)

if __name__=='__main__':unittest.main()
