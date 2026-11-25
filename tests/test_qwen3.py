import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from qwen3_models import *
import unittest
import torch.nn.functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class QwenTests(unittest.TestCase):
    @torch.inference_mode()
    def test_qknorm_gqa_block_matches_hf(self):
        from transformers import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        torch.manual_seed(4)
        cfg=Qwen3Config(**CFG);cfg._attn_implementation='sdpa'
        w=load_weights(0)
        hf=Qwen3DecoderLayer(cfg,0).half().cuda().eval();hf.load_state_dict(w)
        block=QwenBlock(w)
        for b,t in [(1,1),(2,17),(1,65)]:
            x=torch.randn(b,t,2048,device='cuda',dtype=torch.float16)*.1
            co=block.cos[:t][None].expand(b,-1,-1);si=block.sin[:t][None].expand(b,-1,-1)
            mask=torch.full((t,t),float('-inf'),device='cuda').triu(1)[None,None]
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                a=hf(x,attention_mask=None,position_embeddings=(co,si))[0]
                z=block(x,block.cache(b,t))
            err=((a.float()-z.float()).norm()/a.float().norm()).item()
            print('Qwen3 HF relative L2',b,t,err)
            self.assertLess(err,.005)


if __name__=='__main__':unittest.main()
