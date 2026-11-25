"""Independent scorer alignment and comparator integer arithmetic checks."""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
from eval_qa import conditional_scores
from comparison_models import BasisLinear, basis, identity_rotations
from duquant_cuda.decoder import LlamaDecoderBlock, IntegerReferenceLinear
from duquant_cuda.model import LlamaInferenceModel
from duquant_cuda.ptq import symmetric_rtn


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class QATests(unittest.TestCase):
    @torch.inference_mode()
    def test_scores_match_huggingface_and_padding(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.manual_seed(12)
        cfg = LlamaConfig(vocab_size=128, hidden_size=256, intermediate_size=512,
                          num_hidden_layers=2, num_attention_heads=2,
                          num_key_value_heads=2, rms_norm_eps=1e-5)
        cfg._attn_implementation = 'sdpa'
        hf = LlamaForCausalLM(cfg).half().cuda().eval()
        blocks = []
        for layer in hf.model.layers:
            block = LlamaDecoderBlock(dict(layer.state_dict()), {}, 'fp16', eps=cfg.rms_norm_eps, kv_bits=16)
            block.mode = 'reference'
            blocks.append(block)
        model = LlamaInferenceModel(blocks, hf.model.embed_tokens.weight,
                                    hf.model.norm.weight, hf.lm_head.weight, cfg.rms_norm_eps)
        requests = [dict(context=[1, 4, 8], continuation=[3, 6]),
                    dict(context=[7], continuation=[5]),
                    dict(context=[2, 9], continuation=[4, 8, 11, 15])]
        expected = []
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for r in requests:
                a, b = r['context'], r['continuation']
                logits = hf(torch.tensor([(a+b)[:-1]], device='cuda')).logits
                logp = logits[0, len(a)-1:].float().log_softmax(-1)
                expected.append(logp.gather(1, torch.tensor(b, device='cuda')[:, None]).double().sum().item())
            actual = conditional_scores(model, requests, 32)
            padded = conditional_scores(model, requests, 64)
        np.testing.assert_allclose(actual, expected, atol=.004, rtol=0)
        np.testing.assert_allclose(padded, actual, atol=.004, rtol=0)

    @torch.inference_mode()
    def test_comparator_int4_matches_integer_oracle(self):
        torch.manual_seed(13)
        for method, name, k in [('quarot', 'qkv', 256), ('quarot', 'o', 4096),
                                ('quarot', 'down', 11008), ('spinquant', 'o', 4096),
                                ('spinquant', 'down', 11008)]:
            w = torch.randn(128, k, device='cuda', dtype=torch.float16)*.03
            state, _ = symmetric_rtn(w, [.8, 1.])
            fast = BasisLinear(w, name, method, state, .85, 'cuda')
            oracle = IntegerReferenceLinear(w, identity_rotations(k, w.device),
                                            quantized_weight=state, act_clip=.85)
            for shape in [(1, 1, k), (2, 17, k)]:
                x = torch.randn(shape, device='cuda', dtype=torch.float16)
                torch.testing.assert_close(fast(x), oracle(basis(x, name, method)), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
