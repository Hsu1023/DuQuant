import unittest

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

from duquant_cuda.calibration import apply_smoothing, smoothing_scales
from duquant_cuda.decoder import LlamaDecoderBlock, random_rotations
from test_decoder import small_weights


def gqa_weights(kv_heads=1):
    weights = small_weights(512, 1024)
    for name in ("k", "v"):
        weights[f"self_attn.{name}_proj.weight"] = weights[
            f"self_attn.{name}_proj.weight"
        ][:kv_heads * 128].contiguous()
    return weights


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GQATests(unittest.TestCase):
    @torch.inference_mode()
    def test_fp16_hf_prefill_and_decode(self):
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        from transformers.cache_utils import DynamicCache

        torch.manual_seed(182)
        for kv_heads in (1, 2):
            weights = gqa_weights(kv_heads)
            config = LlamaConfig(
                hidden_size=512, intermediate_size=1024, num_attention_heads=4,
                num_key_value_heads=kv_heads, rms_norm_eps=1e-5,
            )
            config._attn_implementation = "sdpa"
            hf = LlamaDecoderLayer(config, 0).half().cuda().eval()
            hf.load_state_dict(weights, strict=True)
            fast = LlamaDecoderBlock(weights, {}, "fp16", max_position=32)
            x = torch.randn(2, 19, 512, device="cuda", dtype=torch.float16)
            cache, hf_cache = fast.cache(2, 32), DynamicCache()
            self.assertEqual(cache.k.shape, (2, kv_heads, 32, 128))
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                for start, end in ((0, 17), (17, 18), (18, 19)):
                    ids = torch.arange(start, end, device="cuda")
                    expected = hf(
                        x[:, start:end].contiguous(),
                        position_ids=ids[None], position_embeddings=(
                            fast.cos[start:end][None], fast.sin[start:end][None]
                        ), past_key_value=hf_cache, use_cache=True,
                        cache_position=ids,
                    )[0]
                    actual = fast(x[:, start:end].contiguous(), cache, start)
                    torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.004)

    @torch.inference_mode()
    def test_integer_reference_and_graph(self):
        torch.manual_seed(183)
        weights = gqa_weights()
        rotations = random_rotations(512, 1024, "cuda")
        fast = LlamaDecoderBlock(weights, rotations, "cuda", max_position=32)
        ref = LlamaDecoderBlock(weights, rotations, "integer_reference", max_position=32)
        x = torch.randn(2, 18, 512, device="cuda", dtype=torch.float16)
        a, b = fast.cache(2, 32), ref.cache(2, 32)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for start, end in ((0, 17), (17, 18)):
                token = x[:, start:end].contiguous()
                actual, expected = fast(token, a, start), ref(token, b, start)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                torch.testing.assert_close(a.k, b.k, atol=0, rtol=0)
                torch.testing.assert_close(a.v, b.v, atol=0, rtol=0)
            previous = a.k[:, :, :17].clone()
            for _ in range(3):
                fast(token, a, 17)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = fast(token, a, 17)
            token.add_(0.1)
            graph.replay()
            torch.testing.assert_close(captured, fast(token, a, 17), atol=0, rtol=0)
            torch.testing.assert_close(a.k[:, :, :17], previous, atol=0, rtol=0)

    @torch.inference_mode()
    def test_smoothing_ties_shared_value_channels(self):
        torch.manual_seed(184)
        original = gqa_weights()
        weights = {k: v.clone() for k, v in original.items()}
        maxima = {name: torch.rand(n, device="cuda", dtype=torch.float16) + 0.5
                  for name, n in (("qkv", 512), ("o", 512), ("gate_up", 512), ("down", 1024))}
        scales = smoothing_scales(weights, maxima, 0.6)
        grouped = scales["o"].reshape(1, 4, 128)
        torch.testing.assert_close(grouped, grouped[:, :1].expand_as(grouped), atol=0, rtol=0)
        apply_smoothing(weights, scales)
        a, b = LlamaDecoderBlock(original, {}, "fp16"), LlamaDecoderBlock(weights, {}, "fp16")
        x = torch.randn(1, 17, 512, device="cuda", dtype=torch.float16)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            expected, actual = a(x, a.cache(1, 17)), b(x, b.cache(1, 17))
        self.assertLess(((actual.float()-expected.float()).norm()/expected.float().norm()).item(), 0.002)
        scales["o"][0] *= 2
        with self.assertRaises(ValueError):
            apply_smoothing({k: v.clone() for k, v in original.items()}, scales)


if __name__ == "__main__":
    unittest.main()
