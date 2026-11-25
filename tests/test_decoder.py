import unittest
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from duquant_cuda.decoder import LlamaDecoderBlock, random_rotations


def small_weights(h=256, i=512):
    w = {}
    for p in ("q", "k", "v", "o"):
        w[f"self_attn.{p}_proj.weight"] = (
            torch.randn(h, h, device="cuda", dtype=torch.float16) * 0.03
        )
    for p in ("gate", "up"):
        w[f"mlp.{p}_proj.weight"] = (
            torch.randn(i, h, device="cuda", dtype=torch.float16) * 0.03
        )
    w["mlp.down_proj.weight"] = (
        torch.randn(h, i, device="cuda", dtype=torch.float16) * 0.03
    )
    for p in ("input_layernorm", "post_attention_layernorm"):
        w[p + ".weight"] = torch.rand(h, device="cuda", dtype=torch.float16) + 0.5
    return w


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        torch.set_grad_enabled(False)
        self.weights = small_weights()
        self.rotations = random_rotations(256, 512, "cuda")

    def test_fp16_matches_huggingface(self):
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        config = LlamaConfig(
            hidden_size=256,
            intermediate_size=512,
            num_attention_heads=2,
            num_key_value_heads=2,
            rms_norm_eps=1.0e-5,
        )
        config._attn_implementation = "sdpa"
        hf = LlamaDecoderLayer(config, 0).half().cuda().eval()
        hf.load_state_dict(self.weights, strict=True)
        fast = LlamaDecoderBlock(self.weights, self.rotations, "fp16")
        x = torch.randn(2, 9, 256, device="cuda", dtype=torch.float16)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            expected = hf(
                x,
                position_ids=torch.arange(9, device="cuda")[None, :],
                position_embeddings=(
                    fast.cos[:9][None].expand(2, -1, -1),
                    fast.sin[:9][None].expand(2, -1, -1),
                ),
            )[0]
            actual = fast(x, fast.cache(2, 16))
        torch.testing.assert_close(actual, expected, rtol=0.004, atol=0.004)

    def test_attention_preparation(self):
        from duquant_cuda.decoder import _rope_cache
        import triton

        block = LlamaDecoderBlock(self.weights, self.rotations, "reference")
        x = torch.randn(2, 9, 768, device="cuda", dtype=torch.float16)
        refcache = block.cache(2, 16)
        fastcache = block.cache(2, 16)
        expected = block._qkv_reference(x, refcache, 0)
        q = torch.empty_like(expected)
        _rope_cache[(triton.cdiv(2 * 9 * 2, 16),)](
            x,
            block.cos,
            block.sin,
            q,
            fastcache.k,
            fastcache.v,
            9,
            256,
            2,
            2,
            16,
            0,
            True,
            16,
            enable_fp_fusion=False,
        )
        torch.testing.assert_close(q, expected, atol=0.002, rtol=0.002)
        torch.testing.assert_close(fastcache.k, refcache.k, atol=0.002, rtol=0.002)
        torch.testing.assert_close(fastcache.v, refcache.v, atol=0, rtol=0)

    def test_full_block_fake_reference(self):
        fast = LlamaDecoderBlock(self.weights, self.rotations, "cuda")
        ref = LlamaDecoderBlock(self.weights, self.rotations, "reference")
        integer_ref = LlamaDecoderBlock(
            self.weights, self.rotations, "integer_reference"
        )
        for b, t in ((1, 1), (2, 9)):
            x = torch.randn(b, t, 256, device="cuda", dtype=torch.float16)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                actual = fast(x, fast.cache(b, t))
                expected = ref(x, ref.cache(b, t))
            rel = (actual.float() - expected.float()).norm() / expected.float().norm()
            # The integer epilogue skips per-element FP16 dequantization
            # rounding. Requantization at later projections amplifies boundary
            # crossings; isolated kernels are checked separately above.
            self.assertLess(rel.item(), 0.02)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                expected_integer = integer_ref(x, integer_ref.cache(b, t))
            integer_rel = (
                actual.float() - expected_integer.float()
            ).norm() / expected_integer.float().norm()
            self.assertEqual(integer_rel.item(), 0.0)
            torch.testing.assert_close(actual, expected_integer, atol=0, rtol=0)

    def test_decode_cache_and_graph(self):
        fast = LlamaDecoderBlock(self.weights, self.rotations, "cuda")
        x = torch.randn(2, 7, 256, device="cuda", dtype=torch.float16)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            fullcache = fast.cache(2, 7)
            full = fast(x, fullcache)
            cache = fast.cache(2, 7)
            outputs = []
            for pos in range(7):
                outputs.append(fast(x[:, pos : pos + 1].contiguous(), cache, pos))
            decoded = torch.cat(outputs, 1)
            self.assertLess(
                ((decoded.float() - full.float()).norm() / full.float().norm()).item(),
                0.015,
            )
            torch.testing.assert_close(cache.k, fullcache.k, atol=0.004, rtol=0.004)
            torch.testing.assert_close(cache.v, fullcache.v, atol=0.004, rtol=0.004)
            previous = cache.k[:, :, :6].clone()
            token = x[:, 6:7].contiguous()
            for _ in range(3):
                fast(token, cache, 6)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = fast(token, cache, 6)
            token.add_(0.4)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, fast(token, cache, 6))
            torch.testing.assert_close(cache.k[:, :, :6], previous, atol=0, rtol=0)
        with self.assertRaises(ValueError):
            fast(x, cache, 1)


if __name__ == "__main__":
    unittest.main()
