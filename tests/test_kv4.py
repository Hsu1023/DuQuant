import unittest

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from duquant_cuda.backend import affine_quant, pack
from duquant_cuda.decoder import LlamaDecoderBlock, random_rotations
from duquant_cuda.kv4 import PackedKVCache
from test_gqa import gqa_weights


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class KV4Tests(unittest.TestCase):
    @torch.inference_mode()
    def test_fused_pack_rope_and_append(self):
        torch.manual_seed(291)
        for heads in (1, 2, 4):
            ref = LlamaDecoderBlock(gqa_weights(heads), random_rotations(512, 1024, "cuda"),
                                    "integer_reference", max_position=1024, kv_bits=4)
            cache = PackedKVCache(2, heads, 4, 137, "cuda")
            float_cache = ref.cache(2, 137)
            for pos, count in ((0, 129), (129, 1), (130, 1)):
                x = torch.randn(2, count, 512+2*heads*128, device="cuda", dtype=torch.float16)
                q, kt, vt = cache.prepare(x, ref.cos, ref.sin, pos)
                expected = ref._qkv_reference(x, float_cache, pos)
                torch.testing.assert_close(q, expected, rtol=0, atol=0)
                k, v = cache.unpack(pos+count)
                torch.testing.assert_close(k, float_cache.k[:, :, :pos+count], rtol=0, atol=0)
                torch.testing.assert_close(v, float_cache.v[:, :, :pos+count], rtol=0, atol=0)
                if count>1:
                    torch.testing.assert_close(kt, k, rtol=0, atol=0)
                    torch.testing.assert_close(vt, v, rtol=0, atol=0)
            self.assertEqual(cache.storage_bytes, 2*heads*137*(128+8))
            self.assertEqual(cache.k.dtype, torch.uint8)
            self.assertEqual(cache.k.shape[-1], 64)

    @torch.inference_mode()
    def test_decode_tail_masks_gqa_and_graph(self):
        torch.manual_seed(292)
        for heads in (1, 2, 4):
            cache = PackedKVCache(2, heads, 4, 1027, "cuda")
            for dst, params in ((cache.k, cache.k_params), (cache.v, cache.v_params)):
                x = torch.randn(2, heads, 513, 128, device="cuda", dtype=torch.float16)
                x[:, :, 0] = 0
                x[:, :, 1] = 3
                x[:, :, 2] = -3
                codes, scale, zero = affine_quant(x, 1.)
                dst[:, :, :513].copy_(pack(codes))
                params[:, :, :513, 0].copy_(scale)
                params[:, :, :513, 1].copy_(zero)
            q = torch.randn(2, 4, 1, 128, device="cuda", dtype=torch.float16)
            for length in (1, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257, 513):
                k, v = cache.unpack(length)
                actual = cache.decode(q, length)
                with sdpa_kernel(SDPBackend.MATH):
                    expected = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), enable_gqa=heads!=4).half()
                torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.001)
            for _ in range(3):
                cache.decode(q, 513)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = cache.decode(q, 513)
            q.add_(0.25)
            graph.replay()
            torch.testing.assert_close(output, cache.decode(q, 513), rtol=0, atol=0)

    @torch.inference_mode()
    def test_full_blocks_and_kv16(self):
        torch.manual_seed(293)
        for heads in (1, 4):
            for bits in (4, 16):
                w = gqa_weights(heads)
                r = random_rotations(512, 1024, "cuda")
                fast = LlamaDecoderBlock(w, r, "cuda", max_position=256, kv_bits=bits)
                ref = LlamaDecoderBlock(w, r, "integer_reference", max_position=256, kv_bits=bits)
                a, b = fast.cache(2, 132), ref.cache(2, 132)
                x = torch.randn(2, 132, 512, device="cuda", dtype=torch.float16)
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    for start, end in ((0, 129), (129, 130), (130, 131), (131, 132)):
                        token = x[:, start:end].contiguous()
                        y, z = fast(token, a, start), ref(token, b, start)
                        error = (y.float()-z.float()).norm()/z.float().norm()
                        self.assertLess(error.item(), 0.003)
                        if bits == 16:
                            torch.testing.assert_close(a.k, b.k, atol=0, rtol=0)
                            torch.testing.assert_close(a.v, b.v, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
