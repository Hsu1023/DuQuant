"""GPU numerical tests. python -m unittest discover -s tests -v"""

import unittest
import torch
import triton
from quantize.quantizer import UniformAffineQuantizer
from duquant_cuda.backend import (
    DuQuantLinearCUDA,
    affine_quant,
    pack,
    transform,
    _quant,
    _rotate,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CUDAKernelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_grad_enabled(False)

    def test_quant_exact(self):
        # Includes non-power-of-two K, zero range, one-sided range and clipping.
        for k in (128, 384, 4096, 11008):
            x = torch.randn(8, k, device="cuda", dtype=torch.float16)
            x[0].zero_()
            x[1].fill_(1)
            x[2].fill_(-1)
            x[3].abs_()
            x[4] = -x[4].abs()
            for clip in (1.0, 0.9):
                q, s, z = affine_quant(x, clip)
                original = UniformAffineQuantizer(
                    n_bits=4,
                    dynamic_method="per_token",
                    quant_method="duquant",
                    rotate=False,
                    lac=clip,
                )
                original_output = original(x)
                self.assertTrue(torch.equal(original.scale.squeeze(-1), s))
                self.assertTrue(torch.equal(original.round_zero_point.squeeze(-1), z))
                self.assertTrue(
                    torch.equal(original_output, (q.half() - z[:, None]) * s[:, None])
                )
                a = torch.empty((8, k // 2), device="cuda", dtype=torch.uint8)
                gs = torch.empty(8, device="cuda", dtype=torch.float16)
                gz = torch.empty_like(gs)
                sums = torch.empty(8, device="cuda", dtype=torch.int32)
                _quant[(8,)](x, a, gs, gz, sums, 8, k, clip, triton.next_power_of_2(k))
                self.assertTrue(torch.equal(a, pack(q)), (k, clip, "packed"))
                self.assertTrue(torch.equal(gs, s), (k, clip, "scale"))
                self.assertTrue(torch.equal(gz, z), (k, clip, "zero"))
                self.assertTrue(torch.equal(sums, q.int().sum(-1)), (k, clip, "sum"))

    def test_integer_gemm_and_gemv(self):
        # Identity rotations isolate packing, INT4 MMA, zero-point correction,
        # padding, output shapes and bias from rotation rounding differences.
        for k in (128, 384, 4096):
            n = 96
            w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.03
            r = torch.eye(128, device="cuda", dtype=torch.float16)
            perm = torch.arange(k, device="cuda")
            bias = torch.randn(n, device="cuda", dtype=torch.float16) * 0.01
            layer = DuQuantLinearCUDA(w, r, perm, r, bias)
            qw, sw, zw = affine_quant(w, 0.8)
            for shape in ((1, k), (2, 3, k), (17, k)):
                x = torch.randn(shape, device="cuda", dtype=torch.float16)
                qa, sa, za = affine_quant(x, 0.9)
                ref = ((qa.float() - za[..., None].float()) * sa[..., None].float()) @ (
                    (qw.float() - zw[:, None].float()) * sw[:, None].float()
                ).T + bias.float()
                if x.numel() // k == 1:
                    torch.testing.assert_close(
                        layer(x), ref.half(), atol=0.002, rtol=0.002
                    )
                else:
                    for fused in (False, True):
                        layer.fused_epilogue = fused
                        for config in (0, 1, 2):
                            layer.gemm_config = config
                            torch.testing.assert_close(
                                layer(x), ref.half(), atol=0.002, rtol=0.002
                            )

    def test_rotations_batch_and_layout(self):
        k = 384
        m = 35
        r = (
            torch.linalg.qr(torch.randn(2, 128, 128, device="cuda"))[0]
            .half()
            .contiguous()
        )
        perm = torch.randperm(k, device="cuda")
        x = torch.randn(5, 7, k, device="cuda", dtype=torch.float16)
        expected = transform(x, r[0], perm, r[1])
        for trans in (False, True):
            for bm, warps in ((16, 4), (32, 8), (64, 8), (128, 8)):
                tmp = torch.empty_like(x)
                y = torch.empty_like(x)
                grid = (triton.cdiv(m, bm), k // 128)
                _rotate[grid](
                    x, r[0], perm, tmp, m, k, False, bm, trans, num_warps=warps
                )
                _rotate[grid](
                    tmp, r[1], perm, y, m, k, True, bm, trans, num_warps=warps
                )
                torch.testing.assert_close(y, expected, atol=0.004, rtol=0.004)

    def test_stream_and_graph(self):
        k = 128
        n = 64
        r = torch.eye(128, device="cuda", dtype=torch.float16)
        w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.03
        layer = DuQuantLinearCUDA(w, r, torch.arange(k, device="cuda"), r)
        x = torch.randn(2, 3, k, device="cuda", dtype=torch.float16)
        baseline = layer(x)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                layer(x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                y = layer(x)
            graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(y, baseline)
        x.add_(0.5)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(y, layer(x))


if __name__ == "__main__":
    unittest.main()
