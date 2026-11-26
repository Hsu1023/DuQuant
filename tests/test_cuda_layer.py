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
        self._grad = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

    def tearDown(self):
        # Do not leak no-grad mode into other test modules (unittest discover).
        torch.set_grad_enabled(self._grad)

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
                        for config in range(layer.num_gemm_configs):
                            layer.gemm_config = config
                            torch.testing.assert_close(
                                layer(x), ref.half(), atol=0.002, rtol=0.002
                            )

    def test_unpacked_quant_matches_packed(self):
        # Hopper consumes s8 codes; they must be the same codes as the packed path.
        for k in (128, 384, 4096):
            x = torch.randn(9, k, device="cuda", dtype=torch.float16)
            outs = []
            for unpacked in (False, True):
                a = torch.empty(
                    (9, k) if unpacked else (9, k // 2),
                    device="cuda",
                    dtype=torch.int8 if unpacked else torch.uint8,
                )
                gs = torch.empty(9, device="cuda", dtype=torch.float16)
                gz = torch.empty_like(gs)
                sums = torch.empty(9, device="cuda", dtype=torch.int32)
                _quant[(9,)](x, a, gs, gz, sums, 9, k, 0.9, triton.next_power_of_2(k), unpacked)
                outs.append((a, gs, gz, sums))
            (p, *meta_p), (u, *meta_u) = outs
            self.assertTrue(torch.equal(pack(u.to(torch.uint8)), p), k)
            for left, right in zip(meta_p, meta_u):
                self.assertTrue(torch.equal(left, right), k)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
        "Hopper only",
    )
    def test_sm90_integer_gemm_exact(self):
        from duquant_cuda.backend import load_gemm_sm90

        lib = load_gemm_sm90()
        for m, n, k in ((2, 96, 128), (17, 96, 384), (300, 4096, 4096), (64, 11008, 4096), (1000, 4096, 11008)):
            qa = torch.randint(0, 16, (m, k), device="cuda", dtype=torch.int8)
            qw = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.uint8)
            ref = qa.double() @ qw.double().T
            for config in range(9):
                c = torch.empty((m, n), device="cuda", dtype=torch.int32)
                status = lib.duquant_sm90_gemm(
                    qa.data_ptr(), pack(qw).data_ptr(), c.data_ptr(), m, n, k,
                    torch.cuda.current_stream().cuda_stream, config,
                )
                self.assertEqual(status, 0, (m, n, k, config))
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(c.double(), ref), (m, n, k, config))

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() in ((9, 0), (8, 0), (8, 6), (8, 7)),
        "Hopper / Ampere only",
    )
    def test_sm90_small_m_path(self):
        # Cooperative small-M kernel (rotation/permutation/quant + u4 GEMV) against
        # the Triton activation (bit-exact codes) and the large-M GEMM path.
        # K = 14336 needs several 128-blocks per worker CTA and/or R2 aliasing on
        # A100 and A40 (see launch_linear_small); A40 also takes those layouts at K = 11008.
        for k, n in ((128, 96), (4096, 4096), (11008, 256), (14336, 256)):
            w = torch.randn(n, k, device="cuda", dtype=torch.float16) / k**0.5
            r = torch.linalg.qr(torch.randn(2, 128, 128, device="cuda"))[0].half().contiguous()
            layer = DuQuantLinearCUDA(w, r[0], torch.randperm(k, device="cuda"), r[1],
                                      torch.randn(n, device="cuda", dtype=torch.float16) * 0.01)
            for m in (1, 2, 3, 4):
                if not layer.small_fits(m):
                    # A40 (99 KB shared memory): no cooperative layout for 3-4 rows
                    # at K >= 11008; forward() uses the GEMM path instead.
                    self.assertTrue(m > 2 and k >= 11008 and torch.cuda.get_device_capability() == (8, 6), (k, m))
                    continue
                x = torch.randn(m, k, device="cuda", dtype=torch.float16)
                small = layer.small_linear(x)
                ref = layer.activation(x, unpacked=False)
                # Tensor Core rotations round exactly like the Triton/torch path.
                for got, want in zip(small, ref):
                    self.assertTrue(torch.equal(got, want), (k, m))
                small_m = layer.small_m
                layer.small_m = 4
                y_small = layer(x)
                layer.small_m = 0
                y_big = layer(x)
                layer.small_m = small_m
                rel = ((y_small.float() - y_big.float()).norm() / y_big.float().norm()).item()
                self.assertLess(rel, 1e-3, (k, m, rel))
                for cfg in range(3):
                    layer.small_gemv_config = cfg
                    # Identical activation codes -> integer-exact GEMV vs GEMM.
                    a, sc, zp, sums = small
                    y = torch.empty(m, n, device="cuda", dtype=torch.float16)
                    st = layer.gemm.duquant_gemv_u4(
                        a.data_ptr(), layer.weight.data_ptr(), y.data_ptr(), m, n, k,
                        torch.cuda.current_stream().cuda_stream, cfg, sc.data_ptr(),
                        zp.data_ptr(), sums.data_ptr(), layer.weight_scale.data_ptr(),
                        layer.weight_zero.data_ptr(), layer.weight_correction.data_ptr(),
                        layer.epilogue_bias.data_ptr())
                    self.assertEqual(st, 0)
                    qa = torch.stack([a & 15, a >> 4], -1).reshape(m, k).double()
                    qw = torch.stack([layer.weight & 15, layer.weight >> 4], -1).reshape(n, k).double()
                    acc = qa @ qw.T
                    expect = ((acc + zp.double()[:, None] * layer.weight_correction.double()
                               - layer.weight_zero.double() * sums.double()[:, None])
                              * sc.double()[:, None] * layer.weight_scale.double()
                              + layer.epilogue_bias.double())
                    torch.testing.assert_close(y.double(), expect, atol=2e-3, rtol=2e-3)
                layer.small_gemv_config = None

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
