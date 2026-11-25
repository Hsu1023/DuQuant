import unittest
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from duquant_cuda.calibration import apply_smoothing
from duquant_cuda.decoder import LlamaDecoderBlock, _rms, rms_reduction_width
from test_decoder import small_weights


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CalibrationTests(unittest.TestCase):
    @torch.inference_mode()
    def test_folded_smoothing_preserves_float_block(self):
        torch.manual_seed(74)
        original = small_weights()
        weights = {k: v.clone() for k, v in original.items()}
        scales = {
            name: torch.rand(size, device="cuda", dtype=torch.float16) + 0.5
            for name, size in (
                ("qkv", 256),
                ("gate_up", 256),
                ("o", 256),
                ("down", 512),
            )
        }
        apply_smoothing(weights, scales)
        a = LlamaDecoderBlock(original, {}, "fp16")
        b = LlamaDecoderBlock(weights, {}, "fp16")
        x = torch.randn(1, 17, 256, device="cuda", dtype=torch.float16)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            expected = a(x, a.cache(1, 17))
            actual = b(x, b.cache(1, 17))
        self.assertLess(
            (
                (actual.float() - expected.float()).norm() / expected.float().norm()
            ).item(),
            0.002,
        )

    @torch.inference_mode()
    def test_rms_matches_torch_rounding(self):
        torch.manual_seed(73)
        for n, h in ((1, 4096), (9, 256), (128, 4096), (2048, 4096)):
            x = torch.randn(n, h, device="cuda", dtype=torch.float16)
            w = torch.rand(h, device="cuda", dtype=torch.float16) + 0.5
            y = torch.empty_like(x)
            for add in (False, True):
                residual = torch.randn_like(x)
                z = x + residual if add else x
                expected = (
                    z.float()
                    * torch.rsqrt(z.float().square().mean(-1, keepdim=True) + 1.0e-5)
                ).half() * w
                r = torch.empty_like(x)
                _rms[(n,)](
                    x,
                    w,
                    y,
                    r,
                    residual,
                    n,
                    h,
                    1.0e-5,
                    add,
                    h,
                    rms_reduction_width(n, h),
                    enable_fp_fusion=False,
                )
                torch.testing.assert_close(y, expected, atol=0, rtol=0)
                if add:
                    torch.testing.assert_close(r, z, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
