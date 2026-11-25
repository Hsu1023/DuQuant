import copy
import unittest
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from duquant_cuda.backend import affine_quant, DuQuantLinearCUDA
from duquant_cuda.decoder import (
    LlamaDecoderBlock,
    IntegerReferenceLinear,
    random_rotations,
)
from duquant_cuda.lwc import LWCDecoder, qdq_ste
from duquant_cuda.lwc_quant import weight_qdq_ste
from quantize.quantizer import UniformAffineQuantizer
from test_decoder import small_weights


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class LWCTests(unittest.TestCase):
    def test_fused_training_trajectory(self):
        torch.manual_seed(173)
        block = LWCDecoder(
            small_weights(),
            random_rotations(256, 512, "cuda"),
            dict(rms_norm_eps=1.0e-5, max_position_embeddings=32),
        )
        fused = copy.deepcopy(block)
        for name in ("qkv", "o", "gate_up", "down"):
            getattr(fused, name).fused_weight_quant = True
        opts = [
            torch.optim.AdamW(b.parameters(), lr=0.01, weight_decay=0)
            for b in (block, fused)
        ]
        with torch.enable_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for _ in range(5):
                x = torch.randn(1, 16, 256, device="cuda", dtype=torch.float16)
                for b, opt in zip((block, fused), opts):
                    opt.zero_grad(set_to_none=True)
                    ((b(x).float() - x.float()).square().mean() * 1024).backward()
                    opt.step()
                for a, b in zip(block.parameters(), fused.parameters()):
                    torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
                    torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_fused_weight_forward_and_gradients(self):
        torch.manual_seed(172)
        for k in (256, 4096, 11008):
            for kind in ("random", "constant", "positive"):
                x = torch.randn(64, k, device="cuda", dtype=torch.float16) * 0.03
                if kind == "constant":
                    x.fill_(0.01)
                elif kind == "positive":
                    x.abs_().add_(0.01)
                low = (
                    (torch.rand(64, 1, device="cuda") * 0.7 + 0.2)
                    .half()
                    .requires_grad_()
                )
                high = (
                    (torch.rand(64, 1, device="cuda") * 0.7 + 0.2)
                    .half()
                    .requires_grad_()
                )
                g = torch.randn_like(x) * 0.01
                with torch.enable_grad():
                    a = qdq_ste(x, low, high)
                    b = weight_qdq_ste(
                        x, x.amin(-1, keepdim=True), x.amax(-1, keepdim=True), low, high
                    )
                    ga = torch.autograd.grad(a, (low, high), g)
                    gb = torch.autograd.grad(b, (low, high), g)
                torch.testing.assert_close(a, b, atol=0, rtol=0)
                for aa, bb in zip(ga, gb):
                    # Degenerate constant rows can overflow the reference's
                    # FP16 backward too; match those nonfinite locations exactly.
                    torch.testing.assert_close(aa, bb, atol=0, rtol=0, equal_nan=True)

    def test_export_matches_upstream_lwc_quantizer(self):
        torch.manual_seed(163)
        x = torch.randn(256, 512, device="cuda", dtype=torch.float16)
        q = (
            UniformAffineQuantizer(
                n_bits=4,
                shape=x.shape,
                lwc=True,
                rotate=False,
                quant_method="duquant",
                dynamic_method="per_channel",
            )
            .cuda()
            .half()
        )
        with torch.no_grad():
            q.lowbound_factor.uniform_(-1, 4)
            q.upbound_factor.uniform_(-1, 4)
            bounds = (q.lowbound_factor.sigmoid(), q.upbound_factor.sigmoid())
            codes, scale, zero = affine_quant(x, bounds)
            deployed = (codes.half() - zero[:, None]) * scale[:, None]
            torch.testing.assert_close(deployed, q(x), atol=0, rtol=0)
            torch.testing.assert_close(qdq_ste(x, *bounds), deployed, atol=0, rtol=0)

    def test_training_gradients_and_fake_export(self):
        torch.manual_seed(165)
        weights = small_weights()
        rotations = random_rotations(256, 512, "cuda")
        cfg = dict(rms_norm_eps=1.0e-5, max_position_embeddings=32)
        block = LWCDecoder(weights, rotations, cfg)
        x = torch.randn(1, 16, 256, device="cuda", dtype=torch.float16)
        with torch.enable_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            y = block(x)
            loss = (y.float() - x.float()).square().mean()
            loss.backward()
        for name, param in block.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
            self.assertGreater(param.grad.abs().sum().item(), 0, name)
        bounds = {
            k: tuple(v.cuda() for v in pair)
            for k, pair in block.export_bounds().items()
        }
        fake = LlamaDecoderBlock(
            weights, rotations, "reference", max_position=32, weight_clips=bounds
        )
        with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            torch.testing.assert_close(
                block(x), fake(x, fake.cache(1, 16)), atol=0, rtol=0
            )

    @torch.no_grad()
    def test_learned_bounds_integer_export(self):
        torch.manual_seed(166)
        w = torch.randn(256, 512, device="cuda", dtype=torch.float16) * 0.03
        r = random_rotations(256, 512, "cuda")["down"]
        clip = tuple(
            torch.rand(256, 1, device="cuda", dtype=torch.float16) * 0.7 + 0.2
            for _ in range(2)
        )
        fast = DuQuantLinearCUDA(w, *r, weight_clip=clip)
        ref = IntegerReferenceLinear(w, r, weight_clip=clip)
        for n in (1, 128):
            x = torch.randn(n, 512, device="cuda", dtype=torch.float16)
            torch.testing.assert_close(fast(x), ref(x), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
