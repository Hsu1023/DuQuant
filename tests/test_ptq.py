import unittest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from duquant_cuda.backend import DuQuantLinearCUDA, transform
from duquant_cuda.decoder import IntegerReferenceLinear, LlamaDecoderBlock, random_rotations
from duquant_cuda.ptq import symmetric_rtn, gptq, search_activation
from test_gqa import gqa_weights


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PTQTests(unittest.TestCase):
    @torch.inference_mode()
    def test_symmetric_export_integer_kernel(self):
        torch.manual_seed(371)
        r = random_rotations(512, 1024, "cuda")["qkv"]
        w = torch.randn(128, 512, device="cuda", dtype=torch.float16)*.1
        state, report = symmetric_rtn(transform(w, *r), [1., .9, .8])
        fast = DuQuantLinearCUDA(w, *r, quantized_weight=state, act_clip=.975)
        ref = IntegerReferenceLinear(w, r, quantized_weight=state, act_clip=.975)
        for m in (1, 17, 128):
            x = torch.randn(m, 512, device="cuda", dtype=torch.float16)
            torch.testing.assert_close(fast(x), ref(x), atol=0, rtol=0)
        self.assertTrue((state["zero"] == 8).all())
        self.assertTrue((state["codes"] <= 15).all())
        self.assertGreater(report["weight_mse"], 0)

    @torch.inference_mode()
    def test_gptq_matches_independent_column_oracle(self):
        torch.manual_seed(372)
        torch.backends.cuda.matmul.allow_tf32 = False
        w = torch.randn(32, 256, device="cuda", dtype=torch.float16)
        x = torch.randn(512, 256, device="cuda")
        x[:, 3] = 0  # dead feature
        h = x.T @ x / len(x)
        state, _ = symmetric_rtn(w, [1., .9, .8])
        actual, _ = gptq(w, h, state["scale"])
        hw = h.clone()
        work = w.float().clone()
        dead = hw.diag() == 0
        hw[dead, dead] = 1
        work[:, dead] = 0
        hw.diagonal().add_(.01*hw.diag().mean())
        hi = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(hw)), upper=True)
        expected = torch.empty_like(actual["codes"])
        scale = state["scale"].float()
        # Full column feedback instead of the optimized block implementation.
        for col in range(w.shape[1]):
            value = work[:, col].clone()
            code = (value/scale).round().clamp(-8, 7)
            expected[:, col] = (code+8).byte()
            err = (value-code*scale)/hi[col, col]
            work[:, col:] -= err[:, None]*hi[col, col:][None, :]
        torch.testing.assert_close(actual["codes"], expected, atol=0, rtol=0)
        diagonal, _ = gptq(w, torch.eye(256, device="cuda"), state["scale"])
        torch.testing.assert_close(diagonal["codes"], state["codes"], atol=0, rtol=0)

    @torch.inference_mode()
    def test_searched_clips_full_block(self):
        torch.manual_seed(373)
        weights, rotations = gqa_weights(1), random_rotations(512, 1024, "cuda")
        groups = dict(qkv=torch.cat([weights[f"self_attn.{p}_proj.weight"] for p in ("q", "k", "v")]),
                      o=weights["self_attn.o_proj.weight"],
                      gate_up=torch.cat([weights[f"mlp.{p}_proj.weight"] for p in ("gate", "up")]),
                      down=weights["mlp.down_proj.weight"])
        plan = dict(weights={}, act_clips={}, kv_clips=(.975, .925))
        for name, w in groups.items():
            plan["weights"][name], _ = symmetric_rtn(transform(w, *rotations[name]), [1., .9])
            sample = torch.randn(129, w.shape[1], device="cuda", dtype=torch.float16)
            plan["act_clips"][name], _ = search_activation(sample, [1., .975, .9])
        fast = LlamaDecoderBlock(weights, rotations, "cuda", max_position=256, kv_bits=4, quantization=plan)
        ref = LlamaDecoderBlock(weights, rotations, "integer_reference", max_position=256, kv_bits=4, quantization=plan)
        a, b = fast.cache(1, 130), ref.cache(1, 130)
        x = torch.randn(1, 129, 512, device="cuda", dtype=torch.float16)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            y, z = fast(x, a), ref(x, b)
        torch.testing.assert_close(y, z, atol=0, rtol=0)
        k, v = a.unpack(129)
        torch.testing.assert_close(k, b.k[:, :, :129], atol=0, rtol=0)
        torch.testing.assert_close(v, b.v[:, :, :129], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
