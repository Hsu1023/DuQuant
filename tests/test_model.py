import unittest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from duquant_cuda.decoder import LlamaDecoderBlock, random_rotations
from duquant_cuda.model import (
    LlamaInferenceModel,
    GenerationSession,
    model_storage_bytes,
)
from test_decoder import small_weights


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FullModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_grad_enabled(False)

    def make_model(self, mode):
        blocks = []
        for _ in range(2):
            w = small_weights()
            blocks.append(
                LlamaDecoderBlock(
                    w, random_rotations(256, 512, "cuda"), mode, max_position=32
                )
            )
        return LlamaInferenceModel(
            blocks,
            torch.randn(128, 256, device="cuda", dtype=torch.float16) * 0.1,
            torch.ones(256, device="cuda", dtype=torch.float16),
            torch.randn(128, 256, device="cuda", dtype=torch.float16) * 0.05,
            1.0e-5,
        )

    def test_growing_generation_matches_full_prefix(self):
        model = self.make_model("fp16")
        prompt = torch.tensor([[1, 2, 3, 4]], device="cuda")
        session = GenerationSession(model, prompt, 4)
        actual = session.eager().clone()
        sequence = prompt
        expected = []
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for _ in range(4):
                logits = model(sequence, model.caches(1, sequence.shape[1]))
                token = logits.argmax(-1, keepdim=True)
                expected.append(token)
                sequence = torch.cat((sequence, token), 1)
        self.assertTrue(torch.equal(actual, torch.cat(expected, 1)))
        self.assertEqual(session.kv_bytes, 2 * 2 * 1 * 2 * 7 * 128 * 2)
        self.assertGreater(model_storage_bytes(model), 0)

    def test_graph_feedback_and_changed_prompt(self):
        for mode in ("fp16", "cuda"):
            model = self.make_model(mode)
            prompt = torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8]], device="cuda")
            session = GenerationSession(model, prompt, 4)
            session.capture()
            session.prompt.add_(1)
            expected = session.eager().clone()
            session.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(expected, session.generated), mode)
            # A clean run verifies the replay does not rely on stale KV contents.
            clean = GenerationSession(model, session.prompt, 4)
            self.assertTrue(torch.equal(clean.eager(), session.generated), mode)

    def test_all_position_logits_match_causal_prefixes(self):
        model = self.make_model("fp16")
        tokens = torch.tensor([[1, 2, 3, 4]], device="cuda")
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            full = model(tokens, model.caches(1, 4), last_token_only=False)
            self.assertEqual(full.shape, (1, 4, 128))
            for end in range(1, 5):
                last = model(tokens[:, :end], model.caches(1, end))
                torch.testing.assert_close(
                    full[:, end - 1], last, rtol=0.01, atol=0.003
                )


if __name__ == "__main__":
    unittest.main()
