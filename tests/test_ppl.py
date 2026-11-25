import unittest

import torch
from benchmarks.eval_ppl import score_next_tokens


class PPLScoringTests(unittest.TestCase):
    def test_next_token_shift_and_scored_token_count(self):
        tokens = torch.tensor([[0, 2, 1], [2, 0, 1]])
        logits = torch.tensor(
            [
                [[0.0, 1.0, 2.0], [1.0, 2.0, 0.0], [9.0, 0.0, 0.0]],
                [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 9.0, 0.0]],
            ],
            dtype=torch.float16,
        )
        losses, predictions, targets = score_next_tokens(logits, tokens)
        self.assertEqual(losses.numel(), 4)
        self.assertEqual(losses.dtype, torch.float32)
        self.assertTrue(torch.equal(predictions, targets))
        expected = torch.logsumexp(torch.tensor([0.0, 1.0, 2.0]), 0) - 2
        torch.testing.assert_close(losses, expected.expand(4))
        # The unscored final position cannot change PPL.
        logits[:, -1] = -999
        torch.testing.assert_close(score_next_tokens(logits, tokens)[0], losses)


if __name__ == "__main__":
    unittest.main()
