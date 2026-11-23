import unittest

import torch

from benchmarks.correctness_protocol import bitwise_equal, compare_logits
from benchmarks.analyze_fixed_path import analyze, decode_bf16


class CorrectnessProtocolTests(unittest.TestCase):
    def test_bit_check_detects_signed_zero_and_one_bit_change(self):
        a = torch.tensor([0.0, 1.0], dtype=torch.bfloat16)
        b = a.clone()
        self.assertTrue(bitwise_equal(a, b))
        b[0] = -0.0
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(bitwise_equal(a, b))
        b = a.clone()
        b.view(torch.int16)[1] ^= 1
        self.assertFalse(bitwise_equal(a, b))

    def test_near_tie_is_reported_separately_from_argmax(self):
        a = torch.tensor([11.3125, 11.3125, -3.0])
        b = torch.tensor([11.3125, 11.375, -3.0])
        row = compare_logits(a, b)
        self.assertTrue(row["passed"])
        self.assertFalse(row["argmax_equal"])
        self.assertTrue(row["within_near_tie_budget"])

    def test_large_error_and_nonfinite_are_rejected(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        self.assertFalse(compare_logits(a, torch.tensor([1.0, 5.0, 3.0]))["passed"])
        self.assertFalse(compare_logits(a, torch.tensor([1.0, float("nan"), 3.0]))["passed"])
        with self.assertRaises(ValueError):
            compare_logits(a, a[:2])

    def test_offline_bf16_decode_preserves_bits(self):
        original = torch.tensor([0.0, -0.0, 1.0625, -11.375], dtype=torch.bfloat16)
        decoded = decode_bf16(original.view(torch.int16).numpy()).to(torch.bfloat16)
        self.assertTrue(bitwise_equal(original, decoded))

    def test_constant_logit_shift_does_not_change_probabilities(self):
        original = torch.tensor([[1.0, 2.0, 4.0]])
        result = analyze(original, original + 8.0)
        self.assertEqual(result["constant_shift_error_energy_fraction"].item(), 1.0)
        self.assertEqual(result["centered_max_abs"].item(), 0.0)
        self.assertLess(result["probability_total_variation"].item(), 1e-6)
        changed = analyze(original, torch.tensor([[1.0, 4.0, 2.0]]))
        self.assertGreater(changed["probability_total_variation"].item(), 0.5)


if __name__ == "__main__":
    unittest.main()
