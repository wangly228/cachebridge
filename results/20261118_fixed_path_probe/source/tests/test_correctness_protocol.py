import unittest

import torch

from benchmarks.correctness_protocol import bitwise_equal, compare_logits


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


if __name__ == "__main__":
    unittest.main()
