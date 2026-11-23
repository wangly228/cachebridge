from dataclasses import asdict, replace
import unittest

from nanovllm.engine.restore_policy import CostShape, RestoreCostTable


class CostTableTests(unittest.TestCase):
    def setUp(self):
        self.shape = CostShape(0, 512, 0, 128, 1, 0)
        self.env = {"block_size": 256, "gpu": "test"}
        self.bucket = dict(shape=asdict(self.shape), paired_waves=4, restore_p50_ms=9, restore_p90_ms=12,
                           recompute_p10_ms=20, recompute_p50_ms=24)
        self.data = dict(version=1, environment=self.env, buckets=[self.bucket])

    def test_quantile_rule_and_exact_lookup(self):
        table = RestoreCostTable(self.data, self.env)
        known = table.decide(self.shape)
        self.assertEqual((known.action, known.restore_ms, known.recompute_ms), ("restore", 9, 24))
        new = table.decide(replace(self.shape, query_tokens=129))
        self.assertEqual((new.action, new.reason, new.restore_ms), ("recompute", "uncalibrated_shape", None))
        self.bucket["restore_p90_ms"] = 20
        choice = RestoreCostTable(self.data, self.env).decide(self.shape)
        self.assertEqual((choice.action, choice.reason), ("recompute", "overlapping_costs"))
        self.bucket.update(restore_p50_ms=30, restore_p90_ms=34)
        self.assertEqual(RestoreCostTable(self.data, self.env).decide(self.shape).reason, "measured_recompute")

    def test_reject_environment_mismatch_and_invalid_data(self):
        with self.assertRaises(ValueError):
            RestoreCostTable(self.data, dict(self.env, gpu="other"))
        for update in ({"paired_waves": 1}, {"restore_p50_ms": float("nan")}, {"restore_p90_ms": 5},
                       {"shape": asdict(replace(self.shape, cpu_tokens=1))}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                RestoreCostTable(dict(self.data, buckets=[dict(self.bucket, **update)]), self.env)
        with self.assertRaises(ValueError):
            RestoreCostTable(dict(self.data, buckets=[self.bucket, self.bucket]), self.env)
