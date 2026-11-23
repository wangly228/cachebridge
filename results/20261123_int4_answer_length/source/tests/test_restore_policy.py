from dataclasses import asdict, replace
import unittest

from nanovllm.engine.restore_policy import CostShape, RestoreCostTable, StepWork


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

    def test_queue_bucket_precedes_shape_and_unseen_queue_uses_shape(self):
        shape = replace(self.shape, outstanding=4)
        legacy = asdict(shape)
        del legacy["peer_query_tokens"], legacy["peer_cpu_tokens"]
        generic = dict(self.bucket, shape=legacy)
        mixed = replace(shape, peer_query_tokens=384, peer_cpu_tokens=1536)
        special = dict(self.bucket, shape=asdict(mixed), restore_p50_ms=30, restore_p90_ms=32)
        policy = RestoreCostTable(dict(self.data, version=2, buckets=[generic, special]), self.env)
        self.assertEqual(policy.queue_shapes, {shape})
        self.assertEqual((policy.decide(mixed).action, policy.decide(mixed).cost_basis), ("recompute", "queue"))
        unseen = policy.decide(replace(mixed, peer_query_tokens=640))
        self.assertEqual((unseen.action, unseen.cost_basis), ("restore", "shape"))
        self.assertEqual(policy.decide(replace(mixed, cpu_tokens=1024)).reason, "uncalibrated_shape")
        old = RestoreCostTable(dict(self.data, buckets=[generic]), self.env)
        self.assertFalse(old.queue_shapes)
        self.assertEqual(old.decide(shape), unseen)

    def test_queue_counts_must_be_present_together_and_cpu_block_aligned(self):
        for query, cpu, version in ((None, 0, 2), (0, None, 2), (-1, 0, 2), (1, 255, 2), (0, 0, 1)):
            shape = asdict(replace(self.shape, peer_query_tokens=query, peer_cpu_tokens=cpu))
            with self.subTest(query=query, cpu=cpu, version=version), self.assertRaises(ValueError):
                RestoreCostTable(dict(self.data, version=version, buckets=[dict(self.bucket, shape=shape)]), self.env)

    def test_step_interpolation_changes_action_only_inside_measured_support(self):
        shape = replace(self.shape, outstanding=4, running=1)
        env = dict(self.env, max_num_seqs=8, max_num_batched_tokens=2048)
        anchor = dict(shape=asdict(shape), action="restore", seqs=4, cpu_tokens=512, seed_count=4)
        left = dict(anchor, query_tokens=500, p10_ms=40, p50_ms=50, p90_ms=60)
        right = dict(anchor, query_tokens=1500, p10_ms=140, p50_ms=150, p90_ms=160)
        other = dict(anchor, action="recompute", seqs=3, cpu_tokens=0, query_tokens=1600,
                     p10_ms=110, p50_ms=120, p90_ms=130)
        data = dict(version=3, environment=env, buckets=[dict(self.bucket, shape=asdict(shape))],
                    step_anchors=[right, left, other])
        table = RestoreCostTable(data, env)
        work = dict(restore=StepWork(4, 700, 512), recompute=StepWork(3, 1600, 0))
        decision = table.decide_step(shape, work)
        self.assertEqual((decision.restore_ms, decision.recompute_ms, decision.action), (70, 120, "restore"))
        self.assertEqual(table.decide_step(shape, dict(work, restore=StepWork(4, 1500, 512))).action, "recompute")
        # At the midpoint p90 equals recompute p10. The inherited rule is strict.
        self.assertEqual(table.decide_step(shape, dict(work, restore=StepWork(4, 1000, 512))).reason, "overlapping_costs")
        for unseen in (StepWork(4, 1600, 512), StepWork(4, 600, 256), StepWork(3, 700, 512)):
            self.assertIsNone(table.decide_step(shape, dict(work, restore=unseen)))
        self.assertEqual(table.decide(shape).cost_basis, "shape")
        for invalid in ([right, right], [dict(left, p10_ms=90)], [dict(left, cpu_tokens=1)],
                        [dict(left, query_tokens=2049)], [dict(left, seed_count=1)], []):
            with self.subTest(anchors=invalid), self.assertRaises(ValueError):
                RestoreCostTable(dict(data, step_anchors=invalid), env)
        with self.assertRaises(ValueError):
            RestoreCostTable(dict(data, version=2), env)
