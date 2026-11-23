import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

from nanovllm.config import Config
from nanovllm.engine.cpu_prefix_cache import TieredBlockManager, prefix_key
from nanovllm.engine.metrics import RequestMetrics
from nanovllm.engine.restore_policy import CostShape, RestoreCostTable, StepWork, cost_environment
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class Transfer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cpu = {}
        self.gpu = {}

    def save(self, gpu_id, slot):
        assert 0 <= slot < self.capacity
        self.cpu[slot] = self.gpu[gpu_id]

    def save_many(self, pairs):
        for gpu_id, slot in pairs:
            self.save(gpu_id, slot)

    def restore(self, slot, gpu_id):
        self.gpu[gpu_id] = self.cpu[slot]


def policy(shape, restore, recompute):
    env = {"block_size": 256}
    bucket = dict(shape=asdict(shape), paired_waves=4, restore_p50_ms=restore, restore_p90_ms=restore,
                  recompute_p10_ms=recompute, recompute_p50_ms=recompute)
    return RestoreCostTable(dict(version=1, environment=env, buckets=[bucket]), env)


class CacheTests(unittest.TestCase):
    def setup_cache(self, capacity=8, batch=2048, admission="always", batch_writes=False, first_use_blocks=None):
        self.transfer = Transfer(capacity)
        self.manager = TieredBlockManager(3, 256, self.transfer, b"engine", admission, batch_writes, first_use_blocks)
        self.sched = Scheduler(SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=batch,
                                               eos=-1, kvcache_block_size=256, num_kvcache_blocks=3))
        self.sched.block_manager = self.manager

    def submit(self, tokens, output=1):
        seq = Sequence(tokens, SamplingParams(temperature=0, max_tokens=output, ignore_eos=True))
        seq.metrics = RequestMetrics(seq.seq_id, len(tokens), 0)
        self.sched.add(seq)
        return seq

    def step(self):
        seqs, prefill = self.sched.schedule()
        for seq in seqs:
            end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // 256
            for i in range(seq.num_cached_tokens // 256, end):
                self.transfer.gpu[seq.block_table[i]] = tuple(seq.block(i))
        self.sched.postprocess(seqs, [17] * len(seqs), prefill)

    def finish(self):
        while not self.sched.is_finished():
            self.step()

    def evict_gpu(self):
        # Claim and release three uncomputed pages, as in a cancelled/preempted prefill.
        other = Sequence([91] * 513)
        self.assertEqual(self.manager.can_allocate(other), 0)
        self.manager.allocate(other, 0)
        self.manager.deallocate(other)
        self.assertFalse(self.manager.hash_to_block_id)

    def test_reuse_admission_bounds_first_use_and_history(self):
        self.setup_cache(capacity=4, admission="reuse")
        for value in range(12):
            cold = self.submit([value] * 257)
            self.finish()
            self.assertEqual(cold.metrics.d2h_blocks, int(value < 3))
        self.assertEqual(len(self.manager.entries), 3)
        self.assertEqual(len(self.manager.recent_uses), 4)
        self.evict_gpu()
        forgotten = self.submit([5] * 257)
        self.finish()
        remembered = self.submit([10] * 257)
        self.finish()
        self.assertEqual((forgotten.metrics.d2h_blocks, remembered.metrics.d2h_blocks), (0, 1))
        self.assertEqual(remembered.metrics.prefill_tokens, 257)
        self.evict_gpu()
        restored = self.submit([10] * 257)
        self.finish()
        self.assertEqual((restored.metrics.cpu_restored_tokens, restored.metrics.prefill_tokens), (256, 1))

    def test_explicit_limit_and_second_use(self):
        for batch_writes in (False, True):
            for limit in (0, 1):
                with self.subTest(batch_writes=batch_writes, limit=limit):
                    self.setup_cache(capacity=4, admission="reuse", batch_writes=batch_writes, first_use_blocks=limit)
                    tokens = list(range(513))
                    first = self.submit(tokens)
                    self.finish()
                    self.assertEqual((first.metrics.d2h_blocks, first.metrics.d2h_batches), (limit, 0))
                    second = self.submit(tokens)
                    self.finish()
                    self.assertEqual((second.metrics.initial_gpu_hit_tokens, second.metrics.d2h_blocks), (512, 2 - limit))
                    self.assertEqual(len(self.manager.entries), 2)
                    self.evict_gpu()
                    replay = self.submit(tokens)
                    self.finish()
                    self.assertEqual((replay.metrics.h2d_blocks, replay.metrics.prefill_tokens), (2, 1))

    def test_public_first_use_limit_and_cost_table_identity(self):
        bucket = dict(shape=asdict(CostShape(0, 256, 0, 1, 1, 0)), paired_waves=4, restore_p50_ms=1,
                      restore_p90_ms=2, recompute_p10_ms=3, recompute_p50_ms=4)
        with (
            patch("nanovllm.config.AutoConfig.from_pretrained", return_value=SimpleNamespace(max_position_embeddings=4096)),
            patch("nanovllm.engine.scheduler.cache_namespace", return_value=b"model"),
            patch("torch.cuda.get_device_name", return_value="test"),
            patch("torch.cuda.get_device_capability", return_value=(8, 6)),
        ):
            for invalid in (-1, True, False, 1.5, "32"):
                with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "cpu_prefix_first_use_blocks"):
                    Config(".", cpu_prefix_first_use_blocks=invalid)
            for gpu, slots in ((3, 8), (8, 3)):
                config = Config(".", num_kvcache_blocks=gpu, cpu_prefix_cache_blocks=slots, cpu_prefix_admission="reuse")
                auto = cost_environment(config, b"model")
                for limit in (None, 0, 2, 3, 32):
                    with self.subTest(gpu=gpu, slots=slots, limit=limit):
                        config.cpu_prefix_first_use_blocks = limit
                        sched = Scheduler(config, Transfer(slots))
                        effective = 3 if limit is None else min(limit, 3)
                        self.assertEqual(sched.block_manager.first_use_limit, effective)
                        env = cost_environment(config, b"model")
                        expected = dict(auto, cpu_prefix_first_use_blocks=effective) if effective < 3 else auto
                        self.assertEqual(env, expected)
                        data = dict(version=1, environment=auto, buckets=[bucket])
                        if effective < 3:
                            with self.assertRaisesRegex(ValueError, "environment"):
                                RestoreCostTable(data, env)
                        else:
                            RestoreCostTable(data, env)
                config.cpu_prefix_admission = "always"
                config.cpu_prefix_first_use_blocks = 0
                self.assertNotIn("cpu_prefix_first_use_blocks", cost_environment(config, b"model"))

    def test_always_admission_ignores_first_use_limit(self):
        self.setup_cache(capacity=4, admission="always", batch_writes=True, first_use_blocks=0)
        seq = self.submit(list(range(513)))
        self.finish()
        self.assertEqual((seq.metrics.d2h_blocks, seq.metrics.d2h_batches), (2, 1))

    def test_batched_first_writes_reserve_free_slots_until_complete(self):
        self.setup_cache(capacity=4, admission="reuse", batch_writes=True)
        copy = self.transfer.save_many
        def inspect(pairs):
            self.assertFalse(self.manager.entries)
            self.assertEqual([slot for _, slot in pairs], [0, 1])
            for gpu_id, slot in pairs:
                self.assertNotIn(slot, self.manager.free_slots)
                self.assertGreater(self.manager.blocks[gpu_id].ref_count, 0)
            copy(pairs)
        with patch.object(self.transfer, "save_many", side_effect=inspect) as grouped:
            first = self.submit(list(range(513)))
            self.finish()
            other = self.submit([72] * 513)
            self.finish()
        # The second request crosses the admission watermark after its first block.
        grouped.assert_called_once()
        self.assertEqual((first.metrics.d2h_blocks, first.metrics.batched_d2h_blocks, first.metrics.d2h_batches), (2, 2, 1))
        self.assertEqual((other.metrics.d2h_blocks, other.metrics.d2h_batches), (1, 0))
        self.evict_gpu()
        replay = self.submit(list(range(513)))
        self.finish()
        self.assertEqual(replay.metrics.cpu_restored_tokens, 512)

    def test_partial_batch_failure_releases_or_retries_every_slot(self):
        for error in (MemoryError, RuntimeError):
            with self.subTest(error=error):
                self.setup_cache(capacity=3, batch_writes=True)
                def partial(pairs):
                    self.transfer.save(*pairs[0])
                    raise error("injected second copy failure")
                with patch.object(self.transfer, "save_many", side_effect=partial):
                    seq = self.submit(list(range(513)))
                    if error is RuntimeError:
                        with self.assertRaises(RuntimeError):
                            self.step()
                        self.assertFalse(self.manager.entries)
                        self.assertEqual(self.manager.free_slots, [2, 1, 0])
                    else:
                        self.finish()
                        self.assertEqual(seq.metrics.cpu_save_failures, 1)
                        self.assertEqual((seq.metrics.d2h_blocks, seq.metrics.d2h_batches), (2, 0))
                        self.assertEqual([slot for slot, _ in self.manager.entries.values()], [0, 1])
                        self.assertEqual(self.manager.free_slots, [2])
                        self.evict_gpu()
                        replay = self.submit(list(range(513)))
                        self.finish()
                        self.assertEqual(replay.metrics.cpu_restored_tokens, 512)

    def test_reuse_admission_ignores_probes_and_same_request_retries(self):
        self.setup_cache(capacity=1, admission="reuse")
        self.submit([1] * 257)
        self.finish()
        tokens = [2] * 257
        seq = self.submit(tokens, output=3)
        self.step()
        key = prefix_key(b"engine", tokens[:256])
        self.assertEqual(self.manager.recent_uses[key], seq.seq_id)
        for _ in range(2):
            peer = Sequence(tokens)
            self.assertEqual(self.manager.can_allocate(peer), 1)
            self.assertEqual(self.manager.pending_prefill_work(peer), (1, 0))
        self.sched.running.remove(seq)
        self.sched.preempt(seq)
        self.evict_gpu()
        self.finish()
        self.assertEqual(seq.metrics.d2h_blocks, 0)
        self.assertNotIn(key, self.manager.entries)
        hot = self.submit(tokens)
        hot.metrics = None
        self.finish()
        self.assertIn(key, self.manager.entries)
        self.assertNotIn(key, self.manager.recent_uses)

    def test_reuse_promotion_copies_after_all_restores_and_retries_failed_save(self):
        tokens = list(range(513))
        first, second = tokens[:256], tokens[256:512]
        root = prefix_key(b"engine", first)
        child = prefix_key(root, second)
        for fail in (False, True):
            with self.subTest(failed_promotion=fail):
                self.setup_cache(capacity=1, admission="reuse")
                # The recomputed aligned tail takes the only CPU slot. The next
                # short prompt records a first use of the now GPU-only root.
                for prompt in (tokens, tokens[:512], tokens[:257]):
                    self.submit(prompt)
                    self.finish()
                self.assertEqual(list(self.manager.entries), [child])
                self.assertIn(root, self.manager.recent_uses)
                child_hit = self.manager._lookup(Sequence(tokens))[1]
                if child_hit.gpu_id is not None:
                    del self.manager.hash_to_block_id[child_hit.gpu_hash]
                mixed = self.submit(tokens)
                if fail:
                    with patch.object(self.transfer, "save", side_effect=MemoryError("promotion failed")):
                        self.finish()
                    self.assertNotIn(root, self.manager.entries)
                    self.assertEqual(mixed.metrics.cpu_save_failures, 1)
                    self.submit(tokens)
                self.finish()
                self.assertEqual(mixed.metrics.h2d_blocks, 1)
                self.assertEqual(mixed.metrics.d2h_blocks, int(not fail))
                self.assertFalse(self.manager.used_block_ids)
                child_hit = self.manager._lookup(Sequence(tokens))[1]
                self.assertEqual(self.transfer.gpu[child_hit.gpu_id], tuple(second))
                self.assertEqual(self.transfer.cpu[self.manager.entries[root][0]], tuple(first))

    def test_edges_and_query_retention(self):
        for length in (255, 256, 257, 512, 513):
            with self.subTest(length=length):
                self.setup_cache(batch=191)
                tokens = list(range(length))
                cold = self.submit(tokens)
                self.finish()
                self.assertEqual(cold.metrics.d2h_blocks, length // 256)
                self.evict_gpu()
                replay = self.submit(tokens)
                self.finish()
                self.assertEqual(replay.metrics.initial_gpu_hit_tokens, 0)
                self.assertEqual(replay.metrics.cpu_restored_tokens, (length - 1) // 256 * 256)
                self.assertEqual(replay.metrics.prefill_tokens, (length - 1) % 256 + 1)

    def test_policy_charges_cpu_opportunity_and_commits_one_decision(self):
        self.setup_cache(batch=256)
        tokens = list(range(257))
        self.submit(tokens)
        self.finish()
        self.evict_gpu()
        self.sched.restore_policy = policy(CostShape(0, 256, 0, 1, 1, 0), 100, 15)
        self.submit([43] * 100)
        replay = self.submit(tokens)
        self.step()  # Insufficient residual token budget to admit the recompute.
        self.assertFalse(replay.block_table or replay.metrics.restore_decisions)
        self.finish()
        m = replay.metrics
        decision, = m.restore_decisions
        self.assertEqual((decision["reason"], decision["recompute_candidate_tokens"]), ("measured_recompute", 256))
        self.assertEqual((m.initial_cpu_hit_tokens, m.cpu_restored_tokens, m.prefill_tokens), (256, 0, 257))
        self.assertEqual((m.h2d_blocks, m.d2h_blocks), (0, 1))
        self.evict_gpu()
        self.sched.restore_policy = policy(CostShape(0, 256, 0, 1, 1, 0), 15, 100)
        replay = self.submit(tokens)
        replay.metrics = None
        with patch.object(self.transfer, "restore", wraps=self.transfer.restore) as copied:
            self.finish()
        copied.assert_called_once()

    def test_queue_lookahead_is_bounded_and_does_not_touch_cache(self):
        self.setup_cache()
        for value in (4, 5):
            self.submit([value]*257)
            self.finish()
        self.evict_gpu()
        self.submit([3]*257)
        warm = self.submit([4]*257)
        self.submit([7]*17)
        self.submit([5]*257)
        lru = list(self.manager.entries.items())
        free = list(self.manager.free_block_ids)
        # One peer was already scheduled. The visible window has two more peer slots.
        self.assertEqual(self.sched._peer_work(1, 10, 256), (28, 512))
        self.assertEqual(list(self.manager.entries.items()), lru)
        self.assertEqual(list(self.manager.free_block_ids), free)
        self.assertFalse(self.manager.used_block_ids)
        self.manager.allocate(warm, 1)
        with patch.object(self.manager, "_lookup", side_effect=AssertionError("re-probed allocated peer")):
            self.assertEqual(self.manager.pending_prefill_work(warm), (1, 0))
        self.manager.deallocate(warm)

    def test_queue_lookup_is_lazy_for_gpu_hits_and_v1(self):
        self.setup_cache()
        seq = self.submit([4]*257)
        self.finish()
        slow = policy(CostShape(0, 256, 0, 1, 1, 0), 30, 10)
        replay = Sequence(seq.prompt_token_ids)
        def unexpected():
            self.fail("queue scanned without a contextual CPU decision")
        self.assertEqual(self.manager.select_prefix(replay, slow, 1, 0, unexpected), (1, None))
        self.evict_gpu()
        n, (_, decision) = self.manager.select_prefix(replay, slow, 1, 0, unexpected)
        self.assertEqual((n, decision.reason), (0, "measured_recompute"))

    def test_step_projection_respects_budget_equality_fifo_and_capacity(self):
        self.setup_cache()
        self.manager = self.sched.block_manager = TieredBlockManager(36, 256, self.transfer, b"engine")
        self.sched.running.append(Sequence([5]*128))
        self.submit([2]*1152)
        peer = self.submit([3]*640)
        cold = self.submit([4]*640)
        shape = CostShape(0, 1024, 0, 128, 4, 1)
        def pending(seq):
            return (128, 512) if seq is peer else (seq.num_tokens, 0)
        with patch.object(self.manager, "pending_prefill_work", side_effect=pending):
            for size in (639, 640, 641, 1663, 1664, 1665):
                with self.subTest(size=size):
                    self.sched.waiting[-1] = Sequence([4]*size)
                    plan = self.sched._step_work(1, 128, 512, shape)
                    self.assertEqual(plan["restore"], StepWork(3 + (size <= 1664), 384 + (size if size <= 1664 else 0), 2048))
                    self.assertEqual(plan["recompute"], StepWork(3 + (size <= 640), 1408 + (size if size <= 640 else 0), 1024))
            self.sched.waiting[1] = Sequence([1]*2048)
            self.sched.waiting[2] = cold
            self.assertEqual(self.sched._step_work(1, 128, 512, shape)["restore"], StepWork(2, 256, 1536))
            self.assertIsNone(self.sched._step_work(1, 1024, 512, shape))
            self.assertIsNone(self.sched._step_work(0, 0, 0, shape))
            self.sched.waiting[1] = Sequence([1]*(36*256))
            self.assertIsNone(self.sched._step_work(1, 128, 512, shape))
        self.assertEqual(len(self.manager.free_block_ids), 36)
        self.assertFalse(self.manager.entries)

    def test_unsupported_step_plan_retains_queue_policy(self):
        self.setup_cache()
        tokens = [1]*257
        self.submit(tokens)
        self.finish()
        self.evict_gpu()
        shape = CostShape(0, 256, 0, 1, 1, 0)
        p = policy(shape, 1, 40)
        p.queue_shapes.add(shape)
        p.step_curves[shape] = {}
        work = dict(restore=StepWork(1, 1, 256), recompute=StepWork(1, 257, 0))
        blocks, (key, decision) = self.manager.select_prefix(Sequence(tokens), p, 1, 0, lambda: (0, 0), lambda s: work)
        self.assertEqual((blocks, key.peer_query_tokens, decision.action, decision.cost_basis), (1, 0, "restore", "shape"))

    def test_policy_preserves_native_prefix_and_protects_active_suffix(self):
        self.setup_cache()
        tokens = list(range(513))
        self.submit(tokens)
        self.finish()
        seq = Sequence(tokens)
        slow = policy(CostShape(256, 256, 0, 1, 1, 0), 50, 10)
        # Invalidate the lookup for the second page, keeping the first page native.
        second = self.manager._lookup(seq)[1]
        del self.manager.hash_to_block_id[second.gpu_hash]
        n, (shape, choice) = self.manager.select_prefix(seq, slow, 1, 0)
        self.assertEqual((n, shape.gpu_prefix_tokens, choice.reason), (1, 256, "measured_recompute"))
        self.assertEqual(self.manager.allocate(seq, n), 0)
        self.assertEqual(seq.num_cached_tokens, 256)
        self.manager.deallocate(seq)
        self.setup_cache()
        self.submit(tokens)
        self.finish()
        first, second = self.manager._lookup(seq)
        del self.manager.hash_to_block_id[first.gpu_hash]
        self.manager._claim_block(second.gpu_id)
        n, (shape, choice) = self.manager.select_prefix(seq, slow, 1, 0)
        self.assertEqual((n, shape.gpu_after_cpu_tokens, choice.reason), (2, 256, "gpu_capacity"))
        self.assertEqual(self.manager.allocate(seq, n), 1)
        self.assertEqual(seq.block_table[1], second.gpu_id)
        self.manager.deallocate(seq)
        self.assertEqual(self.manager.blocks[second.gpu_id].ref_count, 1)

    def test_policy_preemption_replans_without_first_token_prediction(self):
        self.setup_cache()
        self.sched.restore_policy = policy(CostShape(0, 256, 0, 2, 1, 0), 1, 50)
        seq = self.submit([8]*257, output=3)
        self.step()
        self.sched.running.remove(seq)
        self.sched.preempt(seq)
        self.evict_gpu()
        self.finish()
        decision, = seq.metrics.restore_decisions
        self.assertEqual((decision["reason"], decision["allocation"], decision["restored_tokens"]), ("preempted", 1, 0))
        self.assertIsNone(decision["restore_ms"])
        self.assertEqual(len(seq.completion_token_ids), 3)

    def test_full_lru_touch_and_gap(self):
        self.setup_cache(capacity=2)
        for value in (1, 2, 1, 3):
            self.submit([value] * 257)
            self.finish()
        keys = [prefix_key(b"engine", [i] * 256) for i in (1, 2, 3)]
        self.assertEqual(list(self.manager.entries), [keys[0], keys[2]])
        self.assertEqual(self.manager.evictions, 1)
        self.setup_cache(capacity=1)
        self.submit([5] * 513)
        self.finish()
        self.evict_gpu()
        self.assertEqual(len(self.manager.entries), 1)  # Only the second block survived.
        seq = self.submit([5] * 513)
        self.finish()
        self.assertEqual((seq.metrics.cpu_restored_tokens, seq.metrics.prefill_tokens), (0, 513))

    def test_partial_cpu_then_gpu_and_shared_live_reference(self):
        self.setup_cache()
        tokens = list(range(513))
        self.submit(tokens)
        self.finish()
        first, second = (h.gpu_id for h in self.manager._lookup(Sequence(tokens)))
        self.manager.free_block_ids.rotate(1)
        self.submit([15])  # Physically reclaim the first page; the second is still on GPU.
        self.finish()
        hits = self.manager._lookup(Sequence(tokens))
        self.assertEqual([h.gpu_id for h in hits], [None, second])
        self.assertNotEqual(first, second)
        ref = self.submit(tokens, output=2)
        self.step()
        self.assertEqual((ref.metrics.initial_cpu_hit_tokens, ref.metrics.initial_gpu_hit_tokens), (256, 256))
        self.assertEqual(self.transfer.gpu[ref.block_table[1]], tuple(tokens[256:512]))
        self.finish()
        self.evict_gpu()
        one = self.submit(tokens[:257], output=2)
        two = self.submit(tokens[:256] + [42], output=3)
        self.step()
        shared = one.block_table[0]
        self.assertEqual(one.metrics.cpu_restored_tokens, 256)
        self.assertEqual(two.metrics.initial_gpu_hit_tokens, 256)
        self.assertEqual(two.block_table[0], shared)
        self.assertEqual(self.manager.blocks[shared].ref_count, 2)
        self.step()
        self.assertEqual(self.manager.blocks[shared].ref_count, 1)
        self.finish()
        self.assertFalse(self.manager.used_block_ids)

    def test_preempt_resume_and_generated_kv_exclusion(self):
        self.setup_cache()
        seq = self.submit([5] * 257, output=3)
        self.step()
        self.sched.running.remove(seq)
        self.sched.preempt(seq)
        self.evict_gpu()
        self.finish()
        self.assertEqual((seq.metrics.allocations, seq.metrics.preemptions), (2, 1))
        self.assertEqual((seq.metrics.cpu_restored_tokens, seq.metrics.initial_cpu_hit_tokens), (256, 0))
        self.assertEqual((seq.metrics.prefill_tokens, seq.metrics.prompt_computed_tokens), (259, 258))
        self.setup_cache()
        self.submit([1] * 255, output=3)
        self.finish()
        self.assertFalse(self.manager.entries)

    def test_recomputed_full_tail_refreshes_the_cpu_copy(self):
        tokens = list(range(512))
        for admission, grouped in (("always", False), ("reuse", False), ("reuse", True)):
            with self.subTest(admission=admission, batch_writes=grouped):
                self.setup_cache(capacity=2, admission=admission, batch_writes=grouped)
                self.submit(tokens)
                self.finish()
                second = self.submit(tokens)
                seqs, prefill = self.sched.schedule()
                self.assertEqual(second.num_cached_tokens, 256)
                # The same complete tail can be recomputed under a different prefill shape.
                self.transfer.gpu[second.block_table[1]] = b"newly computed tail"
                self.sched.postprocess(seqs, [17], prefill)
                self.evict_gpu()
                longer = self.submit(tokens + [19])
                self.sched.schedule()
                self.assertEqual(longer.num_cached_tokens, 512)
                self.assertEqual(self.transfer.gpu[longer.block_table[1]], b"newly computed tail")

    def test_copy_errors_replan_and_publish_only_ready_data(self):
        self.setup_cache(capacity=2, batch=256)
        tokens = list(range(513))
        self.submit(tokens)
        self.finish()
        self.evict_gpu()
        second_slot = list(self.manager.entries.values())[1][0]
        original = self.transfer.restore

        def fail_second(slot, gpu_id):
            self.assertEqual(self.manager.blocks[gpu_id].ref_count, 1)
            self.assertEqual(self.manager.blocks[gpu_id].hash, -1)
            if slot == second_slot:
                raise MemoryError("injected restore failure")
            original(slot, gpu_id)

        with patch.object(self.transfer, "restore", side_effect=fail_second):
            replay = self.submit(tokens)
            self.finish()
        m = replay.metrics
        self.assertEqual((m.cpu_restore_failures, m.initial_cpu_hit_tokens, m.h2d_blocks), (1, 256, 2))
        self.assertEqual((m.prefill_tokens, len(replay.completion_token_ids)), (257, 1))
        with patch.object(self.transfer, "save", side_effect=MemoryError("injected save failure")):
            bad = self.submit([7] * 257)
            self.finish()
        self.assertEqual(bad.metrics.cpu_save_failures, 1)
        self.assertNotIn(prefix_key(b"engine", [7] * 256), self.manager.entries)
        self.assertEqual(len(self.manager.free_slots) + len(self.manager.entries), 2)
        self.assertFalse(self.manager.used_block_ids)
        self.assertEqual(len(set(self.manager.free_block_ids)), 3)

    def test_token_validation_namespace_and_ancestor_collisions(self):
        self.setup_cache()
        first = [1] * 256
        tail = [2] * 256
        with patch.object(self.manager, "compute_hash", return_value=13):
            self.submit(first + tail + [4])
            self.finish()
            different_parent = Sequence([3] * 256 + tail + [4])
            self.assertEqual(self.manager.can_allocate(different_parent), 0)
        self.evict_gpu()
        key = prefix_key(b"engine", first)
        slot, _ = self.manager.entries[key]
        self.manager.entries[key] = (slot, [42] * 256)
        self.assertEqual(self.manager.can_allocate(Sequence(first + [4])), 0)
        self.assertNotEqual(key, prefix_key(b"other model", first))


if __name__ == "__main__":
    unittest.main()
