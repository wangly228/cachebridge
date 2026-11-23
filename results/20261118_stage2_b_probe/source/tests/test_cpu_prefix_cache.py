import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nanovllm.engine.cpu_prefix_cache import TieredBlockManager, prefix_key
from nanovllm.engine.metrics import RequestMetrics
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

    def restore(self, slot, gpu_id):
        self.gpu[gpu_id] = self.cpu[slot]


class CacheTests(unittest.TestCase):
    def setup_cache(self, capacity=8, batch=2048):
        self.transfer = Transfer(capacity)
        self.manager = TieredBlockManager(3, 256, self.transfer, b"engine")
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
