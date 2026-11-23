import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from time import perf_counter
from types import SimpleNamespace

import torch

from nanovllm.engine.cpu_prefix_cache import TieredBlockManager, prefix_key
from nanovllm.engine.kv_transfer import KVTransfer
from nanovllm.engine.metrics import RequestMetrics
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


class ControlledWrite(Future):
    """The simulated device completes the batch on the first blocking wait."""

    def __init__(self, copy):
        super().__init__()
        self.copy = copy

    def result(self, timeout=None):
        if not self.done():
            try:
                self.copy()
                self.set_result(perf_counter())
            except BaseException as exc:
                self.set_exception(exc)
        return super().result(timeout)


class Transfer:
    def __init__(self, capacity=2):
        self.capacity = capacity
        self.cpu = {}
        self.gpu = {}
        self.before_copy = lambda pairs: None
        self.failure = None
        self.submissions = []

    def save(self, gpu_id, slot):
        self.cpu[slot] = self.gpu[gpu_id]

    def submit_save_many(self, pairs):
        pairs = tuple(pairs)
        self.submissions.append(pairs)

        def copy():
            self.before_copy(pairs)
            for index, (gpu_id, slot) in enumerate(pairs):
                self.save(gpu_id, slot)
                if index == 0 and self.failure:
                    raise self.failure("partial batch write")

        return ControlledWrite(copy)

    def restore(self, slot, gpu_id):
        self.gpu[gpu_id] = self.cpu[slot]


class OverlapTests(unittest.TestCase):
    def setUp(self):
        self.transfer = Transfer()
        self.manager = TieredBlockManager(4, 256, self.transfer, b"test", "always", True, None, "overlap")
        self.sched = Scheduler(SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=2048,
                                               eos=-1, kvcache_block_size=256, num_kvcache_blocks=4))
        self.sched.block_manager = self.manager

    def fill(self, tokens):
        seq = Sequence(tokens)
        seq.metrics = RequestMetrics(seq.seq_id, len(tokens), perf_counter())
        self.assertEqual(self.manager.can_allocate(seq), 0)
        self.manager.allocate(seq, 0)
        seq.num_scheduled_tokens = len(tokens)
        for i in range(len(tokens) // 256):
            self.transfer.gpu[seq.block_table[i]] = tuple(seq.block(i))
        self.manager.hash_blocks(seq)
        return seq

    def evict_gpu(self):
        # Reclaim the entire pool without changing the CPU index.
        pressure = Sequence([91] * 769)
        self.assertEqual(self.manager.can_allocate(pressure), 0)
        self.manager.allocate(pressure, 0)
        self.manager.deallocate(pressure)

    def test_reservations_preemption_and_shared_reference(self):
        tokens = list(range(513))
        source = self.fill(tokens)
        self.assertIsNotNone(self.manager.pending_write)
        self.assertFalse(self.manager.entries)
        self.assertFalse(self.manager.free_slots)
        self.assertEqual(source.metrics.d2h_blocks, 0)
        self.assertEqual(len(self.transfer.submissions), 1)
        original_pages = source.block_table[:2]

        def before_copy(pairs):
            # If preemption, a lookup or allocation drains eagerly, all sources
            # and destination slots must still be reserved when the copy runs.
            self.assertFalse(self.manager.entries)
            for gpu_id, slot in pairs:
                self.assertNotIn(gpu_id, self.manager.free_block_ids)
                self.assertNotIn(slot, self.manager.free_slots)
                self.assertGreater(self.manager.blocks[gpu_id].ref_count, 0)

        self.transfer.before_copy = before_copy
        peer = Sequence(tokens[:256] + [42])
        self.assertEqual(self.manager.can_allocate(peer), 1)
        self.manager.allocate(peer, 1)
        self.assertEqual(peer.block_table[0], original_pages[0])
        self.sched.preempt(source)
        self.assertFalse(source.block_table)
        self.assertEqual(source.metrics.preemptions, 1)
        self.assertEqual(self.transfer.gpu[peer.block_table[0]], tuple(tokens[:256]))
        self.manager.finish_write()
        self.assertIsNone(self.manager.pending_write)
        self.assertEqual(self.manager.blocks[peer.block_table[0]].ref_count, 1)
        self.assertEqual(source.metrics.d2h_blocks, 2)
        self.manager.deallocate(peer)
        self.assertFalse(self.manager.used_block_ids)
        self.evict_gpu()
        replay = Sequence(tokens)
        self.assertEqual(self.manager.can_allocate(replay), 2)
        self.assertEqual(self.manager.allocate(replay, 2), 2)
        for index, page in enumerate(replay.block_table[:2]):
            self.assertEqual(self.transfer.gpu[page], tuple(replay.block(index)))
        self.manager.deallocate(replay)

    def test_last_reference_preempted_while_copy_is_pending(self):
        source = self.fill(list(range(513)))

        def ensure_source_still_owned(pairs):
            for gpu_id, slot in pairs:
                self.assertNotIn(gpu_id, self.manager.free_block_ids)
                self.assertNotIn(slot, self.manager.free_slots)
                self.assertGreater(self.manager.blocks[gpu_id].ref_count, 0)

        self.assertIsNotNone(self.manager.pending_write)
        self.transfer.before_copy = ensure_source_still_owned
        self.sched.preempt(source)
        self.manager.finish_write()
        self.assertFalse(self.manager.used_block_ids)
        self.assertEqual(sorted(self.manager.free_block_ids), [0, 1, 2, 3])
        self.evict_gpu()
        self.assertEqual(self.manager.can_allocate(source), 2)

    def test_submit_reports_failure_after_worker_started(self):
        started, release = Event(), Event()
        transfer = self.transfer
        writer = ThreadPoolExecutor(max_workers=1)
        transfer.writer = writer
        self.addCleanup(writer.shutdown)
        self.addCleanup(release.set)

        def submit_then_fail(pairs):
            def late_copy():
                started.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("scheduler never drained the ambiguous submission")
                for gpu_id, slot in pairs:
                    self.assertNotIn(slot, self.manager.free_slots)
                    self.assertFalse(self.manager.entries)
                    self.assertGreater(self.manager.blocks[gpu_id].ref_count, 0)
                    transfer.cpu[slot] = "partial copy"

            transfer.future = writer.submit(late_copy)
            self.assertTrue(started.wait(timeout=5))
            raise MemoryError("failed after the work was enqueued")

        def stop_writer():
            release.set()
            KVTransfer.stop_writer(transfer)

        transfer.submit_save_many = submit_then_fail
        transfer.stop_writer = stop_writer
        seq = self.fill(list(range(513)))
        self.assertTrue(transfer.future.done())
        transfer.future.result()
        self.assertIsNone(transfer.writer)
        self.assertIsNone(self.manager.pending_write)
        self.assertEqual(seq.metrics.cpu_save_failures, 1)
        self.assertEqual(seq.metrics.d2h_blocks, 2)
        self.assertEqual(len(self.manager.entries), 2)
        for slot, tokens in self.manager.entries.values():
            self.assertEqual(transfer.cpu[slot], tuple(tokens))
        self.manager.deallocate(seq)

    def test_partial_memory_failure_retries_every_page(self):
        self.transfer.failure = MemoryError
        seq = self.fill(list(range(513)))
        self.assertFalse(self.manager.entries)
        self.manager.finish_write()
        self.assertIsNone(self.manager.pending_write)
        self.assertEqual(seq.metrics.cpu_save_failures, 1)
        self.assertEqual(seq.metrics.d2h_blocks, 2)
        self.assertEqual(seq.metrics.d2h_batches, 0)
        self.assertEqual(len(self.manager.entries), 2)
        for slot, ids in self.manager.entries.values():
            self.assertEqual(self.transfer.cpu[slot], tuple(ids))
        self.manager.deallocate(seq)
        self.assertFalse(self.manager.used_block_ids)

    def test_unrecoverable_partial_failure_returns_reserved_slots(self):
        self.transfer.failure = RuntimeError
        seq = self.fill(list(range(513)))
        with self.assertRaisesRegex(RuntimeError, "partial batch"):
            self.manager.finish_write()
        self.assertIsNone(self.manager.pending_write)
        self.assertFalse(self.manager.entries)
        self.assertEqual(sorted(self.manager.free_slots), [0, 1])
        self.manager.deallocate(seq)
        self.assertFalse(self.manager.used_block_ids)

    def test_full_capacity_touches_and_replaces_lru(self):
        tokens = list(range(513))
        seq = self.fill(tokens)
        self.manager.finish_write()
        self.manager.deallocate(seq)
        root = prefix_key(b"test", tokens[:256])
        touched = Sequence(tokens[:256] + [93])
        self.manager.allocate(touched, 1)
        self.manager.deallocate(touched)
        self.assertEqual(next(reversed(self.manager.entries)), root)
        newcomer = self.fill([71] * 257)
        self.manager.finish_write()
        self.manager.deallocate(newcomer)
        self.assertEqual(list(self.manager.entries), [root, prefix_key(b"test", [71] * 256)])
        self.assertEqual(self.manager.evictions, 1)
        self.assertFalse(self.manager.free_slots)
        self.evict_gpu()
        replay = Sequence(tokens)
        self.assertEqual(self.manager.can_allocate(replay), 1)
        self.assertEqual(self.manager.allocate(replay, 1), 1)
        self.assertEqual(self.transfer.gpu[replay.block_table[0]], tuple(tokens[:256]))
        self.manager.deallocate(replay)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA streams and pinned CPU memory")
class OverlapCUDATests(unittest.TestCase):
    def test_background_copy_waits_for_producer_and_round_trips_bf16_bits(self):
        shape = (2, 2, 6, 256, 1, 16)
        # Compare raw encodings, including NaN payloads and signed zeros.
        expected = torch.arange(2 * 2 * 6 * 256 * 16, dtype=torch.int32).to(torch.int16).reshape(shape)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            src = expected.cuda()
            cache = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
            transfer = KVTransfer(cache, 3, batch_writes=True)
            self.addCleanup(transfer.close)
            stream.synchronize()
            torch.cuda._sleep(10_000_000)
            cache.view(torch.int16).copy_(src)
            pairs = [(0, 2), (2, 0), (4, 1)]
            transfer.submit_save_many(pairs).result()
            cache.zero_()
            for gpu_id, slot in pairs:
                transfer.restore(slot, gpu_id + 1)
                actual = cache[:, :, gpu_id + 1].view(torch.int16).cpu()
                self.assertTrue(torch.equal(actual, expected[:, :, gpu_id]))


if __name__ == "__main__":
    unittest.main()
