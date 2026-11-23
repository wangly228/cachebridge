from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import numpy as np
import torch


class KVTransfer:
    """Single-GPU copies sharing bounded staging between saves and restores."""

    def __init__(self, kv_cache: torch.Tensor, capacity: int, batch_writes: bool = False):
        self.kv_cache = kv_cache
        self.capacity = capacity
        shape = (*kv_cache.shape[:2], *kv_cache.shape[3:])
        self.device_stage = torch.empty(shape, dtype=kv_cache.dtype, device=kv_cache.device)
        self.pinned = torch.empty(shape, dtype=kv_cache.dtype, device="cpu", pin_memory=True)
        self.host_bytes = self.pinned.view(torch.uint8).numpy().reshape(-1)
        self.block_bytes = self.host_bytes.size
        self.slots = np.empty((capacity, self.block_bytes), dtype=np.uint8)
        self.writer = None
        self.write_stages = ()
        if batch_writes:
            second = torch.empty(shape, dtype=kv_cache.dtype, device="cpu", pin_memory=True)
            self.write_stages = tuple((buf, buf.view(torch.uint8).numpy().reshape(-1), torch.cuda.Event())
                                      for buf in (self.pinned, second))

    def save(self, gpu_id: int, slot: int):
        try:
            self.device_stage.copy_(self.kv_cache[:, :, gpu_id])
            self.pinned.copy_(self.device_stage)  # Blocking D2H also waits for packing.
        except Exception:
            torch.cuda.current_stream(self.kv_cache.device).synchronize()
            raise
        np.copyto(self.slots[slot], self.host_bytes)

    def save_many(self, pairs: list[tuple[int, int]]):
        """Complete every (GPU block, CPU slot) copy before returning to the scheduler."""
        if not self.write_stages or len(pairs) < 2:
            for gpu_id, slot in pairs:
                self.save(gpu_id, slot)
            return
        stream = torch.cuda.current_stream(self.kv_cache.device)

        def enqueue(index):
            gpu_id, _ = pairs[index]
            pinned, _, ready = self.write_stages[index % 2]
            self.device_stage.copy_(self.kv_cache[:, :, gpu_id])
            pinned.copy_(self.device_stage, non_blocking=True)
            ready.record(stream)

        try:
            enqueue(0)
            for i, (_, slot) in enumerate(pairs):
                if i + 1 < len(pairs):
                    enqueue(i + 1)
                _, source, ready = self.write_stages[i % 2]
                ready.synchronize()
                np.copyto(self.slots[slot], source)
        except BaseException:
            # A later D2H may already be in flight, including when the CPU write fails.
            stream.synchronize()
            raise

    def submit_save_many(self, pairs):
        """The scheduler owns exclusive staging until the returned future completes."""
        if self.writer is None:
            self.write_stream = torch.cuda.Stream(device=self.kv_cache.device)
            self.writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kv-write")
        ready = torch.cuda.current_stream(self.kv_cache.device).record_event()

        def copy():
            with torch.inference_mode(), torch.cuda.stream(self.write_stream):
                self.write_stream.wait_event(ready)
                self.save_many(pairs)
            return perf_counter()

        return self.writer.submit(copy)

    def restore(self, slot: int, gpu_id: int):
        np.copyto(self.host_bytes, self.slots[slot])
        try:
            self.device_stage.copy_(self.pinned, non_blocking=True)
            self.kv_cache[:, :, gpu_id].copy_(self.device_stage)
        finally:
            torch.cuda.current_stream(self.kv_cache.device).synchronize()

    def close(self):
        if self.writer is not None:
            self.writer.shutdown(wait=True)
        del self.kv_cache, self.device_stage, self.slots, self.host_bytes, self.pinned, self.write_stages
