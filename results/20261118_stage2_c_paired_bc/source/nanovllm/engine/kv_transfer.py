import numpy as np
import torch


class KVTransfer:
    """Synchronous single-GPU copies. One packed block of staging per device."""

    def __init__(self, kv_cache: torch.Tensor, capacity: int):
        self.kv_cache = kv_cache
        self.capacity = capacity
        shape = (*kv_cache.shape[:2], *kv_cache.shape[3:])
        self.device_stage = torch.empty(shape, dtype=kv_cache.dtype, device=kv_cache.device)
        self.pinned = torch.empty(shape, dtype=kv_cache.dtype, device="cpu", pin_memory=True)
        self.host_bytes = self.pinned.view(torch.uint8).numpy().reshape(-1)
        self.block_bytes = self.host_bytes.size
        self.slots = np.empty((capacity, self.block_bytes), dtype=np.uint8)

    def save(self, gpu_id: int, slot: int):
        try:
            self.device_stage.copy_(self.kv_cache[:, :, gpu_id])
            self.pinned.copy_(self.device_stage)  # Blocking D2H also waits for packing.
        except Exception:
            torch.cuda.current_stream(self.kv_cache.device).synchronize()
            raise
        np.copyto(self.slots[slot], self.host_bytes)

    def restore(self, slot: int, gpu_id: int):
        np.copyto(self.host_bytes, self.slots[slot])
        try:
            self.device_stage.copy_(self.pinned, non_blocking=True)
            self.kv_cache[:, :, gpu_id].copy_(self.device_stage)
        finally:
            torch.cuda.current_stream(self.kv_cache.device).synchronize()

    def close(self):
        del self.kv_cache, self.device_stage, self.slots, self.host_bytes, self.pinned
