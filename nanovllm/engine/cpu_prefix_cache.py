from collections import OrderedDict
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.restore_policy import CostShape, RestoreDecision
from nanovllm.engine.sequence import Sequence


def cache_namespace(config) -> bytes:
    """Identify the loaded model/config. Each engine also owns a private store."""
    model = Path(config.model).resolve()
    weights = []
    for shard in sorted(model.glob("*.safetensors")):
        metadata = model / ".cache/huggingface/download" / (shard.name + ".metadata")
        identity = metadata.read_text().splitlines()[:2] if metadata.is_file() else []
        stat = shard.stat()
        weights.append((shard.name, identity, stat.st_size, stat.st_mtime_ns))
    spec = (str(model), weights, json.loads(config.hf_config.to_json_string(use_diff=False)),
            str(config.hf_config.dtype), config.kvcache_block_size, config.tensor_parallel_size)
    if getattr(config, "cpu_prefix_cache_format", "native") == "int4":
        spec += ("int4-symmetric-group32-bf16-absmax-v1",)
    return sha256(json.dumps(spec, sort_keys=True).encode()).digest()


def prefix_key(parent: bytes, tokens: list[int]) -> bytes:
    return sha256(parent + np.asarray(tokens, dtype="<i8").tobytes()).digest()


@dataclass(slots=True)
class PrefixHit:
    gpu_id: int | None
    slot: int | None
    gpu_hash: int
    key: bytes
    tokens: list[int]


class TieredBlockManager(BlockManager):
    """Bounded CPU prefix cache. All metadata runs on the scheduler thread."""

    def __init__(self, num_blocks, block_size, transfer, namespace: bytes, admission="always", batch_writes=False,
                 first_use_blocks: int | None = None, transfer_mode="sync"):
        super().__init__(num_blocks, block_size)
        self.transfer = transfer
        self.namespace = namespace
        self.gpu_keys = [None] * num_blocks
        self.entries: OrderedDict[bytes, tuple[int, list[int]]] = OrderedDict()
        self.free_slots = list(reversed(range(transfer.capacity)))
        self.evictions = 0
        self.transfer_mode = transfer_mode
        self.pending_write = None
        self.admission = admission
        self.batch_writes = batch_writes
        self.first_use_limit = min(num_blocks, transfer.capacity)
        if first_use_blocks is not None:
            self.first_use_limit = min(self.first_use_limit, first_use_blocks)
        self.recent_uses: OrderedDict[bytes, int] = OrderedDict()

    def _lookup(self, seq):
        hits = []
        gpu_hash, parent = -1, self.namespace
        for i in range(seq.num_blocks - 1):  # Keep the upstream query/tail block.
            tokens = seq.block(i)
            gpu_hash = self.compute_hash(tokens, gpu_hash)
            key = prefix_key(parent, tokens)
            gpu_id = self.hash_to_block_id.get(gpu_hash)
            if gpu_id is not None and (self.gpu_keys[gpu_id] != key or self.blocks[gpu_id].token_ids != tokens):
                gpu_id = None
            slot = None
            if gpu_id is None:
                entry = self.entries.get(key)
                if i >= seq.num_prompt_tokens // self.block_size or entry is None or entry[1] != tokens:
                    break
                slot = entry[0]
            hits.append(PrefixHit(gpu_id, slot, gpu_hash, key, tokens))
            parent = key
        return hits

    def _fits(self, seq, hits):
        shared = sum(hit.gpu_id in self.used_block_ids for hit in hits)
        return len(self.free_block_ids) >= seq.num_blocks - shared

    def can_allocate(self, seq: Sequence) -> int:
        hits = self._lookup(seq)
        return len(hits) if self._fits(seq, hits) else -1

    def pending_prefill_work(self, seq):
        """Query tokens and CPU tokens for a waiting peer if it takes all contiguous hits."""
        if seq.block_table:
            return seq.num_tokens - seq.num_cached_tokens, 0
        hits = self._lookup(seq)
        cpu_blocks = sum(hit.slot is not None for hit in hits)
        return seq.num_tokens - len(hits) * self.block_size, cpu_blocks * self.block_size

    def select_prefix(self, seq, policy, outstanding, running, peer_work=None, step_work=None):
        hits = self._lookup(seq)
        if not self._fits(seq, hits):
            return -1, None
        first_cpu = next((i for i, hit in enumerate(hits) if hit.slot is not None), len(hits))
        if first_cpu == len(hits):
            return len(hits), None
        cpu_count = sum(hit.slot is not None for hit in hits)
        shape = CostShape(first_cpu * self.block_size, cpu_count * self.block_size,
                          (len(hits) - first_cpu - cpu_count) * self.block_size,
                          seq.num_tokens - len(hits) * self.block_size, outstanding, running)
        if seq.num_completion_tokens:
            decision = RestoreDecision("recompute", "preempted")
        else:
            decision = None
            if step_work is not None and shape in policy.step_curves:
                work = step_work(shape)
                if work is not None:
                    decision = policy.decide_step(shape, work)
            if decision is None:
                if peer_work is not None and shape in policy.queue_shapes:
                    query_tokens, cpu_tokens = peer_work()
                    shape = replace(shape, peer_query_tokens=query_tokens, peer_cpu_tokens=cpu_tokens)
                decision = policy.decide(shape)
        if decision.action == "recompute":
            if self._fits(seq, hits[:first_cpu]):
                return first_cpu, (shape, decision)
            # Later shared, active GPU pages can make only the full restore fit.
            decision = replace(decision, action="restore", reason="gpu_capacity")
        return len(hits), (shape, decision)

    def deallocate(self, seq: Sequence):
        # A source is owned by an active sequence from reservation to publication.
        # Finish before any release (completion, preemption or allocation rollback).
        if self.pending_write is not None:
            self.finish_write()
        super().deallocate(seq)

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        # Hits can change LRU order, promote GPU pages or reuse transfer staging.
        # A cold allocation only uses free GPU blocks, disjoint from the active sources.
        if self.pending_write is not None and num_cached_blocks:
            self.finish_write()
        hits = self._lookup(seq)
        assert not seq.block_table and 0 <= num_cached_blocks <= len(hits)
        if num_cached_blocks < len(hits):
            hits = hits[:num_cached_blocks]
        # Protect every GPU hit before taking any free ID, including hits after a CPU hit.
        table = [None] * seq.num_blocks
        for i, hit in enumerate(hits):
            if hit.gpu_id is not None:
                self._claim_block(hit.gpu_id)
                table[i] = hit.gpu_id
        for i, block_id in enumerate(table):
            if block_id is None:
                table[i] = self._allocate_block()

        seq.block_table = table
        metrics = seq.metrics
        restores = 0
        try:
            for i, hit in enumerate(hits):
                if hit.slot is None:
                    continue
                start = perf_counter() if metrics else 0
                self.transfer.restore(hit.slot, table[i])
                restores += 1
                if metrics:
                    metrics.h2d_blocks += 1
                    metrics.h2d_ms += (perf_counter() - start) * 1000
        except (MemoryError, torch.OutOfMemoryError):
            # Invalidating the failed slot makes the scheduler's next lookup shorter.
            # Re-admission recalculates the query length and the batch budget.
            self.free_slots.append(self.entries.pop(hit.key)[0])
            if metrics:
                metrics.cpu_restore_failures += 1
            self.deallocate(seq)
            return -1
        except BaseException:
            self.deallocate(seq)
            raise

        # A promotion may evict a CPU slot used by a later hit in this allocation.
        # Finish all restores before allowing any new writes to those slots.
        try:
            for i, hit in enumerate(hits):
                block_id = table[i]
                if hit.slot is not None:
                    self.blocks[block_id].update(hit.gpu_hash, hit.tokens)
                    self.gpu_keys[block_id] = hit.key
                    self.hash_to_block_id[hit.gpu_hash] = block_id
                if hit.key in self.entries:
                    self.entries.move_to_end(hit.key)
                elif self.admission == "reuse" and hit.gpu_id is not None and i < seq.num_prompt_tokens // self.block_size:
                    self._admit(hit.key, self.blocks[block_id], seq)
        except BaseException:
            self.deallocate(seq)
            raise
        seq.num_cached_tokens = len(hits) * self.block_size
        return restores

    def _admit(self, key, block, seq):
        # Seed up to the configured limit of first-use prefixes. Once populated, a new
        # key must be observed on two distinct requests within the bounded history.
        if self.admission == "reuse" and key not in self.entries and len(self.entries) >= self.first_use_limit:
            previous = self.recent_uses.get(key)
            if previous is None or previous == seq.seq_id:
                self.recent_uses[key] = seq.seq_id
                self.recent_uses.move_to_end(key)
                if len(self.recent_uses) > self.transfer.capacity:
                    self.recent_uses.popitem(last=False)
                return
        self._save(key, block, seq.metrics)
        if self.recent_uses and key in self.entries:
            self.recent_uses.pop(key, None)

    def _save(self, key, block, metrics):
        if key in self.entries:
            # Recomputing an aligned query block can produce different BF16 bits.
            # Match the GPU's latest publication, including when the token key is unchanged.
            slot, _ = self.entries.pop(key)
        elif self.free_slots:
            slot = self.free_slots.pop()
        else:
            _, (slot, _) = self.entries.popitem(last=False)
            self.evictions += 1
        try:
            start = perf_counter() if metrics else 0
            self.transfer.save(block.block_id, slot)
            if metrics:
                metrics.d2h_blocks += 1
                metrics.d2h_ms += (perf_counter() - start) * 1000
        except (MemoryError, torch.OutOfMemoryError):
            self.free_slots.append(slot)
            if metrics:
                metrics.cpu_save_failures += 1
            return
        except BaseException:
            self.free_slots.append(slot)
            raise
        self.entries[key] = (slot, block.token_ids)

    def _admit_computed(self, pages, seq):
        # Only group unconditional first writes into unused slots. Mixed groups
        # follow normal admission in order, including refreshes, LRU and second uses.
        if self.pending_write is not None:
            self.finish_write()
        room = len(self.free_slots)
        if self.admission == "reuse":
            room = min(room, self.first_use_limit - len(self.entries))
        count = len(pages)
        if 2 <= count <= room:
            keys = {key for key, _ in pages}
            if len(keys) == count and not any(key in self.entries for key in keys):
                if self._save_new_pages(pages, seq.metrics):
                    return
        for key, block in pages:
            self._admit(key, block, seq)

    def _save_new_pages(self, pages, metrics):
        count = len(pages)
        slots = self.free_slots[-count:][::-1]
        pairs = [(block.block_id, slot) for (_, block), slot in zip(pages, slots)]
        del self.free_slots[-count:]
        start = perf_counter() if metrics else 0
        try:
            if self.transfer_mode == "overlap":
                future = self.transfer.submit_save_many(pairs)
                self.pending_write = (future, pages, slots, metrics, start)
                return True
            self.transfer.save_many(pairs)
        except BaseException as exc:
            if self.transfer_mode == "overlap":
                # submit() or registering the handle may fail after enqueue.
                # Cancel queued work and join any running copy before slot reuse.
                self.transfer.stop_writer()
                self.pending_write = None
            self.free_slots.extend(reversed(slots))
            if not isinstance(exc, (MemoryError, torch.OutOfMemoryError)):
                raise
            if metrics:
                metrics.cpu_save_failures += 1
                metrics.d2h_ms += (perf_counter() - start) * 1000
            return False  # Retry via the regular one-block path.
        self._publish_group(pages, slots, metrics, perf_counter() - start if metrics else 0)
        return True

    def finish_write(self):
        if self.pending_write is None:
            return
        future, pages, slots, metrics, start = self.pending_write
        try:
            completed = future.result()
        except BaseException as exc:
            if not future.done():
                raise  # An interrupted waiter must keep the reservation.
            self.pending_write = None
            self.free_slots.extend(reversed(slots))
            if not isinstance(exc, (MemoryError, torch.OutOfMemoryError)):
                raise
            if metrics:
                metrics.cpu_save_failures += 1
                metrics.d2h_ms += (perf_counter() - start) * 1000
            for key, block in pages:
                self._save(key, block, metrics)
                if key in self.entries:
                    self.recent_uses.pop(key, None)
            return
        self.pending_write = None
        self._publish_group(pages, slots, metrics, completed - start)

    def _publish_group(self, pages, slots, metrics, elapsed):
        if metrics:
            count = len(pages)
            metrics.d2h_blocks += count
            metrics.d2h_batches += 1
            metrics.batched_d2h_blocks += count
            metrics.d2h_ms += elapsed * 1000
        try:
            for (key, block), slot in zip(pages, slots):
                self.entries[key] = (slot, block.token_ids)
                self.recent_uses.pop(key, None)
        except BaseException:
            for key, _ in pages:
                self.entries.pop(key, None)
            self.free_slots.extend(reversed(slots))
            raise

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end:
            return
        super().hash_blocks(seq)
        parent = self.gpu_keys[seq.block_table[start - 1]] if start else self.namespace
        pages = [] if self.batch_writes else None
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            parent = prefix_key(parent, block.token_ids)
            self.gpu_keys[block.block_id] = parent
            if i < seq.num_prompt_tokens // self.block_size:
                if pages is None:
                    self._admit(parent, block, seq)
                else:
                    pages.append((parent, block))
        if pages:
            self._admit_computed(pages, seq)
