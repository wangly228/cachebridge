from collections import deque
from dataclasses import asdict
from functools import partial
from itertools import islice
from time import perf_counter

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.cpu_prefix_cache import TieredBlockManager, cache_namespace
from nanovllm.engine.restore_policy import RestoreCostTable, StepWork, cost_environment


class Scheduler:

    def __init__(self, config: Config, kv_transfer=None):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.restore_policy = None
        if kv_transfer is None:
            self.block_manager = BlockManager(config.num_kvcache_blocks, self.block_size)
        else:
            self.block_manager = TieredBlockManager(config.num_kvcache_blocks, self.block_size,
                                                   kv_transfer, cache_namespace(config))
            if config.cpu_restore_cost_table is not None:
                environment = cost_environment(config, self.block_manager.namespace)
                self.restore_policy = RestoreCostTable.load(config.cpu_restore_cost_table, environment)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _peer_work(self, scheduled_count, query_tokens, cpu_tokens):
        # Already admitted peers contribute their actual work in this step. Estimate
        # the remaining visible peers from current cache metadata, without touching LRU.
        for seq in islice(self.waiting, 1, self.max_num_seqs - scheduled_count):
            query, cpu = self.block_manager.pending_prefill_work(seq)
            query_tokens += query
            cpu_tokens += cpu
        return query_tokens, cpu_tokens

    def _step_work(self, scheduled_count, query_tokens, cpu_tokens, shape):
        # Calibrated at first admission, with earlier running requests all in this step.
        if len(self.running) != scheduled_count:
            return None
        window = list(islice(self.waiting, self.max_num_seqs - scheduled_count))
        # Only project batches that fit even without sharing any active GPU page.
        if (any(seq.block_table or seq.num_completion_tokens for seq in window)
                or sum(seq.num_blocks for seq in window) > len(self.block_manager.free_block_ids)):
            return None
        recompute = shape.query_tokens + shape.cpu_tokens + shape.gpu_after_cpu_tokens
        if query_tokens + recompute > self.max_num_batched_tokens:
            return None
        peers = [self.block_manager.pending_prefill_work(seq) for seq in window[1:]]
        work = {}
        for action, target_query, copy in (("restore", shape.query_tokens, shape.cpu_tokens),
                                           ("recompute", recompute, 0)):
            count = scheduled_count + 1
            queries = query_tokens + target_query
            copies = cpu_tokens + copy
            for peer_query, peer_copy in peers:
                if queries + peer_query > self.max_num_batched_tokens:
                    break
                queries += peer_query
                copies += peer_copy
                count += 1
            work[action] = StepWork(count, queries, copies)
        return work

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0
        restored_batch_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                choice = None
                if self.restore_policy is None:
                    num_cached_blocks = self.block_manager.can_allocate(seq)
                else:
                    peer_work = None
                    step_work = None
                    if getattr(self.restore_policy, "queue_shapes", None):
                        peer_work = partial(self._peer_work, len(scheduled_seqs), num_batched_tokens, restored_batch_tokens)
                    if getattr(self.restore_policy, "step_curves", None):
                        step_work = partial(self._step_work, len(scheduled_seqs), num_batched_tokens, restored_batch_tokens)
                    num_cached_blocks, choice = self.block_manager.select_prefix(
                        seq, self.restore_policy, len(self.waiting) + len(self.running), len(self.running), peer_work,
                        step_work)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                restored = self.block_manager.allocate(seq, num_cached_blocks)
                if restored == -1:
                    continue
                cpu_tokens = restored * self.block_size
                restored_batch_tokens += cpu_tokens
                if seq.metrics is not None:
                    gpu_tokens = seq.num_cached_tokens - cpu_tokens
                    candidate_tokens = cpu_tokens
                    if choice is not None:
                        shape, decision = choice
                        candidate_tokens = shape.cpu_tokens
                        seq.metrics.restore_decisions.append(dict(
                            shape=asdict(shape), **asdict(decision), allocation=seq.metrics.allocations,
                            restored_tokens=cpu_tokens, recompute_candidate_tokens=candidate_tokens-cpu_tokens))
                    if seq.metrics.allocations == 0:
                        seq.metrics.initial_gpu_hit_tokens = gpu_tokens
                        seq.metrics.initial_cpu_hit_tokens = candidate_tokens
                    seq.metrics.allocation_gpu_hit_tokens += gpu_tokens
                    seq.metrics.cpu_restored_tokens += cpu_tokens
                    seq.metrics.allocations += 1
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        if seq.metrics is not None:
            seq.metrics.preemptions += 1
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            if seq.metrics is not None:
                if is_prefill:
                    seq.metrics.prefill_tokens += seq.num_scheduled_tokens
                    seq.metrics.prompt_computed_tokens += max(0, min(
                        seq.num_scheduled_tokens, seq.num_prompt_tokens - seq.num_cached_tokens))
                else:
                    seq.metrics.decode_tokens += seq.num_scheduled_tokens
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if seq.metrics is not None:
                seq.metrics.token_times.append(perf_counter())
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                if seq.metrics is not None:
                    seq.metrics.finish_time = perf_counter()
