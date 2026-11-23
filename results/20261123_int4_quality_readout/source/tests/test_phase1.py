import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nanovllm.engine.metrics import RequestMetrics
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.sampling_params import SamplingParams


def scheduler(blocks=8, batch=256):
    return Scheduler(SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=batch,
                                    eos=-1, kvcache_block_size=256, num_kvcache_blocks=blocks))


def request(tokens, output=2):
    seq = Sequence(tokens, SamplingParams(temperature=0, max_tokens=output, ignore_eos=True))
    seq.metrics = RequestMetrics(seq.seq_id, len(tokens), 0)
    return seq


def finish(sched):
    steps = 0
    while not sched.is_finished():
        seqs, prefill = sched.schedule()
        sched.postprocess(seqs, [17] * len(seqs), prefill)
        steps += 1
        assert steps < 100, "scheduler did not make progress"


class MetricsTests(unittest.TestCase):
    def test_partial_cold_peer_fills_budget_and_keeps_fifo_and_tail(self):
        for pack in (False, True):
            sched = scheduler(blocks=8, batch=640)
            sched.pack_cold = pack
            first, second = request([1] * 260), request([2] * 900)
            sched.add(first)
            sched.add(second)
            seqs, prefill = sched.schedule()
            self.assertEqual([s.num_scheduled_tokens for s in seqs], [260, 380] if pack else [260])
            sched.postprocess(seqs, [17] * len(seqs), prefill)
            self.assertEqual(second.num_cached_tokens, 380 if pack else 0)
            self.assertEqual(second.metrics.token_times, [])
            self.assertIs(sched.waiting[0], second)
            finish(sched)
            self.assertEqual(second.metrics.prompt_computed_tokens, 900)
            self.assertEqual(len(second.metrics.token_times), 2)
            replay = request([2] * 899 + [3])
            sched.add(replay)
            finish(sched)
            self.assertEqual(replay.metrics.initial_gpu_hit_tokens, 768)

    def test_partial_admission_waits_for_peer_to_publish_shared_root(self):
        sched = scheduler(blocks=8, batch=640)
        sched.pack_cold = True
        first, second = request([1] * 260), request([1] * 256 + [2] * 644)
        sched.add(first)
        sched.add(second)
        seqs, prefill = sched.schedule()
        self.assertEqual(seqs, [first])
        sched.postprocess(seqs, [17], prefill)
        finish(sched)
        self.assertEqual(second.metrics.initial_gpu_hit_tokens, 256)
        self.assertEqual(second.metrics.prompt_computed_tokens, 644)

    def test_partial_admission_keeps_tiny_remaining_budget_unused(self):
        sched = scheduler(blocks=8, batch=640)
        sched.pack_cold = True
        first, second = request([1] * 400), request([2] * 900)
        sched.add(first)
        sched.add(second)
        seqs, prefill = sched.schedule()
        self.assertEqual(seqs, [first])
        sched.postprocess(seqs, [17], prefill)
        finish(sched)

    def test_timing_definitions(self):
        metrics = RequestMetrics(1, 256, 1.0, token_times=[2.0, 2.25, 2.75], finish_time=3.0)
        row = metrics.as_dict(origin=0.5)
        self.assertEqual(row["ttft_ms"], 1000)
        self.assertEqual(row["tpot_ms"], 375)
        self.assertEqual(row["itl_ms"], [250, 500])
        self.assertEqual(row["enqueue_s"], 0.5)
        self.assertEqual(row["finish_s"], 2.5)

    def test_chunk_does_not_emit_and_boundary_hit_retains_query(self):
        for length in (255, 256, 257, 512, 513):
            with self.subTest(length=length):
                sched = scheduler()
                cold = request(list(range(length)))
                sched.add(cold)
                seqs, prefill = sched.schedule()
                sched.postprocess(seqs, [17], prefill)
                self.assertEqual(len(cold.metrics.token_times), int(length <= 256))
                finish(sched)
                warm = request(list(range(length)))
                sched.add(warm)
                finish(sched)
                hit = ((length - 1) // 256) * 256
                self.assertEqual(warm.metrics.initial_gpu_hit_tokens, hit)
                self.assertEqual(warm.metrics.prefill_tokens, length - hit)
                self.assertEqual(cold.metrics.prefill_tokens, length)
                self.assertEqual(warm.metrics.decode_tokens, 1)
                self.assertEqual(len(warm.metrics.token_times), 2)
                self.assertGreaterEqual(warm.metrics.finish_time, warm.metrics.token_times[-1])

    def test_shared_refs_survive_one_finish_and_eviction(self):
        sched = scheduler(blocks=3, batch=2048)
        base = request([1] * 257)
        sched.add(base)
        finish(sched)
        first, second = request([1] * 257, 1), request([1] * 256 + [2], 2)
        sched.add(first)
        sched.add(second)
        seqs, prefill = sched.schedule()
        shared = first.block_table[0]
        self.assertEqual(second.block_table[0], shared)
        self.assertEqual(sched.block_manager.blocks[shared].ref_count, 2)
        sched.postprocess(seqs, [17, 17], prefill)
        self.assertEqual(sched.block_manager.blocks[shared].ref_count, 1)
        finish(sched)
        self.assertFalse(sched.block_manager.used_block_ids)
        for value in (3, 4, 5):
            seq = request([value] * 257)
            sched.add(seq)
            finish(sched)
        self.assertEqual(sched.block_manager.can_allocate(request([1] * 257)), 0)

    def test_preemption_counts_recomputed_generated_token(self):
        sched = scheduler(blocks=4)
        seq = request([1] * 257, 3)
        sched.add(seq)
        while not seq.num_completion_tokens:
            seqs, prefill = sched.schedule()
            sched.postprocess(seqs, [17], prefill)
        sched.running.remove(seq)
        sched.preempt(seq)
        finish(sched)
        self.assertEqual(seq.metrics.preemptions, 1)
        self.assertEqual(seq.metrics.allocations, 2)
        self.assertEqual(seq.metrics.initial_gpu_hit_tokens, 0)
        self.assertEqual(seq.metrics.allocation_gpu_hit_tokens, 256)
        self.assertEqual(seq.metrics.prefill_tokens, 259)
        self.assertEqual(seq.metrics.prompt_computed_tokens, 258)
        self.assertEqual(len(seq.metrics.token_times), 3)

    def test_metrics_disabled_has_same_output_and_cache_state(self):
        sched = scheduler()
        seq = request([1] * 513)
        seq.metrics = None
        sched.add(seq)
        finish(sched)
        self.assertEqual(seq.completion_token_ids, [17, 17])
        self.assertFalse(sched.block_manager.used_block_ids)


class BudgetAndSamplingTests(unittest.TestCase):
    def test_admission_at_last_output_kv_boundary(self):
        engine = LLMEngine.__new__(LLMEngine)
        engine.config = SimpleNamespace(max_model_len=512, num_kvcache_blocks=1, enable_metrics=False)
        engine.scheduler = scheduler(blocks=1)
        # The final emitted output does not itself need a KV slot.
        engine.add_request([1] * 256, SamplingParams(max_tokens=1))
        for prompt, output in (([], 1), ([1] * 256, 2), ([1] * 512, 1)):
            with self.subTest(length=len(prompt), output=output), self.assertRaises(ValueError):
                engine.add_request(prompt, SamplingParams(max_tokens=output))

    def test_cap_never_exceeds_safe_budget(self):
        for cap, expected in ((None, 10), (4, 4), (20, 10)):
            runner = ModelRunner.__new__(ModelRunner)
            runner.world_size = 1
            runner.block_size = 256
            hf = SimpleNamespace(num_key_value_heads=1, head_dim=1, hidden_size=1,
                                 num_attention_heads=1, num_hidden_layers=1, dtype=torch.bfloat16)
            runner.config = SimpleNamespace(hf_config=hf, gpu_memory_utilization=1,
                                            max_kvcache_blocks=cap, num_kvcache_blocks=-1)
            runner.model = SimpleNamespace(modules=lambda: [])
            with patch("torch.cuda.mem_get_info", return_value=(10240, 10240)), \
                 patch("torch.cuda.memory_stats", return_value={"allocated_bytes.all.peak": 0, "allocated_bytes.all.current": 0}), \
                 patch("torch.empty", return_value=None):
                runner.allocate_kv_cache()
            self.assertEqual(runner.config.num_kvcache_blocks, expected)
            self.assertEqual(runner.kv_cache_budget["safe_blocks"], 10)

    def test_mixed_greedy_and_sampling_finite(self):
        logits = torch.tensor([[1.0, 10.0, -3.0], [5.0, -2.0, 1.0]])
        temperatures = torch.tensor([0.0, 0.7])
        sampler = Sampler()
        for seed in range(3):
            torch.manual_seed(seed)
            result = sampler(logits.clone(), temperatures)
            self.assertEqual(result[0].item(), 1)
            self.assertIn(result[1].item(), (0, 1, 2))
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(AssertionError):
                SamplingParams(temperature=value)


if __name__ == "__main__":
    unittest.main()
