from dataclasses import dataclass, field


@dataclass
class RequestMetrics:
    """Host timestamps: run() has materialized sampled IDs before postprocess.

    Prefill counts include recomputation after preemption. Initial GPU hits are
    kept separate from repeated allocations, which can also reuse generated KV.
    """

    seq_id: int
    prompt_tokens: int
    enqueue_time: float
    token_times: list[float] = field(default_factory=list)
    finish_time: float | None = None
    initial_gpu_hit_tokens: int = 0
    allocation_gpu_hit_tokens: int = 0
    allocations: int = 0
    prefill_tokens: int = 0
    prompt_computed_tokens: int = 0
    decode_tokens: int = 0
    preemptions: int = 0
    initial_cpu_hit_tokens: int = 0
    cpu_restored_tokens: int = 0
    d2h_blocks: int = 0
    h2d_blocks: int = 0
    d2h_ms: float = 0.
    h2d_ms: float = 0.
    cpu_save_failures: int = 0
    cpu_restore_failures: int = 0

    def as_dict(self, origin: float = 0):
        times = self.token_times
        return dict(
            seq_id=self.seq_id, prompt_tokens=self.prompt_tokens,
            enqueue_s=self.enqueue_time - origin,
            token_s=[t - origin for t in times],
            finish_s=None if self.finish_time is None else self.finish_time - origin,
            output_tokens=len(times),
            ttft_ms=(times[0] - self.enqueue_time) * 1000 if times else None,
            tpot_ms=(times[-1] - times[0]) * 1000 / (len(times) - 1) if len(times) > 1 else None,
            itl_ms=[(b - a) * 1000 for a, b in zip(times, times[1:])],
            initial_gpu_hit_tokens=self.initial_gpu_hit_tokens,
            allocation_gpu_hit_tokens=self.allocation_gpu_hit_tokens,
            allocations=self.allocations, prefill_tokens=self.prefill_tokens,
            prompt_computed_tokens=self.prompt_computed_tokens,
            decode_tokens=self.decode_tokens, preemptions=self.preemptions,
            initial_cpu_hit_tokens=self.initial_cpu_hit_tokens, cpu_restored_tokens=self.cpu_restored_tokens,
            d2h_blocks=self.d2h_blocks, h2d_blocks=self.h2d_blocks,
            d2h_ms=self.d2h_ms, h2d_ms=self.h2d_ms,
            cpu_save_failures=self.cpu_save_failures, cpu_restore_failures=self.cpu_restore_failures,
        )
