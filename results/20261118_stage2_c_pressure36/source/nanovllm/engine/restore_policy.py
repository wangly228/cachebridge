from dataclasses import dataclass
from importlib.metadata import version
import json
from math import isfinite
from pathlib import Path

import torch


@dataclass(frozen=True, slots=True)
class CostShape:
    gpu_prefix_tokens: int
    cpu_tokens: int
    gpu_after_cpu_tokens: int
    query_tokens: int
    outstanding: int
    running: int


@dataclass(frozen=True, slots=True)
class RestoreDecision:
    action: str
    reason: str
    restore_ms: float | None = None
    recompute_ms: float | None = None


def cost_environment(config, namespace):
    return dict(namespace=namespace.hex(), gpu=torch.cuda.get_device_name(),
                capability=list(torch.cuda.get_device_capability()), torch=str(torch.__version__),
                cuda=torch.version.cuda, flash_attention=version("flash-attn"),
                block_size=config.kvcache_block_size, gpu_blocks=config.num_kvcache_blocks,
                cpu_blocks=config.cpu_prefix_cache_blocks, eager=config.enforce_eager,
                max_num_batched_tokens=config.max_num_batched_tokens, max_num_seqs=config.max_num_seqs,
                max_model_len=config.max_model_len, tensor_parallel_size=config.tensor_parallel_size)


class RestoreCostTable:
    """Exact measured buckets. Restore only if its p90 beats recomputation's p10."""

    def __init__(self, data, environment):
        if data.get("version") != 1 or data.get("environment") != environment:
            raise ValueError("restore cost table version or engine environment does not match")
        self.buckets = {}
        for row in data["buckets"]:
            shape = CostShape(**row["shape"])
            dims = (shape.gpu_prefix_tokens, shape.cpu_tokens, shape.gpu_after_cpu_tokens)
            if (any(type(x) is not int or x < 0 or x % environment["block_size"] for x in dims)
                    or shape.cpu_tokens == 0 or type(shape.query_tokens) is not int or shape.query_tokens < 1
                    or type(shape.running) is not int or type(shape.outstanding) is not int
                    or not 0 <= shape.running < shape.outstanding or shape in self.buckets
                    or type(row["paired_waves"]) is not int or row["paired_waves"] < 3):
                raise ValueError(f"invalid or duplicate restore cost bucket {shape}")
            costs = [row[name] for name in ("restore_p50_ms", "restore_p90_ms", "recompute_p10_ms", "recompute_p50_ms")]
            if any(type(v) not in (int, float) or not isfinite(v) or v <= 0 for v in costs):
                raise ValueError("expected finite positive calibration times")
            restore, upper, lower, recompute = costs
            if restore > upper or lower > recompute:
                raise ValueError("inconsistent calibration quantiles")
            if upper < lower:
                choice = RestoreDecision("restore", "measured_restore", restore, recompute)
            else:
                reason = "measured_recompute" if recompute <= restore else "overlapping_costs"
                choice = RestoreDecision("recompute", reason, restore, recompute)
            self.buckets[shape] = choice
        if not self.buckets:
            raise ValueError("empty restore cost table")

    @classmethod
    def load(cls, path, environment):
        return cls(json.loads(Path(path).read_text()), environment)

    def decide(self, shape):
        decision = self.buckets.get(shape)
        if decision is None:
            return RestoreDecision("recompute", "uncalibrated_shape")
        return decision
