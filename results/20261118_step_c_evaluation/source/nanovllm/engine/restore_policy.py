from bisect import bisect_left
from dataclasses import dataclass, replace
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
    peer_query_tokens: int | None = None
    peer_cpu_tokens: int | None = None

    def without_peers(self):
        return replace(self, peer_query_tokens=None, peer_cpu_tokens=None)


@dataclass(frozen=True, slots=True)
class StepWork:
    seqs: int
    query_tokens: int
    cpu_tokens: int


@dataclass(frozen=True, slots=True)
class RestoreDecision:
    action: str
    reason: str
    restore_ms: float | None = None
    recompute_ms: float | None = None
    cost_basis: str | None = None
    step_work: dict[str, StepWork] | None = None


def cost_environment(config, namespace):
    return dict(namespace=namespace.hex(), gpu=torch.cuda.get_device_name(),
                capability=list(torch.cuda.get_device_capability()), torch=str(torch.__version__),
                cuda=torch.version.cuda, flash_attention=version("flash-attn"),
                block_size=config.kvcache_block_size, gpu_blocks=config.num_kvcache_blocks,
                cpu_blocks=config.cpu_prefix_cache_blocks, eager=config.enforce_eager,
                max_num_batched_tokens=config.max_num_batched_tokens, max_num_seqs=config.max_num_seqs,
                max_model_len=config.max_model_len, tensor_parallel_size=config.tensor_parallel_size)


class RestoreCostTable:
    """Measured costs. Restore only if its p90 beats recomputation's p10."""

    def __init__(self, data, environment):
        version = data.get("version")
        if version not in (1, 2, 3) or data.get("environment") != environment:
            raise ValueError("restore cost table version or engine environment does not match")
        self.buckets = {}
        self.queue_shapes = set()
        for row in data["buckets"]:
            shape = CostShape(**row["shape"])
            contextual = shape.peer_query_tokens is not None or shape.peer_cpu_tokens is not None
            if contextual:
                if (version < 2 or type(shape.peer_query_tokens) is not int or shape.peer_query_tokens < 0
                        or type(shape.peer_cpu_tokens) is not int or shape.peer_cpu_tokens < 0
                        or shape.peer_cpu_tokens % environment["block_size"]):
                    raise ValueError(f"invalid queue workload in restore cost bucket {shape}")
                self.queue_shapes.add(shape.without_peers())
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
            basis = "queue" if contextual else "shape"
            self.buckets[shape] = self._choice(restore, upper, lower, recompute, basis)
        if not self.buckets:
            raise ValueError("empty restore cost table")
        anchors = data.get("step_anchors", [])
        if bool(anchors) != (version == 3):
            raise ValueError("step cost anchors require a version 3 table")
        self.step_curves = {}
        for row in anchors:
            shape = CostShape(**row["shape"])
            action, seqs, cpu, query = (row[k] for k in ("action", "seqs", "cpu_tokens", "query_tokens"))
            quantiles = tuple(row[k] for k in ("p10_ms", "p50_ms", "p90_ms"))
            if (shape != shape.without_peers() or shape not in self.buckets or action not in ("restore", "recompute")
                    or type(seqs) is not int or not 1 <= seqs <= environment["max_num_seqs"]
                    or type(query) is not int or not 1 <= query <= environment["max_num_batched_tokens"]
                    or type(cpu) is not int or cpu < 0 or cpu % environment["block_size"]
                    or type(row["seed_count"]) is not int or row["seed_count"] < 3
                    or any(type(x) not in (float, int) or not isfinite(x) or x <= 0 for x in quantiles)
                    or not quantiles[0] <= quantiles[1] <= quantiles[2]):
                raise ValueError("invalid prefill step cost anchor")
            curve = self.step_curves.setdefault(shape, {}).setdefault((action, seqs, cpu), [])
            index = bisect_left(curve, query, key=lambda point: point[0])
            if index < len(curve) and curve[index][0] == query:
                raise ValueError("duplicate prefill step cost anchor")
            curve.insert(index, (query, quantiles))

    @staticmethod
    def _choice(restore, upper, lower, recompute, basis, work=None):
        if upper < lower:
            return RestoreDecision("restore", "measured_restore", restore, recompute, basis, work)
        reason = "measured_recompute" if recompute <= restore else "overlapping_costs"
        return RestoreDecision("recompute", reason, restore, recompute, basis, work)

    def decide_step(self, shape, work):
        """Interpolate only between measurements with the same batch size and H2D volume."""
        costs = {}
        for action, step in work.items():
            curve = self.step_curves[shape].get((action, step.seqs, step.cpu_tokens), [])
            q = step.query_tokens
            i = bisect_left(curve, q, key=lambda point: point[0])
            if i < len(curve) and curve[i][0] == q:
                costs[action] = curve[i][1]
            elif 0 < i < len(curve):
                left, low = curve[i-1]
                right, high = curve[i]
                weight = (q-left) / (right-left)
                costs[action] = tuple(a + weight*(b-a) for a, b in zip(low, high, strict=True))
            else:
                return None
        _, restore, upper = costs["restore"]
        lower, recompute, _ = costs["recompute"]
        return self._choice(restore, upper, lower, recompute, "step", work)

    @classmethod
    def load(cls, path, environment):
        return cls(json.loads(Path(path).read_text()), environment)

    def decide(self, shape):
        decision = self.buckets.get(shape)
        if decision is None and shape.peer_query_tokens is not None:
            decision = self.buckets.get(shape.without_peers())
        if decision is None:
            return RestoreDecision("recompute", "uncalibrated_shape")
        return decision
