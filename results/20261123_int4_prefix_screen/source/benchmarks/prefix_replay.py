"""Replay recorded waves through cachebridge's scheduler and both cache tiers without CUDA."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import Future
import json
from math import ceil, floor
from pathlib import Path
import shlex
import sys
from time import perf_counter

import numpy as np
from types import SimpleNamespace

from benchmarks.common import ROOT, sha256, write_json, write_jsonl
from nanovllm.engine.cpu_prefix_cache import TieredBlockManager, prefix_key
from nanovllm.engine.metrics import RequestMetrics
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


REFERENCE = "https://github.com/cacheMon/PrefixSim/tree/160d5f81724dd9ec28b10d6abb64615e6adbb332"
COUNTERS = ("initial_gpu_hit_tokens", "initial_cpu_hit_tokens", "allocation_gpu_hit_tokens", "allocations",
            "cpu_restored_tokens", "prompt_computed_tokens", "prefill_tokens", "decode_tokens", "preemptions",
            "d2h_blocks", "h2d_blocks", "d2h_batches", "batched_d2h_blocks", "cpu_save_failures", "cpu_restore_failures")


class NoPayload:
    """Only model slot readiness. Block lifetime and completion barriers stay in the real manager."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.written = set()

    def save(self, gpu_id, slot):
        self.written.add(slot)

    def save_many(self, pairs):
        self.written.update(slot for _, slot in pairs)

    def submit_save_many(self, pairs):
        self.save_many(pairs)
        done = Future()
        done.set_result(perf_counter())
        return done

    def restore(self, slot, gpu_id):
        assert slot in self.written


class ObservedCache(TieredBlockManager):
    """Tap the production admission, lookup, allocation, eviction and publication calls."""
    def start_cycle(self, tag, policy):
        self.policy = policy
        self.parents = {}
        self.tag = tag
        self.wave = None
        self.step = 0
        self.actor = None
        self.history = defaultdict(list)
        self.events = []
        self.seen = {}
        self.admissions = {}

    def event(self, key, action, **fields):
        e = dict(**self.tag, wave=self.wave, step=self.step, by_seq=self.actor,
                 event=action, block_key=key.hex(), **fields)
        self.history[key].append(e)
        self.events.append(e)

    def allocate(self, seq, num_cached_blocks):
        self.actor = seq.seq_id
        if seq.metrics.allocations == 0:
            hits = self._lookup(seq)
            parent = self.namespace
            keys = []
            # Full prompt pages only, always leave the last upstream query block.
            for i in range(min(seq.num_blocks - 1, seq.num_prompt_tokens // self.block_size)):
                child = prefix_key(parent, seq.block(i))
                self.parents[child] = parent
                parent = child
                keys.append(parent)
            count = len(keys)
            reused = bool(keys and keys[-1] in self.seen)
            record = dict(prefix_blocks=count, revisited_prefix=reused,
                          matched_blocks=num_cached_blocks, matched_tiers=["GPU" if h.gpu_id is not None else "CPU" for h in hits],
                          required_query_tail_tokens=seq.num_prompt_tokens-count*self.block_size)
            if reused and num_cached_blocks < count:
                pos = num_cached_blocks
                key = keys[pos]
                history = self.history[key].copy()
                published = any(e["event"] == "cpu_published" for e in history)
                rejected = any(e["event"] == "cpu_deferred_by_admission" for e in history)
                evicted = any(e["event"] == "cpu_lru_evicted" for e in history)
                reason = "previously_written_evicted" if evicted else "never_written_admission"
                if published and not evicted or not published and not rejected:
                    reason = "other_or_unresolved"
                # Count CPU pages stranded beyond a hole separately from pages actually recovered.
                # Current epochs preserve key -> tokens, and sync/overlap perform no data conversion.
                stranded = [i for i in range(pos+1, count) if keys[i] in self.entries]
                record["gap"] = dict(block_index_zero_based=pos, key=key.hex(), first_request=self.seen[key],
                                     lost_prefix_tokens=(count-pos)*self.block_size,
                                     downstream_tokens=(count-pos-1)*self.block_size,
                                     stranded_cpu_indices=stranded, cause=reason, history=history)
            self.admissions[seq.seq_id] = record
            for key in keys:
                self.seen.setdefault(key, dict(wave=self.wave, seq_id=seq.seq_id))
        return super().allocate(seq, num_cached_blocks)

    def _allocate_block(self):
        victim = self.free_block_ids[0]
        block = self.blocks[victim]
        key = self.gpu_keys[victim]
        if key is not None and self.hash_to_block_id.get(block.hash) == victim:
            self.event(key, "gpu_page_reclaimed", gpu_id=victim)
        return super()._allocate_block()

    def may_append(self, seq):
        self.actor = seq.seq_id
        super().may_append(seq)

    def hash_blocks(self, seq):
        self.actor = seq.seq_id
        if self.policy == "leaf_lru":
            # An exactly aligned last query block can be stored but was excluded from lookup.
            start = seq.num_cached_tokens // self.block_size
            end = min((seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size,
                      seq.num_prompt_tokens // self.block_size)
            parent = self.gpu_keys[seq.block_table[start-1]] if start else self.namespace
            for pos in range(start, end):
                key = prefix_key(parent, seq.block(pos))
                self.parents[key] = parent
                parent = key
        super().hash_blocks(seq)

    def deallocate(self, seq):
        self.actor = seq.seq_id
        super().deallocate(seq)

    def _admit(self, key, block, seq):
        old = self.recent_uses.get(key)
        super()._admit(key, block, seq)
        if key not in self.entries and self.recent_uses.get(key) == seq.seq_id:
            self.event(key, "cpu_deferred_by_admission", prior_observer=old, resident=len(self.entries),
                       limit=self.first_use_limit, gpu_id=block.block_id)

    def _save(self, key, block, metrics):
        victim = None
        if not self.free_slots and key not in self.entries:
            if self.policy == "leaf_lru":
                # Preserve earlier pages wherever a resident descendant depends on them.
                # Only use parents already observed through the current allocation.
                ancestors = set()
                for resident in self.entries:
                    parent = self.parents.get(resident)
                    while parent is not None and parent not in ancestors:
                        ancestors.add(parent)
                        parent = self.parents.get(parent)
                oldest_leaf = next(k for k in self.entries if k not in ancestors)
                self.entries.move_to_end(oldest_leaf, last=False)
            victim = next(iter(self.entries))
        super()._save(key, block, metrics)
        if victim is not None and victim not in self.entries:
            self.event(victim, "cpu_lru_evicted", replacement_key=key.hex())
        if key in self.entries:
            self.event(key, "cpu_published", owner_seq=metrics.seq_id, slot=self.entries[key][0])

    def _save_new_pages(self, pages, metrics):
        saved = super()._save_new_pages(pages, metrics)
        if saved and self.pending_write:
            for (key, block), slot in zip(pages, self.pending_write[2], strict=True):
                self.event(key, "cpu_slot_reserved", gpu_id=block.block_id, slot=slot, owner_seq=metrics.seq_id)
        return saved

    def _publish_group(self, pages, slots, metrics, elapsed):
        super()._publish_group(pages, slots, metrics, elapsed)
        for (key, _), slot in zip(pages, slots, strict=True):
            self.event(key, "cpu_published", owner_seq=metrics.seq_id, slot=slot)


def read_lines(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def require_equal(actual, expected, where):
    if actual != expected:
        raise ValueError(f"{where}: observed {actual}, expected {expected}")


def run_wave(scheduler, cache, requests, expected_steps, wave, out_rows):
    sequences = {}
    for r in requests:
        if (len(r["input_token_ids"]) + r["output_tokens"] - 2) // cache.block_size + 1 > len(cache.blocks):
            raise ValueError(f"Request {r['seq_id']} cannot fit in the GPU pool")
        seq = Sequence(r["input_token_ids"], SamplingParams(temperature=0, max_tokens=r["output_tokens"], ignore_eos=True))
        seq.seq_id = r["seq_id"]
        seq.metrics = RequestMetrics(seq.seq_id, len(seq), 0)
        scheduler.add(seq)
        sequences[seq.seq_id] = seq, r
    simulated_steps = 0
    while not scheduler.is_finished():
        step_index = simulated_steps
        simulated_steps += 1
        cache.step = step_index
        seqs, prefill = scheduler.schedule()
        ids = [s.seq_id for s in seqs]
        shape = (prefill, sum(s.num_scheduled_tokens for s in seqs), ids, len(cache.used_block_ids))
        if expected_steps is not None:
            actual = expected_steps[step_index]
            reference = tuple(actual[k] for k in ("is_prefill", "scheduled_tokens", "seq_ids", "used_blocks"))
            require_equal(shape, reference, (cache.tag, cache.wave, step_index))
        # postprocess ignores the token in incomplete prefill chunks.
        generated = [sequences[s.seq_id][1]["output_token_ids"][s.num_completion_tokens] for s in seqs]
        scheduler.postprocess(seqs, generated, prefill)
    require_equal(scheduler.is_finished(), True, "wave drained")
    require_equal(cache.pending_write, None, "in-flight writes at wave end")
    if expected_steps is not None:
        require_equal(simulated_steps, len(expected_steps), "step count")
        require_equal((len(cache.entries), cache.evictions), (wave["cpu_resident_blocks"], wave["cpu_evictions_since_reset"]),
                      (cache.tag, cache.wave, "end occupancy"))
    for fifo, (seq, r) in enumerate(sequences.values()):
        counts = {k: getattr(seq.metrics, k) for k in COUNTERS}
        if expected_steps is not None:
            require_equal(counts, {k: r[k] for k in COUNTERS}, seq.seq_id)
        require_equal(seq.completion_token_ids, r["output_token_ids"], seq.seq_id)
        record = dict(**cache.tag, wave=cache.wave, fifo_zero_based=fifo, seq_id=seq.seq_id,
                      **counts, **cache.admissions[seq.seq_id])
        if expected_steps is not None:
            record["measured_ttft_ms"] = r["ttft_ms"]
        out_rows.append(record)


def recorded_waves(run_dir, requested_arm):
    waves = read_lines(run_dir / "waves.jsonl")
    boundary = "wave_index" in waves[0]
    wave_field = "wave_index" if boundary else "phase"
    available = {w["arm"] for w in waves if w["cpu_cache_enabled"]}
    if requested_arm is not None:
        arm = requested_arm
    elif boundary:
        arm = "cold_pack" if "cold_pack" in available else "CPU"
    else:
        arm = "candidate"
    if arm not in available:
        raise ValueError(f"Select one of the recorded CPU-policy arms {sorted(available)}")
    group_fields = ("warmup", "seed", "revisit_waves") if boundary else ("warmup", "seed", "cycle", "workload")
    fields = (*group_fields, wave_field)
    reqs, steps = defaultdict(list), defaultdict(list)
    for name, index in (("requests", reqs), ("steps", steps)):
        for r in read_lines(run_dir / (name + ".jsonl")):
            if r["arm"] == arm and not r["warmup"]:
                index[tuple(r[k] for k in fields)].append(r)
    groups = defaultdict(list)
    for wave in waves:
        if wave["arm"] == arm and not wave["warmup"]:
            groups[tuple(wave[k] for k in group_fields)].append(wave)

    return groups, group_fields, wave_field, arm, reqs, steps


def new_waves(path):
    """One JSONL row per wave: cycle, wave, requests with input_token_ids and output_token_ids."""
    groups, reqs = defaultdict(list), {}
    ident = 0
    for item in read_lines(path):
        key = (item["cycle"], item["wave"])
        if key in reqs or not item["requests"]:
            raise ValueError(f"Duplicate or empty wave {key}")
        batch = []
        for r in item["requests"]:
            if not r["input_token_ids"] or not r["output_token_ids"]:
                raise ValueError(f"Provide nonempty prompt and frozen output IDs for wave {key}")
            ident += 1
            batch.append(dict(seq_id=ident, input_token_ids=r["input_token_ids"],
                              output_token_ids=r["output_token_ids"], output_tokens=len(r["output_token_ids"])))
        reqs[key] = batch
        groups[(key[0],)].append(dict(wave=key[1]))
    return groups, ("cycle",), "wave", "transcript", reqs, {}


def arm_settings(engine, arm):
    cfg = engine["config"]
    arms = engine.get("arms", {}) or engine.get("modes", {}) or {"CPU": None}
    if arm not in arms:
        raise ValueError(f"Unknown CPU trace arm {arm!r}. Recorded arms: {sorted(arms)}")
    details = engine.get("arms", {}).get(arm, {})
    mode = details.get("transfer_mode", engine.get("modes", {}).get(arm, cfg.get("cpu_prefix_transfer_mode", "sync")))
    pack = details.get("pack_cold", engine.get("pack_cold_by_arm", {}).get(arm, cfg.get("cpu_prefix_pack_cold", False)))
    capacity = details.get("cpu_cache_blocks", engine["cpu_slots"])
    if mode not in ("overlap", "sync") or type(pack) is not bool or not 0 < capacity <= engine["cpu_slots"]:
        raise ValueError(f"Unsupported CPU configuration for {arm}: {details}, mode={mode}, pack_cold={pack}")
    return mode, pack, capacity


def run(run_dir, out, policy, transcript=None, requested_arm=None):
    engine = json.loads((run_dir / "engine.json").read_text())
    cfg = engine["config"]
    if cfg.get("cpu_restore_cost_table"):
        raise ValueError("Replay supports hit-and-restore policy B")
    Sequence.block_size = engine.get("block_tokens", cfg.get("kvcache_block_size", 256))
    budget = engine.get("actual_gpu_budget", engine.get("kv_budget"))
    loader = new_waves(transcript) if transcript is not None else recorded_waves(run_dir, requested_arm)
    groups, group_fields, wave_field, arm, reqs, steps = loader
    if transcript is not None:
        arm = requested_arm or ("candidate" if "candidate" in engine.get("modes", {}) else
                                "cold_pack" if "cold_pack" in engine.get("arms", {}) else "CPU")
    mode, pack_cold, capacity = arm_settings(engine, arm)
    calibrate = transcript is None and policy == "baseline"
    all_rows, wave_count = [], 0
    out.mkdir(parents=True, exist_ok=False)
    for group_id, cycle in groups.items():
        tag = dict(zip(group_fields, group_id, strict=True))
        scheduler = Scheduler(SimpleNamespace(max_num_seqs=cfg["max_num_seqs"],
                              max_num_batched_tokens=cfg["max_num_batched_tokens"], eos=-1,
                              kvcache_block_size=Sequence.block_size, num_kvcache_blocks=budget["actual_blocks"],
                              cpu_prefix_pack_cold=pack_cold))
        # Namespace changes labels only; this single-engine replay never matches across model namespaces.
        namespace = bytes.fromhex(engine["namespace"]) if "namespace" in engine else b"cachebridge-metadata-replay-v1"
        cache = ObservedCache(budget["actual_blocks"], Sequence.block_size, NoPayload(capacity), namespace,
                              cfg["cpu_prefix_admission"], cfg["cpu_prefix_batch_writes"],
                              cfg["cpu_prefix_first_use_blocks"], mode)
        scheduler.block_manager = cache
        if policy == "all_free":
            # Deliberately relax the engine's conservative clamp to GPU pool size.
            # Physical GPU blocks and CPU slots remain identical to the calibrated run.
            cache.first_use_limit = cache.transfer.capacity
        cache.start_cycle(tag, policy)
        for wave in cycle:
            if "pack_cold" in wave:
                require_equal(pack_cold, wave["pack_cold"], (arm, "pack_cold"))
            if "transfer_mode" in wave:
                require_equal(mode, wave["transfer_mode"], (arm, "transfer_mode"))
            cache.wave = wave[wave_field]
            key = (*group_id, cache.wave)
            expected_steps = steps.pop(key, None)
            if calibrate and expected_steps is None:
                raise ValueError(f"Missing calibration steps for {key}")
            run_wave(scheduler, cache, reqs.pop(key), expected_steps if calibrate else None, wave, all_rows)
            wave_count += 1
        write_jsonl(out / "events.jsonl", cache.events)
    if reqs or steps or not all_rows:
        raise ValueError("Missing waves or unexpected input schema")
    write_jsonl(out / "requests.jsonl", all_rows)
    gaps = [r for r in all_rows if "gap" in r]
    summary = {}
    for group in sorted({r.get("revisit_waves", r.get("workload", "all")) for r in all_rows}):
        selected = [r for r in all_rows if r.get("revisit_waves", r.get("workload", "all")) == group]
        misses = [r["gap"] for r in selected if "gap" in r]
        group_summary = dict(n=len(selected), revisits=sum(r["revisited_prefix"] for r in selected),
                             incomplete=len(misses), causes=dict(Counter(g["cause"] for g in misses)),
                             lost_tokens_by_cause={cause:sum(g["lost_prefix_tokens"] for g in misses if g["cause"]==cause)
                                                  for cause in sorted({g["cause"] for g in misses})},
                             first_gap_indices=dict(Counter(g["block_index_zero_based"] for g in misses)),
                             stranded_cpu_blocks=sum(len(g["stranded_cpu_indices"]) for g in misses),
                             gpu_hit_tokens=sum(r["initial_gpu_hit_tokens"] for r in selected),
                             d2h_blocks=sum(r["d2h_blocks"] for r in selected),
                             h2d_blocks=sum(r["h2d_blocks"] for r in selected),
                             prompt_computed_tokens=sum(r["prompt_computed_tokens"] for r in selected))
        summary[str(group)] = group_summary
        if calibrate:
            tts = sorted(selected, key=lambda x:x["measured_ttft_ms"])
            cutoff = .95*(len(tts)-1)
            def describe(r):
                return {k:r[k] for k in (*group_fields, "wave", "seq_id","measured_ttft_ms","revisited_prefix")}
            group_summary.update(measured_p95_ms=float(np.percentile([r["measured_ttft_ms"] for r in tts],95)),
                    measured_p95_support=[describe(tts[i]) for i in sorted({floor(cutoff), ceil(cutoff)})],
                    slowest_five_percent=[describe(r) for r in tts[-ceil(.05*len(tts)):]],
                    # Rank sensitivity only. This even erases all original GPU/H2D and residual tail work on revisits.
                    idealized_zero_revisit_latency_p95_ms=float(np.percentile(
                        [0 if r["revisited_prefix"] else r["measured_ttft_ms"] for r in tts],95)))
    inputs = ("requests.jsonl", "waves.jsonl", "steps.jsonl", "engine.json", "design.json", "trace.json")
    input_hashes = {f: sha256(run_dir/f) for f in (inputs if transcript is None else ("engine.json",))}
    if transcript is not None:
        input_hashes[str(transcript)] = sha256(transcript)
    fps = ("nanovllm/engine/cpu_prefix_cache.py","nanovllm/engine/block_manager.py","nanovllm/engine/scheduler.py",
           "nanovllm/engine/sequence.py","nanovllm/engine/metrics.py")
    (out / "prefix_replay.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "summary.json", dict(reference=REFERENCE, transfer_mode=mode, pack_cold=pack_cold,
               cpu_slots=capacity, input=str(run_dir), arm=arm, groups=summary,
               policy=policy, replayed_requests=len(all_rows), replayed_waves=wave_count,
               calibrated=calibrate, request_counters=COUNTERS,
               gap_records=len(gaps), skipped_warmups=True, input_sha256=input_hashes,
               code_sha256={**{f:sha256(ROOT/f) for f in fps},"benchmarks/prefix_replay.py":sha256(__file__)},
               output_sha256={f:sha256(out/f) for f in ("requests.jsonl","events.jsonl")},
               publication="Simulation future completes immediately; production manager still reserves/publishes at its own barriers.",
               measurement="Baseline latency comes from the source log. Alternatives have no simulated latency or numeric output validation."))
    (out / "command.txt").write_text(shlex.join([sys.executable, "-m", "benchmarks.prefix_replay", *sys.argv[1:]])+"\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Complete recorded transfer_comparison or revisit_boundary run")
    p.add_argument("--output", type=Path, required=True, help="New directory for attribution and calibration results")
    p.add_argument("--policy", choices=("baseline", "all_free", "leaf_lru"), default="baseline",
                   help="Baseline checks request and step logs. Alternatives re-schedule at fixed physical capacities.")
    p.add_argument("--transcript", type=Path, help="Optional new JSONL waves. Each row supplies cycle, wave and "
                   "requests with input_token_ids/output_token_ids. Only engine.json is needed from --input; "
                   "the replay reports metadata without TTFT.")
    p.add_argument("--arm", help="Recorded CPU arm to replay (default: candidate, cold_pack or CPU, by run type)")
    args = p.parse_args()
    run(args.input, args.output, args.policy, args.transcript, args.arm)
