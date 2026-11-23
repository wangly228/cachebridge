"""Compare complete A/B traces offline and audit which full prompt blocks B wrote."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from benchmarks.common import distribution, write_json


def rows(folder, filename):
    with (folder / filename).open() as file:
        return [row for line in file if (row := json.loads(line))["phase"] != "warmup"]


def latency(requests):
    by_seed = defaultdict(list)
    for row in requests:
        by_seed[row["seed"]].append(row["ttft_ms"])
    return dict(ttft_ms=distribution([row["ttft_ms"] for row in requests]),
                p95_ms_by_seed={s: distribution(v)["p95"] for s, v in by_seed.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    engine = json.loads((args.run / "engine.json").read_text())
    a_root = Path(engine["baseline"])
    a = rows(a_root, "requests.jsonl")
    b = rows(args.run, "requests.jsonl")
    assert len(a) == len(b) == 96
    a_waves = rows(a_root, "waves.jsonl")
    waves = rows(args.run, "waves.jsonl")
    assert all(w["cpu_evictions_since_reset"] == 0 for w in waves)
    seen = defaultdict(set)
    identical = Counter()
    writes = refreshes = 0
    for x, y in zip(a, b, strict=True):
        for field in ("seed", "cycle", "phase", "input_token_ids"):
            assert x[field] == y[field]
        if x["output_token_ids"] == y["output_token_ids"]:
            identical[y["phase"]] += 1
        assert y["preemptions"] == y["cpu_save_failures"] == y["cpu_restore_failures"] == 0
        computed_from = y["initial_gpu_hit_tokens"] + y["initial_cpu_hit_tokens"]
        assert y["prompt_computed_tokens"] == y["prompt_tokens"] - computed_from
        keys = seen[y["seed"], y["cycle"]]
        count = 0
        for end in range(computed_from + 256, y["prompt_tokens"] + 1, 256):
            key = tuple(y["input_token_ids"][:end])
            refreshes += key in keys
            keys.add(key)
            count += 1
        writes += count
        assert count == y["d2h_blocks"]
    a_lat, b_lat = latency(a), latency(b)
    change = 100 * (b_lat["ttft_ms"]["p95"] / a_lat["ttft_ms"]["p95"] - 1)
    write_json(args.output, dict(A=a_lat, B=b_lat, pooled_p95_change_percent=change,
               same_phase_exact_output_requests=dict(identical), full_prompt_block_writes=writes,
               writes_of_previously_stored_prefix=refreshes, cpu_evictions=0,
               cpu_blocks_per_cycle=sorted(len(keys) for keys in seen.values()),
               A_memory_max_bytes={k: max(w["memory"][k] for w in a_waves) for k in waves[0]["memory"]},
               B_memory_max_bytes={k: max(w["memory"][k] for w in waves) for k in waves[0]["memory"]}))


if __name__ == "__main__":
    main()
