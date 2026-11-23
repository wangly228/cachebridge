"""Counterbalance B, the frozen queue policy and the step policy on held-out boundary traces."""
import argparse
from itertools import permutations
import json
from pathlib import Path
from shutil import copyfile
from tempfile import mkdtemp
import traceback

import torch

from nanovllm import LLM
from nanovllm.engine.restore_policy import RestoreCostTable, cost_environment
from benchmarks.common import prepare_run, sha256, write_json
from benchmarks.queue_evaluation import prefix_roots
from benchmarks.queue_calibration import RecomputeOne, engine_config, run_cycle
from benchmarks.restore_holdout import oracle
from benchmarks.step_calibration import BASE_SHA, EVAL_SEEDS, EVAL_SIZES, LENGTHS, OLD_PROBE, TARGET, observe, trace_for


ARMS = ("B", "C2", "C3")
PHASES = ("fill", "interference", "evicted")


def read_json(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_cycle(folder):
    waves = lines(folder/"waves.jsonl")
    rows, steps = (lines(folder/f"{name}.jsonl") for name in ("requests", "steps"))
    assert [w["phase"] for w in waves] == list(PHASES)
    batches = [([r for r in rows if r["phase"] == phase], [s for s in steps if s["phase"] == phase], w)
               for phase, w in zip(PHASES, waves, strict=True)]
    return batches, read_json(folder/"state.json")["sha256"]


def cycle(llm, folder, trace, policy, tokens, tag):
    # Publish only complete arms. A restarted run can read each already finished arm from disk.
    if folder.is_dir():
        return load_cycle(folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(mkdtemp(dir=folder.parent, prefix=f".pending-{folder.name}-"))
    batches, state = run_cycle(llm, attempt, trace, policy, tokens, tag, cold_peers=(3,))
    observe(batches[-1], tag["cold_peer_tokens"])
    write_json(attempt/"state.json", dict(sha256=state))
    attempt.rename(folder)
    return batches, state


def same_output(ref, measured):
    fields = ("input_token_ids", "output_token_ids", "cpu_restored_tokens", "prompt_computed_tokens")
    return all(all(a[k] == b[k] for k in fields) for a, b in zip(ref, measured, strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.module = "benchmarks.step_evaluation"
    out = Path(args.output) if args.resume else prepare_run(args.output, args, args.model)
    cal = Path(args.calibration)
    status = read_json(cal/"status.json")
    assert status["passed"] and sha256(cal/"v2.json") == BASE_SHA
    llm = None
    try:
        if args.resume:
            env = read_json(out/"environment.json")
            assert Path(args.model).resolve() == Path(env["model_path"])
            assert all(sha256(Path(f)) == h for f, h in env["source_sha256"].items() if f.startswith("nanovllm/"))
        for name, check in (("v2.json", BASE_SHA), ("cost_table.json", status["table_sha256"])):
            if not (out/name).exists():
                copyfile(cal/name, out/name)
            assert sha256(out/name) == sha256(cal/name) == check
        config = dict(engine_config(), cpu_restore_cost_table=str(out/"cost_table.json"))
        pool = "step-eval-v3"
        traces = {(s, z): trace_for(pool, s, z) for s in (1500, *EVAL_SEEDS) for z in EVAL_SIZES}
        trace_file = out/"trace.json"
        serialized = [[s, z, tr] for (s, z), tr in traces.items()]
        if args.resume:
            assert json.loads(json.dumps(serialized)) == read_json(trace_file)
        else:
            write_json(trace_file, serialized)
        roots = {tuple(p[:256]) for tr in traces.values() for p in tr[-1][1]}
        for old in (cal, OLD_PROBE, Path("results/20261118_queue_c_calibration"), Path("results/20261118_queue_c_evaluation")):
            assert roots.isdisjoint(prefix_roots(old/"requests.jsonl"))
        orders = list(permutations(ARMS))
        design = dict(config=config, seeds=list(EVAL_SEEDS), cold_lengths=list(EVAL_SIZES), warmup_seed=1500,
                      pool=pool, arms=list(ARMS), orders=[list(o) for o in orders], output_tokens=128,
                      alternate_output_tokens=2, trace_sha256=sha256(trace_file), v2_sha256=BASE_SHA,
                      table_sha256=status["table_sha256"], disjoint_roots=len(roots), profiler=False,
                      order="six permutations per cold length; rotate length order per seed")
        if args.resume:
            assert read_json(out/"design.json") == design
        else:
            write_json(out/"design.json", design)
        torch.manual_seed(0)
        llm = LLM(args.model, **config)
        env = cost_environment(llm.config, llm.scheduler.block_manager.namespace)
        policies = dict(B=None, C2=RestoreCostTable.load(out/"v2.json", env), C3=llm.scheduler.restore_policy)
        assert llm.model_runner.kv_cache.dtype == torch.bfloat16 and 4 in llm.model_runner.graphs
        write_json(out/"engine.json", dict(environment=env, actual_budget=llm.model_runner.kv_cache_budget,
                                           graph_bs=llm.model_runner.graph_bs, dtype=str(llm.model_runner.kv_cache.dtype)))
        warm = Path(mkdtemp(dir=out, prefix="warmup-"))
        for size in EVAL_SIZES:
            for arm, pol in (*policies.items(), ("force_recompute", RecomputeOne(LENGTHS, TARGET))):
                tag = dict(seed=1500, cold_peer_tokens=size, arm=arm, warmup=True)
                cycle(llm, warm/str(size)/arm, traces[1500,size], pol, 2, tag)
        checks = []
        for index, seed in enumerate(EVAL_SEEDS):
            offset = index % len(EVAL_SIZES)
            for size in EVAL_SIZES[offset:] + EVAL_SIZES[:offset]:
                name = f"{seed}-{size}"
                folder = out/"measurements"/name
                trace = traces[seed,size]
                states, batches = set(), {}
                order = orders[(index + EVAL_SIZES.index(size)) % len(orders)]
                for pos, arm in enumerate(order):
                    tag = dict(seed=seed, cold_peer_tokens=size, arm=arm, order_position=pos, warmup=False)
                    batches[arm], state = cycle(llm, folder/arm, trace, policies[arm], 128, tag)
                    states.add(state)
                assert len(states) == 1
                b_replay = batches["B"][-1][0]
                for arm in ("C2", "C3"):
                    for i in (0, 1):
                        assert same_output(batches["B"][i][0], batches[arm][i][0])
                    c_rows = batches[arm][-1][0]
                    if b_replay[TARGET]["cpu_restored_tokens"] == c_rows[TARGET]["cpu_restored_tokens"]:
                        assert same_output(b_replay, c_rows)
                    else:
                        checks.append((name, arm, seed, trace, c_rows, state))
                print(f"seed={seed} cold={size} order={','.join(order)} complete", flush=True)
        # Measure only counterfactual routes absent from the three formal arms.
        # The target reaches its first token before any decode, also in the two-output controls.
        alternate_count = 0
        for seed in EVAL_SEEDS:
            for size in EVAL_SIZES:
                name = f"{seed}-{size}"
                folder = out/"measurements"/name
                formal = {a: load_cycle(folder/a) for a in ARMS}
                actions = {observe(b[-1], size)["action"] for b, _ in formal.values()}
                if "recompute" not in actions:
                    tag = dict(seed=seed, cold_peer_tokens=size, arm="force_recompute", warmup=False)
                    batch, state = cycle(llm, out/"controls"/name, traces[seed,size], RecomputeOne(LENGTHS, TARGET), 2, tag)
                    assert state == formal["B"][1] and observe(batch[-1], size)["action"] == "recompute"
                    alternate_count += 1
        llm.scheduler.restore_policy = None
        for name, arm, seed, trace, rows, state in checks:
            folder = out/"reference"/name/arm
            if not folder.is_dir():
                folder.parent.mkdir(parents=True, exist_ok=True)
                attempt = Path(mkdtemp(dir=folder.parent, prefix=".pending-"))
                oracle(llm, attempt, seed, trace, rows, state, workload=name, setup_output_tokens=2)
                attempt.rename(folder)
            exact = lines(folder/"checks.jsonl")
            assert len(exact) == 4 and all(c["exact"] and c["output_tokens"] == 128 for c in exact)
        cases = len(EVAL_SEEDS)*len(EVAL_SIZES)
        write_json(out/"status.json", dict(passed=True, matched_cases=cases, per_arm_requests=12*cases,
                   replay_per_arm=4*cases, greedy_C2_C3_verified=24*cases,
                   extra_independent_reference_replays=4*len(checks), two_output_counterfactuals=alternate_count))
    except BaseException:
        (out/"failure.txt").write_text(traceback.format_exc())
        write_json(out/"status.json", dict(passed=False))
        raise
    finally:
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
