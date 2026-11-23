"""Compare archived PegaFlow outputs with an isolated checkout of its base commit."""
import argparse
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from unittest.mock import patch


BASE = "bb823b3e06983d71485a8e1f23715ebd87d98ef8"


def worker(tree, source, out, task):
    sys.path.insert(0, str(tree))
    import numpy as np
    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.block_manager import BlockManager
    from nanovllm.engine.model_runner import ModelRunner

    engine = json.loads((source / "engine.json").read_text())
    config = engine["config"].copy()
    limit = config.pop("max_kvcache_blocks")
    config.pop("enable_metrics")
    model = json.loads((source / "environment.json").read_text())["model_path"]
    config.pop("model", None)
    # Upstream overwrites this output field inside allocate_kv_cache. Interpose
    # only that numeric assignment, keeping its safe budget and real allocation.
    allocate = ModelRunner.allocate_kv_cache
    def capped(runner):
        original = runner.config
        class Budget:
            def __getattr__(self, name):
                return getattr(original, name)
            def __setattr__(self, name, value):
                assert name == "num_kvcache_blocks"
                original.num_kvcache_blocks = min(limit, value)
        with patch.object(runner, "config", Budget()):
            allocate(runner)
    torch.manual_seed(0)
    with patch.object(ModelRunner, "allocate_kv_cache", capped):
        llm = LLM(model, **config)
    runner = llm.model_runner
    assert runner.kv_cache.size(2) == limit and runner.kv_cache.dtype == torch.bfloat16
    for name, module in sys.modules.copy().items():
        if (name == "nanovllm" or name.startswith("nanovllm.")) and getattr(module, "__file__", None):
            assert Path(module.__file__).resolve().is_relative_to(tree), name
    def reset():
        assert llm.is_finished() and not llm.scheduler.block_manager.used_block_ids
        llm.scheduler.block_manager = BlockManager(limit, 256)
    def play(inputs, n, oracle=None, path=None):
        ids = {}
        params = SamplingParams(temperature=1, max_tokens=n, ignore_eos=True)
        for i, prompt in enumerate(inputs):
            llm.add_request(prompt, params)
            ids[llm.scheduler.waiting[-1].seq_id] = i
        records = []
        original_run, original_model = runner.run, runner.run_model
        active = []
        def model_logits(*args):
            logits = original_model(*args)
            if oracle is not None:
                for row, seq in enumerate(active):
                    if seq.num_cached_tokens + seq.num_scheduled_tokens != len(seq):
                        continue
                    i, pos = ids[seq.seq_id], seq.num_completion_tokens
                    actual = logits[row].detach().view(torch.int16).cpu().numpy()
                    records.append(dict(request=i, position=pos, sha256=hashlib.sha256(actual).hexdigest(),
                                        exact=np.array_equal(actual, oracle[i, pos])))
            return logits
        def run(seqs, prefill):
            active[:] = seqs
            outputs = original_run(seqs, prefill)
            if path is not None:
                for j, seq in enumerate(seqs):
                    assert seq.completion_token_ids == path[ids[seq.seq_id]][:seq.num_completion_tokens]
                    outputs[j] = path[ids[seq.seq_id]][seq.num_completion_tokens]
            return outputs
        finished = {}
        with ExitStack() as stack:
            # Base API forbids zero temperature. Compare pre-sampling logits,
            # or select plain argmax explicitly for free-running pressure traces.
            stack.enter_context(patch.object(runner.sampler, "forward", lambda logits, _: logits.argmax(-1)))
            stack.enter_context(patch.object(runner, "run_model", model_logits))
            stack.enter_context(patch.object(runner, "run", run))
            while not llm.is_finished():
                done, _ = llm.step()
                finished.update(done)
        return [finished[i] for i in ids], records

    with (out / f"{task}.jsonl").open("x") as log:
        if task == "fixed_path":
            for g in json.loads((source / "inputs.json").read_text()):
                reset()
                name = g["name"]
                path = json.loads((source / f"{name}_path.json").read_text())
                with np.load(source / f"{name}_logits_bf16.npz") as data:
                    for mode in ("cold", "warm"):
                        _, records = play(g["inputs"], 16, data[mode], path)
                        for r in records:
                            log.write(json.dumps(dict(group=name, cache=mode, **r)) + "\n")
                print(name, flush=True)
        else:
            rows = [json.loads(s) for s in (source / "requests.jsonl").read_text().splitlines()]
            for seed, cycle, waves in json.loads((source / "trace.json").read_text()):
                reset()
                torch.manual_seed(seed)
                for phase, inputs in waves:
                    refs = [r for r in rows if r.get("seed") == seed and r.get("cycle") == cycle and r["phase"] == phase]
                    assert inputs == [r["input_token_ids"] for r in refs]
                    outputs, _ = play(inputs, 128)
                    for i, (actual, ref) in enumerate(zip(outputs, refs, strict=True)):
                        log.write(json.dumps(dict(seed=seed, cycle=cycle, phase=phase, request=i,
                            exact=actual == ref["output_token_ids"], output_ids=actual)) + "\n")
                log.flush()
                print(seed, cycle, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fixed-path", default="results/20261118_fixed_path_probe")
    parser.add_argument("--pressure", default="results/20261118_pressure36")
    args = parser.parse_args()
    from benchmarks.common import prepare_run, sha256, write_json
    args.module = "benchmarks.upstream_acceptance"
    sources = dict(fixed_path=Path(args.fixed_path).resolve(), pressure=Path(args.pressure).resolve())
    model = json.loads((sources["pressure"] / "environment.json").read_text())["model_path"]
    out = prepare_run(args.output, args, model).resolve()
    ref = subprocess.check_output(["git", "archive", "--format=tar", BASE, "nanovllm"])
    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp).resolve()
        with tarfile.open(fileobj=io.BytesIO(ref)) as tar:
            tar.extractall(tree, filter="data")
        write_json(out / "reference.json", dict(base=BASE,
                   upstream_sha256={str(p.relative_to(tree)): sha256(p) for p in tree.rglob("*.py")},
                   source_runs={k: str(v) for k, v in sources.items()},
                   stage1_reference="Compare BF16 bits before sampling at fixed history and complete greedy paths on the same trace",
                   adapters=["same safe KV cap", "explicit argmax or teacher-forced IDs after logits"]))
        summary = {}
        for task, source in sources.items():
            cmd = [sys.executable, "-I", str(Path(__file__).resolve()), "worker", str(tree), str(source), str(out), task]
            with (out / f"{task}.console.txt").open("x") as log:
                subprocess.run(cmd, cwd=tree, stdout=log, stderr=subprocess.STDOUT, check=True)
            rows = [json.loads(s) for s in (out / f"{task}.jsonl").read_text().splitlines()]
            expected = 288 if task == "fixed_path" else 96
            summary[task] = dict(compared=len(rows), exact=sum(r["exact"] for r in rows),
                                 passed=len(rows) == expected and all(r["exact"] for r in rows))
            write_json(out / "summary.json", summary)
        if not all(s["passed"] for s in summary.values()):
            raise RuntimeError("upstream parity failed")


if __name__ == "__main__":
    if sys.argv[1:2] == ["worker"]:
        worker(*map(Path, sys.argv[2:5]), task=sys.argv[5])
    else:
        main()
