"""Diagnose first-token cold/warm logits at identical input positions."""
import argparse
import json
from pathlib import Path
import traceback

import numpy as np
import torch

from nanovllm import LLM
from benchmarks.common import prepare_run, write_json
from benchmarks.phase1 import reset_cache, wave, save_wave


def compare(reference, actual):
    delta = (actual - reference).abs()
    top = torch.topk(reference, 5)
    ref_id, actual_id = reference.argmax().item(), actual.argmax().item()
    return dict(max_abs=delta.max().item(), mean_abs=delta.mean().item(),
                relative_l2=(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference)).item(),
                reference_argmax=ref_id, actual_argmax=actual_id,
                reference_top5_ids=top.indices.tolist(), reference_top5_logits=top.values.tolist(),
                actual_at_reference_top5=actual[top.indices].tolist(),
                reference_top2_margin=(top.values[0] - top.values[1]).item(),
                reference_at_actual_argmax=reference[actual_id].item(),
                actual_top2_margin=torch.diff(torch.topk(actual, 2).values).abs().item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    args.module = "benchmarks.cache_probe"
    run = Path(args.input_run)
    config = json.loads((run / "engine.json").read_text())["config"]
    inputs = [r["input_token_ids"] for r in map(json.loads, (run / "requests.jsonl").read_text().splitlines())
              if r["phase"] == "baseline" and r["length"] == 2048 and r["concurrency"] == 4 and r["cache"] == "cold"][:4]
    assert len(inputs) == 4
    output = prepare_run(args.output, args, config["model"])
    llm = None
    try:
        config.pop("num_kvcache_blocks")
        llm = LLM(**config)
        runner = llm.model_runner
        original_run, original_model = runner.run, runner.run_model
        active, captured = [], {}

        def run(seqs, is_prefill):
            active[:] = seqs
            return original_run(seqs, is_prefill)

        def model(input_ids, positions, is_prefill):
            logits = original_model(input_ids, positions, is_prefill)
            for i, seq in enumerate(active):
                if seq.num_completion_tokens == 0 and seq.num_cached_tokens + seq.num_scheduled_tokens == len(seq):
                    captured[seq.seq_id] = logits[i].detach().float().cpu().clone()
            return logits

        runner.run, runner.run_model = run, model
        arrays = {}
        for case in ("cold_batch", "warm_batch", "warm_single"):
            if case == "cold_batch":
                reset_cache(llm)
            batches = [[p] for p in inputs] if case == "warm_single" else [inputs]
            logits = []
            for batch in batches:
                result = wave(llm, batch, 1, dict(phase="logit_probe", case=case))
                save_wave(output, result)
                logits.extend(captured[r["seq_id"]] for r in result[0])
            arrays[case] = torch.stack(logits)
        report = {case: [compare(a, b) for a, b in zip(arrays["cold_batch"], values)]
                  for case, values in arrays.items() if case != "cold_batch"}
        np.savez_compressed(output / "first_logits.npz", **{k: v.numpy() for k, v in arrays.items()})
        write_json(output / "comparison.json", report)
        write_json(output / "status.json", dict(success=True, interpretation="diagnostic only; no tolerance chosen post hoc"))
        print(json.dumps(report, indent=2), flush=True)
    except BaseException:
        (output / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
