"""Offline logit-shift/probability diagnostics; never changes protocol verdicts."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.common import write_json, write_jsonl


def decode_bf16(array):
    # fixed_path_probe archives raw BF16 bits using NumPy's supported int16 dtype.
    return torch.from_numpy(array.copy()).view(torch.bfloat16).float()


def analyze(reference, actual):
    delta = actual - reference
    bias = delta.mean(dim=-1, keepdim=True)
    centered_error = delta - bias
    centered_reference = reference - reference.mean(dim=-1, keepdim=True)
    log_p = reference.log_softmax(dim=-1)
    log_q = actual.log_softmax(dim=-1)
    p, q = log_p.exp(), log_q.exp()
    return dict(
        mean_logit_shift=bias.squeeze(-1),
        constant_shift_error_energy_fraction=(bias.square().squeeze(-1) * reference.shape[-1] /
                                              delta.square().sum(-1).clamp_min(1e-30)),
        centered_max_abs=centered_error.abs().amax(-1),
        centered_relative_l2=(torch.linalg.vector_norm(centered_error, dim=-1) /
                              torch.linalg.vector_norm(centered_reference, dim=-1).clamp_min(1e-12)),
        probability_total_variation=(p - q).abs().sum(-1) / 2,
        kl_reference_to_actual=(p * (log_p - log_q)).sum(-1),
        max_abs_probability_change=(p - q).abs().amax(-1),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.input_run), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(x) for x in (source / "comparisons.jsonl").read_text().splitlines()]
    summaries = []
    for group in json.loads((source / "summary.json").read_text()):
        name = group["group"]
        with np.load(source / f"{name}_logits_bf16.npz") as arrays:
            diagnostics = analyze(decode_bf16(arrays["cold"]), decode_bf16(arrays["warm"]))
        records = []
        for row in rows:
            if row["group"] != name:
                continue
            i, position = row["request"], row["position"]
            records.append(dict(row, **{k: v[i, position].item() for k, v in diagnostics.items()}))
        write_jsonl(output / "positions.jsonl", records)
        summaries.append(dict(group=name, positions=len(records),
            original_numerical_failures=sum(not r["passed"] for r in records),
            max_centered_relative_l2=max(r["centered_relative_l2"] for r in records),
            max_probability_total_variation=max(r["probability_total_variation"] for r in records),
            max_kl_reference_to_actual=max(r["kl_reference_to_actual"] for r in records),
            max_abs_probability_change=max(r["max_abs_probability_change"] for r in records)))
    write_json(output / "summary.json", summaries)
    write_json(output / "status.json", dict(analysis_completed=True, input_run=str(source),
        original_status=json.loads((source / "status.json").read_text()),
        interpretation="Post-hoc diagnostics only; no new acceptance thresholds or verdict changes."))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
