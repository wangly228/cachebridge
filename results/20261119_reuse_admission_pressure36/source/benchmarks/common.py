"""Local-only experiment provenance; no downloads or dependency installation."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import shutil
import struct
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path, rows):
    with Path(path).open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def model_manifest(model):
    files = []
    for path in sorted(Path(model).iterdir()):
        if path.suffix not in (".json", ".safetensors", ".txt"):
            continue
        record = dict(name=path.name, bytes=path.stat().st_size, sha256=sha256(path))
        metadata = Path(model) / ".cache/huggingface/download" / (path.name + ".metadata")
        if metadata.exists():
            record["download_metadata"] = metadata.read_text().splitlines()
        if path.suffix == ".safetensors":
            with path.open("rb") as f:
                header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
            record["tensor_dtypes"] = sorted({v["dtype"] for k, v in header.items() if k != "__metadata__"})
        files.append(record)
    return files


def prepare_run(output, args, model):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    print(f"Artifacts: {output.resolve()}", flush=True)
    (output / "command.txt").write_text(shlex.join([sys.executable, "-m", args.module, *sys.argv[1:]]) + "\n")
    (output / "git.diff").write_text(command("git", "diff", "HEAD"))
    source_hashes = {}
    paths = [ROOT / "pyproject.toml", ROOT / "AGENTS.md"]
    for directory in ("nanovllm", "benchmarks", "tests", "docs"):
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.suffix in (".py", ".md"))
    for path in sorted(set(paths)):
        relative = path.relative_to(ROOT)
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        source_hashes[str(relative)] = sha256(path)
    environment = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        argv=sys.argv, args=vars(args), cwd=str(ROOT),
        git_head=command("git", "rev-parse", "HEAD"),
        git_branch=command("git", "branch", "--show-current"),
        git_status=command("git", "status", "--short"),
        source_sha256=source_hashes,
        python=sys.version, executable=sys.executable, platform=platform.platform(),
        packages={p: importlib.metadata.version(p) for p in (
            "nano-vllm", "torch", "triton", "flash-attn", "transformers", "numpy", "xxhash", "einops", "safetensors")},
        cuda_runtime=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
        bf16_supported=torch.cuda.is_bf16_supported(),
        nvidia_smi=command("nvidia-smi"),
        meminfo=Path("/proc/meminfo").read_text(),
        model_path=str(Path(model).resolve()), model_files=model_manifest(model),
    )
    write_json(output / "environment.json", environment)
    return output


def memory():
    free, total = torch.cuda.mem_get_info()
    status = Path("/proc/self/status").read_text().splitlines()
    return dict(gpu_allocated=torch.cuda.memory_allocated(), gpu_reserved=torch.cuda.memory_reserved(),
                gpu_peak_allocated=torch.cuda.max_memory_allocated(),
                gpu_peak_reserved=torch.cuda.max_memory_reserved(), gpu_device_used=total - free,
                cpu_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                cpu_rss=int(next(line.split()[1] for line in status if line.startswith("VmRSS:"))) * 1024)


def distribution(values):
    return dict(n=len(values), p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95)), min=float(min(values)), max=float(max(values)))


def p95_interval(values):
    # Descriptive IID bootstrap only; requests in a wave are correlated.
    rng = np.random.default_rng(20261117)
    samples = rng.choice(values, size=(2000, len(values)), replace=True)
    return [float(x) for x in np.percentile(np.percentile(samples, 95, axis=1), [2.5, 97.5])]
