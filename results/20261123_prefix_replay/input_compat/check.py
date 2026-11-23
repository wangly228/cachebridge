"""Run from the public repository root with PYTHONPATH=. and a new output directory argument."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=False)
sample = Path('results/20261123_prefix_replay/protection/held_out_waves.jsonl.gz')
transcript = out / 'trace.jsonl'
transcript.write_bytes(gzip.decompress(sample.read_bytes()))
source = Path('results/20261123_revisit_boundary/engine.json')
engine = json.loads(source.read_text())
engine['config'].pop('cpu_prefix_transfer_mode')
base = [sys.executable, '-m', 'benchmarks.prefix_replay', '--transcript', str(transcript)]
for case in ('implicit_sync','explicit_sync'):
    cfg = out / case
    cfg.mkdir()
    if case == 'explicit_sync':
        engine['config']['cpu_prefix_transfer_mode'] = 'sync'
    (cfg/'engine.json').write_text(json.dumps(engine))
    subprocess.run([*base,'--arm','CPU','--input',str(cfg),'--output',str(cfg/'replay')],check=True)
implicit = out/'implicit_sync/replay'
explicit = out/'explicit_sync/replay'
hashes = {}
for name in ('requests.jsonl','events.jsonl'):
    hashes[name] = digest(implicit/name)
    assert hashes[name] == digest(explicit/name)
summary = json.loads((implicit/'summary.json').read_text())
assert summary['replayed_requests'] == 20
assert summary['replayed_waves'] == 5
assert (summary['transfer_mode'],summary['pack_cold'],summary['cpu_slots']) == ('sync',False,64)
rejected = subprocess.run([*base, '--arm','unknown_arm','--input',str(out/'implicit_sync'),
                           '--output',str(out/'invalid_output')],capture_output=True,text=True)
assert rejected.returncode != 0 and 'Unknown CPU trace arm' in rejected.stderr
assert not (out/'invalid_output').exists()
(out/'evidence.json').write_text(json.dumps(dict(trace_sha256=digest(transcript),source_config_sha256=digest(source),
    source_script_sha256=summary['code_sha256']['benchmarks/prefix_replay.py'],
    script_sha256=digest(Path(__file__)),implicit_sync_and_explicit_sync_identical=hashes,
    n=summary['replayed_requests'],waves=summary['replayed_waves'],invalid_arm_rejected_before_writing=True),indent=2)+'\n')
