"""Run verified replicates 43 then 44; stop on any orchestration error."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import subprocess
import sys

base = Path(__file__).resolve().parents[1]
project = base / 'paper gpt/dp-fl-ids code'
out = project / 'results/reviewer_corrected_2026-09-26'
for seed in (43, 44):
    validation = json.loads((project / f'data_original53_v3_f64_p{seed}/reviewer_validation.json').read_text())
    if validation.get('status') != 'passed' or validation.get('partition_seed') != seed:
        raise RuntimeError(f'Partition {seed} has not passed validation')
    if (out / f'LAUNCH_s{seed}.json').exists():
        raise RuntimeError(f'Partition {seed} already launched; inspect before resuming')
with (out / 'REMAINING_QUEUE_20260928.json').open('x') as handle:
    json.dump({'pid':os.getpid(), 'started_utc':datetime.now(timezone.utc).isoformat(),
               'seeds':[43,44], 'policy':'sequential; stop on nonzero exit'}, handle, indent=2)
for seed in (43, 44):
    with (out / f'seed{seed}.stdout.log').open('x') as stdout, (out / f'seed{seed}.stderr.log').open('x') as stderr:
        subprocess.run([sys.executable, '-u', str(base/'work/run_reviewer_corrected.py'),
                        '--seed', str(seed)], cwd=base, stdout=stdout, stderr=stderr, check=True)
with (out / 'REMAINING_QUEUE_FINISHED.json').open('x') as handle:
    json.dump({'status':'finished', 'finished_utc':datetime.now(timezone.utc).isoformat()}, handle)
