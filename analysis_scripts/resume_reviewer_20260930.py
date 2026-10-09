"""Resume inspected interrupted seed 43, then launch unstarted seed 44."""
from pathlib import Path
import subprocess
import sys
base = Path(__file__).resolve().parents[1]
out = base/'paper gpt/dp-fl-ids code/results/reviewer_corrected_2026-09-26'
if (out/'FINISHED_s43.json').exists() or (out/'LAUNCH_s44.json').exists():
    raise RuntimeError('Queue state changed; inspect before launching')
for seed in (43,44):
    with (out/f'restart_20260930_s{seed}.stdout.log').open('x') as stdout, (out/f'restart_20260930_s{seed}.stderr.log').open('x') as stderr:
        command = [sys.executable, '-u', str(base/'work/run_reviewer_corrected.py'), '--seed', str(seed)]
        if seed == 43: command.append('--resume')
        subprocess.run(command, cwd=base, stdout=stdout, stderr=stderr, check=True)
