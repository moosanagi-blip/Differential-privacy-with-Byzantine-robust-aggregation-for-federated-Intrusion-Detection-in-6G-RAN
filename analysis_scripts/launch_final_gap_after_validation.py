"""Gate the authorized campaign on both instrumentation and integration checks."""
from pathlib import Path
import json, subprocess, sys, time
base=Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,str(base/'work/test_final_gap_instrumentation.py')],check=True)
smoke=base/'paper gpt/dp-fl-ids code/results/final_gap_smoke_20261003'
deadline=time.monotonic()+3600
while not (smoke/'PASSED.json').exists():
    record=smoke/'noise_selection_dp5_s42/record.json'
    if record.exists() and json.loads(record.read_text())['status']!='completed':
        raise RuntimeError('Full-size integration failed; campaign not launched')
    if time.monotonic()>deadline:raise RuntimeError('Integration timeout; campaign not launched')
    time.sleep(10)
print('Both validation gates passed; starting nine full runs',flush=True)
subprocess.run([sys.executable,str(base/'work/run_final_gap_experiments.py')],check=True)
print('Campaign terminal records ready for audit; not yet manuscript evidence',flush=True)
