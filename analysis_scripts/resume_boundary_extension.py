"""Resume the authorized matrix, retaining interrupted attempts verbatim."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
import torch

base = Path(__file__).resolve().parents[1]
project = base / 'paper gpt/dp-fl-ids code'
sys.path.insert(0, str(project / 'code'))
from src.experiment_config import RunConfig
from src.fl.simulation import run_federated
from src.utils.run_record import execute

out = project / 'results/boundary_extension_20261001'
started = json.loads((out / 'STARTED.json').read_text())
configs = json.loads((out / 'matrix.json').read_text())
assert len(configs) == 36
assert not (out / 'FINISHED.json').exists()
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def check_sources():
    for path, digest in started['source_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path

def save(path, data):
    with path.open('x') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)

check_sources()
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
save(out / f'RESUMED_{stamp}.json', {
    'pid': os.getpid(), 'utc': datetime.now(timezone.utc).isoformat(),
    'policy': 'Skip terminal records; archive incomplete attempts; restart those from round 1 with identical configuration.',
    'resume_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
})
statuses = []
for data in configs:
    check_sources()
    cfg = RunConfig.from_dict(data)
    dest = out / cfg.name
    record_path = dest / 'record.json'
    if record_path.exists():
        record = json.loads(record_path.read_text())
        assert record['status'] in ('completed', 'numerical_failure')
        statuses.append({'name': cfg.name, 'status': record['status']})
        continue
    if dest.exists():
        assert json.loads((dest / 'config.json').read_text()) == data
        archive = out / 'interrupted_attempts'
        archive.mkdir(exist_ok=True)
        target = archive / f'{cfg.name}_{stamp}'
        assert not target.exists()
        dest.rename(target)
    dest.mkdir()
    save(dest / 'config.json', data)
    with (dest / 'progress.jsonl').open('x') as log:
        def runner(config, verbose=False, on_event=None):
            def observe(event, payload):
                if on_event:
                    on_event(event, payload)
                if event != 'provenance':
                    log.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'event': event, **payload}, allow_nan=False) + '\n')
                    log.flush()
                if event in ('run_start', 'round_end', 'run_end'):
                    print(json.dumps({'name': cfg.name, 'event': event, 'round': payload.get('round')}, allow_nan=False), flush=True)
            return run_federated(config, verbose=False, on_event=observe)
        record = execute(cfg, runner)
    save(record_path, record)
    statuses.append({'name': cfg.name, 'status': record['status']})
    if record['status'] not in ('completed', 'numerical_failure'):
        raise RuntimeError(f"Execution stopped: {record.get('error')}")
save(out / 'FINISHED.json', statuses)
