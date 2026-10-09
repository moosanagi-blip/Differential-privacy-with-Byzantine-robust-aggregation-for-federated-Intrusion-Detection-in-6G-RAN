"""Frozen reviewer matrix; execute one verified partition replicate at a time."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import shutil
import sys

BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE / 'paper gpt/dp-fl-ids code'
sys.path.insert(0, str(PROJECT / 'code'))
from src.experiment_config import RunConfig

OUT = PROJECT / 'results/reviewer_corrected_2026-09-26'

def matrix(seed):
    common = dict(model='mlp', hidden=64, optimizer='sgd', lr=.1, momentum=.9,
        rounds=30, n_chunks=30, round_chunks=True, num_clients=10,
        local_epochs=1, batch_size=64, delta=1e-5, max_grad_norm=1.,
        seed=seed, device='cpu', engine='sequential',
        data_root=str(PROJECT / f'data_original53_v3_f64_p{seed}'))
    result = []
    def add(name, private, **kwargs):
        result.append(RunConfig(**common, name=f'{name}_s{seed}', dp=private,
            target_epsilon=kwargs.pop('epsilon', 5.) if private else None, **kwargs))
    for private in (False, True):
        tag = 'dp5' if private else 'nonprivate'
        for agg in ('multikrum', 'uniformavg', 'fedavg'):
            for attack in ('clean', 'static', 'rotating'):
                add(f'primary_{tag}_{agg}_{attack}', private, aggregator=agg,
                    attack='none' if attack == 'clean' else 'adaptive_krum',
                    byzantine_fraction=0. if attack == 'clean' else .3,
                    n_byzantine_estimate=3, attack_scale=2.,
                    rotate_byzantine=attack == 'rotating')
        add(f'radio_masked_{tag}', private, aggregator='multikrum',
            n_byzantine_estimate=3, mask_ran_features=True)
        for estimate in (0, 1, 2, 4):
            add(f'assumed_f{estimate}_{tag}', private, aggregator='multikrum',
                attack='adaptive_krum', attack_scale=2., byzantine_fraction=.3,
                n_byzantine_estimate=estimate)
        for cap in (.5, 5.):
            add(f'cap{cap:g}_{tag}', private, aggregator='multikrum',
                attack='adaptive_krum', attack_scale=cap, byzantine_fraction=.3,
                n_byzantine_estimate=3)
    for epsilon in (1., 10.):
        add(f'epsilon{epsilon:g}_static', True, epsilon=epsilon,
            aggregator='multikrum', attack='adaptive_krum', attack_scale=2.,
            byzantine_fraction=.3, n_byzantine_estimate=3)
    return result

def save_new(path, data):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, choices=(42,43,44), default=42)
    ap.add_argument('--plan-only', action='store_true')
    ap.add_argument('--resume', action='store_true', help='Only after verifying no previous worker is running')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    plan = [cfg.to_dict() for seed in (42,43,44) for cfg in matrix(seed)]
    assert len(plan) == 102 and len({p['name'] for p in plan}) == 102
    plan_path = OUT / 'matrix.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise RuntimeError('Frozen matrix differs; refusing silent changes')
    else:
        save_new(plan_path, plan)
    if args.plan_only:
        print(f'Validated frozen matrix: {len(plan)} conditions; 34 per replicate')
        return
    import torch
    from src.data.loaders import DataRoot
    from src.fl.simulation import run_federated
    from src.utils.run_record import execute
    root = Path(matrix(args.seed)[0].data_root)
    if DataRoot(root).feature_dtype != 'float64':
        raise RuntimeError('Float64 dataset required')
    if args.seed != 42:
        validation = json.loads((root / 'reviewer_validation.json').read_text())
        if validation.get('status') != 'passed':
            raise RuntimeError('Independent partition validation required')
    if shutil.disk_usage(BASE).free < 10 * 2**30:
        raise RuntimeError('Less than 10 GiB free; not launching')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # Exclusive marker prevents concurrent launches and unsafe crash resumes.
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    launch_name = f'RESUME_s{args.seed}_{stamp}.json' if args.resume else f'LAUNCH_s{args.seed}.json'
    save_new(OUT / launch_name, {
        'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(),
        'partition_seed': args.seed, 'training_seed': args.seed,
        'augmentation_seed': 42, 'threads': 1,
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    statuses = []
    for cfg in matrix(args.seed):
        dest = OUT / cfg.name
        if args.resume and dest.exists():
            if (dest / 'record.json').exists():
                saved = json.loads((dest / 'record.json').read_text())
                if saved['config'] != cfg.to_dict():
                    raise RuntimeError('Existing record configuration mismatch')
                if saved['status'] not in ('completed', 'numerical_failure'):
                    raise RuntimeError('Existing execution error requires inspection')
                statuses.append({'name':cfg.name, 'status':saved['status']})
                continue
            if (dest / 'reuse.json').exists():
                saved = json.loads((dest / 'reuse.json').read_text())
                if saved['config'] != cfg.to_dict() or hashlib.sha256(Path(saved['source']).read_bytes()).hexdigest() != saved['sha256']:
                    raise RuntimeError('Reused record mismatch')
                statuses.append({'name':cfg.name, 'status':'reused_completed'})
                continue
            archive = OUT / 'interrupted_attempts'
            archive.mkdir(exist_ok=True)
            dest.rename(archive / f'{cfg.name}_{stamp}')
        dest.mkdir(exist_ok=False)
        # Reuse only the exact completed configuration, ignoring its display name.
        if args.seed == 42 and cfg.name == 'primary_dp5_multikrum_static_s42':
            source = PROJECT / 'results/main_original53_float64_2026-09-26/record.json'
            record = json.loads(source.read_text())
            old, new = dict(record['config']), cfg.to_dict()
            old.pop('name'); new.pop('name')
            if old != new or record['status'] != 'completed':
                raise RuntimeError('Prior main run does not match frozen condition')
            save_new(dest / 'reuse.json', {'source': str(source),
                'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'config': cfg.to_dict()})
            statuses.append({'name':cfg.name,'status':'reused_completed'})
            continue
        save_new(dest / 'config.json', cfg.to_dict())
        with (dest / 'progress.jsonl').open('x', encoding='utf-8') as log:
            def runner(config, verbose=False, on_event=None):
                def observe(event, payload):
                    if on_event: on_event(event, payload)
                    if event != 'provenance':
                        log.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),
                            'event':event, **payload}, allow_nan=False) + '\n')
                        log.flush()
                    if event in ('run_start','round_end','run_end'):
                        print(json.dumps({'name':cfg.name,'event':event,**payload}), flush=True)
                return run_federated(config, verbose=False, on_event=observe)
            record = execute(cfg, runner)
        save_new(dest / 'record.json', record)
        statuses.append({'name':cfg.name,'status':record['status']})
        if record['status'] in ('execution_error', 'interrupted'):
            raise RuntimeError(f"Campaign stopped: {cfg.name}: {record.get('error')}")
        # Numerical failure is an experimental outcome, never a valid final score.
    save_new(OUT / f'FINISHED_s{args.seed}.json', statuses)

if __name__ == '__main__':
    main()
