"""Authorized recovery of only the two unfinished final-gap conditions."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os
import run_final_gap_experiments as g

out=(g.project/'results/final_gap_20261003').resolve()
started=json.loads((out/'STARTED.json').read_text())
configs=json.loads((out/'matrix.json').read_text())
expected={'displacement_dp10_s44','noise_selection_dp5_s44'}

def check_sources():
    for p,h in started['source_sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p

check_sources()
assert not (out/'FINISHED.json').exists()
pending={c['name'] for c in configs if not (out/c['name']/'record.json').exists()}
assert pending==expected,pending
completed_hashes={str(out/c['name']/'record.json'):hashlib.sha256((out/c['name']/'record.json').read_bytes()).hexdigest() for c in configs if c['name'] not in pending}
g.torch.set_num_threads(1)
g.torch.set_num_interop_threads(1)
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
g.save(out/f'RESUMED_{stamp}.json',{'pid':os.getpid(),'utc':datetime.now(timezone.utc).isoformat(),
    'pending':sorted(pending),'completed_record_sha256':completed_hashes,
    'resume_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'policy':'Preserve incomplete attempt; restart from round 1; skip seven terminal records.'})
statuses=[]
for data in configs:
    check_sources()
    cfg=g.RunConfig.from_dict(data)
    dest=(out/cfg.name).resolve()
    assert dest.parent==out
    if cfg.name not in pending:
        record=json.loads((dest/'record.json').read_text())
        assert record['status']=='completed'
        statuses.append({'name':cfg.name,'status':record['status']})
        continue
    if dest.exists():
        archive=(out/'interrupted_attempts').resolve()
        assert archive.parent==out
        archive.mkdir(exist_ok=True)
        target=(archive/f'{cfg.name}_{stamp}').resolve()
        assert target.parent==archive and not target.exists()
        before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.is_file()}
        dest.rename(target)
        assert all(hashlib.sha256((target/n).read_bytes()).hexdigest()==h for n,h in before.items())
        print('Preserved interrupted attempt:',target,flush=True)
    dest.mkdir()
    g.save(dest/'config.json',data)
    g.counterfactuals.clear();g.diagnostics.clear()
    paired=cfg.attack=='none'
    g.simulation.local_train=g.paired_local if paired else g.original_local
    g.simulation.aggregation.aggregate=g.paired_aggregate if paired else g.original_aggregate
    with (dest/'progress.jsonl').open('x') as log:
        def runner(config,verbose=False,on_event=None):
            def observe(event,payload):
                if on_event:on_event(event,payload)
                if event!='provenance':
                    log.write(json.dumps({'event':event,**payload},allow_nan=False)+'\n');log.flush()
                if event=='round_end':
                    if paired:g.save(dest/f'noise_round_{payload["round"]:02d}.json',g.diagnostics[-1])
                    print(cfg.name,event,payload['round'],flush=True)
            return g.simulation.run_federated(config,verbose=False,on_event=observe)
        record=g.execute(cfg,runner)
    g.save(dest/'record.json',record)
    statuses.append({'name':cfg.name,'status':record['status']})
    if record['status'] not in ('completed','numerical_failure'):raise RuntimeError(record.get('error'))
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in completed_hashes.items())
g.save(out/'FINISHED.json',statuses)
print('All nine terminal records ready for audit.',flush=True)
