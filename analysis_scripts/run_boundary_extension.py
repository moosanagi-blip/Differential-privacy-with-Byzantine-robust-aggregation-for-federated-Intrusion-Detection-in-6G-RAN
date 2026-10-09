"""Authorized targeted 36-condition extension; preserve the original campaign."""
from pathlib import Path
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import sys
import time
import torch

base=Path(__file__).resolve().parents[1]
project=base/'paper gpt/dp-fl-ids code'
sys.path.insert(0,str(project/'code'))
from src.experiment_config import RunConfig
from src.fl.simulation import run_federated
from src.utils.run_record import execute

for mode in ('private','nonprivate'):
    check=json.loads((base/f'work/boundary_integration_20261001/{mode}.validation.json').read_text())
    assert check['status']=='integration_passed'
if shutil.disk_usage(base).free < 10*2**30:
    raise RuntimeError('At least 10 GiB free required')
out=project/'results/boundary_extension_20261001'
out.mkdir(exist_ok=False)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
default=RunConfig(model='mlp',hidden=64,optimizer='sgd',lr=.1,momentum=.9,
    rounds=30,n_chunks=30,round_chunks=True,num_clients=10,local_epochs=1,batch_size=64,
    aggregator='multikrum',delta=1e-5,max_grad_norm=1.,byzantine_fraction=.3,
    n_byzantine_estimate=3,device='cpu',engine='sequential',rotate_byzantine=False)
configs=[]
for seed in (42,43,44):
    for private in (False,True):
        for cap in (.5,2.,5.):
            for attack in ('adaptive_krum','boundary_krum'):
                name=f'{attack}_{"dp5" if private else "nonprivate"}_cap{cap:g}_s{seed}'
                configs.append(replace(default,name=name,seed=seed,dp=private,
                    target_epsilon=5. if private else None,attack=attack,attack_scale=cap,
                    data_root=str(project/f'data_original53_v3_f64_p{seed}')))
assert len(configs)==36
def save(path,data):
    with path.open('x') as handle:json.dump(data,handle,indent=2,allow_nan=False)
save(out/'matrix.json',[c.to_dict() for c in configs])
sources=[project/'code/src/attacks/boundary.py',project/'code/src/attacks/byzantine.py',
         project/'code/src/fl/simulation.py',Path(__file__)]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
save(out/'STARTED.json',{'pid':os.getpid(),'utc':datetime.now(timezone.utc).isoformat(),
    'source_hashes':hashes,'threads':1,'scope':'finite-grid selection-aware search; not global optimisation',
    'protocol':'work/BOUNDARY_EXTENSION_PROTOCOL.md','estimated_active_hours':[20,30]})
statuses=[]
for cfg in configs:
    for path,digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Frozen source changed; stopping before next run')
    dest=out/cfg.name;dest.mkdir()
    save(dest/'config.json',cfg.to_dict())
    with (dest/'progress.jsonl').open('x') as log:
        def runner(config,verbose=False,on_event=None):
            def observe(event,payload):
                if on_event:on_event(event,payload)
                if event!='provenance':
                    log.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),
                        'event':event,**payload},allow_nan=False)+'\n');log.flush()
                if event in ('run_start','round_end','run_end'):
                    print(json.dumps({'name':cfg.name,'event':event,
                        'round':payload.get('round')},allow_nan=False),flush=True)
            return run_federated(config,verbose=False,on_event=observe)
        record=execute(cfg,runner)
    save(dest/'record.json',record)
    statuses.append({'name':cfg.name,'status':record['status']})
    if record['status'] not in ('completed','numerical_failure'):
        raise RuntimeError(f"Execution stopped: {record.get('error')}")
save(out/'FINISHED.json',statuses)
