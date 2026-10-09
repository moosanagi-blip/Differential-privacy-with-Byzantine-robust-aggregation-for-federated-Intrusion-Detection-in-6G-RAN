"""One normal-sized round per privacy mode; integration/timing, not final evidence."""
from pathlib import Path
import json
import sys
import time
import torch

base=Path(__file__).resolve().parents[1]
project=base/'paper gpt/dp-fl-ids code'
sys.path.insert(0,str(project/'code'))
from src.experiment_config import RunConfig
from src.fl.simulation import run_federated
from src.utils.run_record import execute

out=base/'work/boundary_integration_20261001'
out.mkdir(exist_ok=False)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
main=json.loads((project/'results/main_original53_float64_2026-09-26/record.json').read_text())
sigma=main['result']['noise_schedule'][0]
for private in (False,True):
    name='private' if private else 'nonprivate'
    cfg=RunConfig(model='mlp',hidden=64,optimizer='sgd',lr=.1,momentum=.9,
        rounds=1,n_chunks=30,round_chunks=True,num_clients=10,local_epochs=1,batch_size=64,
        aggregator='multikrum',dp=private,target_epsilon=None,noise_multiplier=sigma,
        delta=1e-5,max_grad_norm=1.,attack='boundary_krum',attack_scale=2.,
        byzantine_fraction=.3,n_byzantine_estimate=3,seed=42,device='cpu',
        engine='sequential',data_root=str(project/'data_original53_v3_f64_p42'),
        name='integration_only_boundary_'+name)
    start=time.monotonic()
    with (out/f'{name}.progress.jsonl').open('x') as log:
        def runner(config,verbose=False,on_event=None):
            def observe(event,payload):
                if on_event: on_event(event,payload)
                if event!='provenance':
                    log.write(json.dumps({'event':event,**payload},allow_nan=False)+'\n');log.flush()
            return run_federated(config,verbose=False,on_event=observe)
        record=execute(cfg,runner)
    with (out/f'{name}.record.json').open('x') as handle: json.dump(record,handle,indent=2,allow_nan=False)
    assert record['status']=='completed',record.get('error')
    event=record['completed_rounds'][0]
    attack=event['attack_measurement']
    assert len(attack['search']['candidates'])==41 and len(attack['clients'])==3
    if attack['search']['feasible_candidate_found']:
        assert set(event['byz_selected'])==set(event['byzantine_clients'])
    assert {c['client_id'] for c in attack['clients']}==set(event['byzantine_clients'])
    assert all(c['perturbation_l2']>=0 for c in attack['clients'])
    result={'status':'integration_passed','mode':name,'elapsed_seconds':time.monotonic()-start,
        'round_seconds':event['time_s'],'scope':'one of 30 slices; no final utility claim',
        'noise_multiplier':sigma if private else 0.,'chosen_scale':attack['search']['chosen_scale']}
    with (out/f'{name}.validation.json').open('x') as handle: json.dump(result,handle,indent=2)
    print(json.dumps(result),flush=True)
