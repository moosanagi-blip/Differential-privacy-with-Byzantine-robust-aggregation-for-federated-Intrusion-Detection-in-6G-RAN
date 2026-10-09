"""Separate, instrumented reviewer tests; never modifies frozen source modules."""
from pathlib import Path
from dataclasses import replace
import sys, json, hashlib, os
import numpy as np
import torch
base=Path(__file__).resolve().parents[1]
project=base/'paper gpt/dp-fl-ids code'
sys.path.insert(0,str(project/'code'))
from src.experiment_config import RunConfig
from src.fl import simulation
from src.fl.params import flatten
from src.utils.run_record import execute
from opacus.optimizers import DPOptimizer

original_local=simulation.local_train
original_aggregate=simulation.aggregation.aggregate
original_noise=DPOptimizer.add_noise
counterfactuals=[]
diagnostics=[]

def suppressed_noise(self):
    # Consume exactly the usual noise RNG draws, but remove their contribution.
    original_noise(self)
    for p in self.params:
        p.grad=p.summed_grad.clone().view_as(p)

def paired_local(**kwargs):
    actual=original_local(**kwargs)
    state=torch.get_rng_state().clone()
    try:
        DPOptimizer.add_noise=suppressed_noise
        replay=original_local(**kwargs)
    finally:
        DPOptimizer.add_noise=original_noise
        torch.set_rng_state(state)
    assert actual[1]==replay[1] and actual[2]==replay[2]
    counterfactuals.append(replay[0])
    return actual

def paired_aggregate(name,updates,**kwargs):
    result=original_aggregate(name,updates,**kwargs)
    assert len(counterfactuals)==len(updates)==10
    _,cf=original_aggregate(name,counterfactuals,**kwargs)
    clients=[]
    for i,(actual,clean) in enumerate(zip(updates,counterfactuals)):
        norm=float(np.linalg.norm(flatten(actual)-flatten(clean)))
        replacement=list(updates);replacement[i]=clean
        _,one=original_aggregate(name,replacement,**kwargs)
        clients.append({'client':i,'noise_induced_payload_change_l2':norm,
            'selected_actual':i in result[1]['selected'],
            'selected_all_noise_suppressed':i in cf['selected'],
            'selected_own_noise_suppressed':i in one['selected'],
            'actual_score':float(result[1]['scores'][i]),
            'own_noise_suppressed_score':float(one['scores'][i])})
    diagnostics.append({'round':len(diagnostics)+1,'clients':clients})
    counterfactuals.clear()
    return result

def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    smoke='--smoke' in sys.argv
    out=project/'results'/('final_gap_smoke_20261003' if smoke else 'final_gap_20261003')
    out.mkdir(exist_ok=False)
    default=RunConfig(model='mlp',hidden=64,optimizer='sgd',lr=.1,momentum=.9,
        rounds=30,n_chunks=30,round_chunks=True,num_clients=10,local_epochs=1,
        batch_size=64,aggregator='multikrum',dp=True,delta=1e-5,max_grad_norm=1.,
        n_byzantine_estimate=3,device='cpu',engine='sequential')
    configs=[]
    for seed in (42,43,44):
        for eps in (1.,10.):
            configs.append(replace(default,name=f'displacement_dp{eps:g}_s{seed}',
                seed=seed,target_epsilon=eps,attack='adaptive_krum',attack_scale=2.,
                byzantine_fraction=.3,data_root=str(project/f'data_original53_v3_f64_p{seed}')))
        configs.append(replace(default,name=f'noise_selection_dp5_s{seed}',seed=seed,
            target_epsilon=5.,attack='none',byzantine_fraction=0.,
            data_root=str(project/f'data_original53_v3_f64_p{seed}')))
    if smoke:
        # Full-size first slice, not publication evidence.
        configs=[replace(configs[2],rounds=1,target_epsilon=None,noise_multiplier=1.)]
    else:
        assert json.loads((project/'results/final_gap_smoke_20261003/PASSED.json').read_text())['passed']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (project/'code/src').rglob('*.py')}
    hashes[str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save(out/'STARTED.json',{'pid':os.getpid(),'source_sha256':hashes,'smoke_only':smoke})
    save(out/'matrix.json',[c.to_dict() for c in configs])
    statuses=[]
    for cfg in configs:
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        counterfactuals.clear();diagnostics.clear()
        paired=cfg.attack=='none'
        simulation.local_train=paired_local if paired else original_local
        simulation.aggregation.aggregate=paired_aggregate if paired else original_aggregate
        dest=out/cfg.name;dest.mkdir()
        with (dest/'progress.jsonl').open('x') as log:
            def runner(config,verbose=False,on_event=None):
                def observe(event,payload):
                    if on_event:on_event(event,payload)
                    if event!='provenance':
                        log.write(json.dumps({'event':event,**payload},allow_nan=False)+'\n');log.flush()
                    if event=='round_end':
                        if paired:save(dest/f'noise_round_{payload["round"]:02d}.json',diagnostics[-1])
                        print(cfg.name,event,payload['round'],flush=True)
                return simulation.run_federated(config,verbose=False,on_event=observe)
            record=execute(cfg,runner)
        save(dest/'record.json',record)
        statuses.append({'name':cfg.name,'status':record['status']})
        if record['status'] not in ('completed','numerical_failure'):
            raise RuntimeError(record.get('error'))
        if smoke:
            assert record['status']=='completed' and len(diagnostics)==1
            assert all(np.isfinite(c['noise_induced_payload_change_l2']) for c in diagnostics[0]['clients'])
            save(out/'PASSED.json',{'passed':True,'scope':'one full-size slice, instrumented clean DP'})
    save(out/'FINISHED.json',statuses)

if __name__=='__main__':main()
