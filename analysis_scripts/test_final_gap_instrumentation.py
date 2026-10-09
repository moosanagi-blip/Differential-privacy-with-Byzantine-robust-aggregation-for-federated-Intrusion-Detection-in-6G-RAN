"""Small deterministic instrumentation test, never publication evidence."""
import numpy as np
import torch
import run_final_gap_experiments as gap
from src.models.lstm_ids import build_model
from src.fl.params import get_params
torch.set_num_threads(1)
torch.manual_seed(7)
params=get_params(build_model('mlp',53,34,hidden=64).double())
rng=np.random.default_rng(9)
kw=dict(global_params=params,x=rng.normal(size=(96,53)),
    y=rng.integers(0,34,size=96),n_features=53,n_classes=34,model_name='mlp',
    hidden=64,epochs=1,batch_size=32,lr=.1,device=torch.device('cpu'),
    dp=True,noise_multiplier=1.,max_grad_norm=1.,seed=123,
    optimizer_name='sgd',momentum=.9)
baseline=gap.original_local(**kw);state=torch.get_rng_state().clone()
paired=gap.paired_local(**kw)
assert all(np.array_equal(a,b) for a,b in zip(baseline[0],paired[0]))
assert torch.equal(state,torch.get_rng_state())
first=gap.counterfactuals.pop()
gap.paired_local(**kw)
assert all(np.array_equal(a,b) for a,b in zip(first,gap.counterfactuals.pop()))
assert any(not np.array_equal(a,b) for a,b in zip(first,paired[0]))
updates=[[a+i*.0001 for a in paired[0]] for i in range(10)]
gap.counterfactuals.extend([[a+i*.0001 for a in first] for i in range(10)])
before=gap.original_aggregate('multikrum',updates,weights=[96]*10,n_byzantine=3)
after=gap.paired_aggregate('multikrum',updates,weights=[96]*10,n_byzantine=3)
assert before[1]==after[1]
assert all(np.array_equal(a,b) for a,b in zip(before[0],after[0]))
assert len(gap.diagnostics[-1]['clients'])==10
print('PASS: actual payloads, RNG and aggregation unchanged; replay deterministic and nonzero')
