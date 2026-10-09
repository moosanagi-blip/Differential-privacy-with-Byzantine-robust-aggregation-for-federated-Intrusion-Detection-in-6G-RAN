import unittest
from unittest.mock import patch

import numpy as np
import torch
from opacus.accountants import RDPAccountant

from src.experiment_config import RunConfig
from src.fl.local import local_train
from src.fl.params import get_params
from src.models.lstm_ids import build_model
from src.privacy.dp_engine import epsilon_from_schedule, _RDP_ALPHAS
from src.privacy.execution import (
    client_profiles, calibrated_schedule, ExecutionLedger, loader_mechanism,
)


class ExecutedPrivacy(unittest.TestCase):
    def test_uneven_chunk_profile_matches_loader_boundaries(self):
        profiles=client_profiles([65, 97], rounds=4, batch_size=16, epochs=2, n_chunks=3)
        for client,size in enumerate([65,97]):
            for r in range(4):
                j=r%3
                n=size*(j+1)//3-size*j//3
                self.assertEqual(profiles[client][r],loader_mechanism(n,16,2))
        self.assertNotEqual(loader_mechanism(65,32,1)[0],32/65)

    def test_live_opacus_history_and_rdp_parity(self):
        model=build_model('mlp',3,2,hidden=8)
        rng=np.random.default_rng(10)
        _,_,info=local_train(global_params=get_params(model),
            x=rng.normal(size=(65,3)).astype('float32'), y=np.arange(65)%2,
            n_features=3,n_classes=2,model_name='mlp',hidden=8,
            epochs=2,batch_size=32,lr=.01,device=torch.device('cpu'),
            dp=True,noise_multiplier=1.3,max_grad_norm=1,seed=100,
            optimizer_name='sgd',momentum=0)
        self.assertEqual(info['sample_rate'],1/3)
        self.assertEqual(info['steps'],6)
        self.assertEqual(info['expected_batch_size'],21)
        ledger=ExecutionLedger(2,1e-5)
        ledger.record(0,info,(1/3,6),1.3)
        ledger.record(0,info,(1/3,6),1.3)
        reference=RDPAccountant()
        for _ in range(12):
            reference.step(noise_multiplier=1.3,sample_rate=1/3)
        self.assertAlmostEqual(ledger.epsilon_by_client()[0],
            reference.get_epsilon(delta=1e-5,alphas=_RDP_ALPHAS),places=12)
        self.assertEqual(ledger.epsilon_by_client()[1],0)
        with self.assertRaises(ValueError):
            ledger.record(0,info,(32/65,6),1.3)
        with self.assertRaises(ValueError):
            ledger.record(0,{},(1/3,6),1.3)

    def test_calibration_bounds_every_profile(self):
        cfg=RunConfig(dp=True,rounds=2,target_epsilon=5,dp_calibrate='worst')
        profiles=client_profiles([65,97],rounds=2,batch_size=32,epochs=1,n_chunks=2)
        schedule=calibrated_schedule(cfg,profiles)
        for profile in profiles:
            eps=epsilon_from_schedule([(s,q,n) for s,(q,n) in zip(schedule,profile)])
            self.assertLessEqual(eps,5)
        cfg.dp_calibrate='mean'
        with self.assertRaises(ValueError):
            calibrated_schedule(cfg,profiles)

    def test_nonpositive_data_rejected(self):
        with self.assertRaises(ValueError):
            client_profiles([1],rounds=3,batch_size=4,epochs=1,n_chunks=3)

    def test_sequential_engine_persists_each_client_history(self):
        from src.fl import simulation
        class Root:
            def num_features(self): return 3
            def label_map(self): return {'BenignTraffic':0,'DDoS-UDP_Flood':1}
            name='synthetic-test-only'
            def n_clients(self): return 2
            def client_rows(self,c): return [65,97][c]
            def num_classes(self): return 2
            def load_client(self,c,**kwargs):
                return np.zeros((self.client_rows(c),3),dtype='float32'),np.zeros(self.client_rows(c),dtype='int64')
            def test_loader(self,**kwargs):
                return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.zeros(4,3),torch.zeros(4,dtype=torch.long)))
        def train(**kwargs):
            n=len(kwargs['y']);q,steps=loader_mechanism(n,32,1)
            return kwargs['global_params'],n,{'accountant_history':[(1.3,q,steps)]}
        cfg=RunConfig(dp=True,rounds=2,num_clients=2,model='mlp',hidden=8,
                      batch_size=32,noise_multiplier=1.3,eval_on_subset=False,device='cpu')
        with patch.object(simulation.loaders,'DataRoot',return_value=Root()), \
             patch.object(simulation.loaders,'N_FEATURES',3), \
             patch.object(simulation,'local_train',side_effect=train), \
             patch.object(simulation,'evaluate',return_value={'accuracy':.5,'macro_f1':.3,'loss':1.}), \
             patch.object(simulation.M,'evaluate_full',return_value={'accuracy':.5,'macro_f1':.3,'fpr':.2}):
            result=simulation.run_federated(cfg,verbose=False)
        histories=result['privacy_accounting']['history_by_client']
        self.assertEqual(histories[0],[(1.3,1/3,3)]*2)
        self.assertEqual(histories[1],[(1.3,.25,4)]*2)
        self.assertEqual(result['final_epsilon'],max(result['privacy_accounting']['epsilon_by_client'].values()))
        for entry in result['timeline']:
            diagnostics = entry['diagnostics']
            self.assertEqual(diagnostics['participants'], [0, 1])
            self.assertEqual(diagnostics['honest_rms_radius'], 0.)
            self.assertIsNone(diagnostics['score_selection'])


if __name__=='__main__':
    unittest.main()
