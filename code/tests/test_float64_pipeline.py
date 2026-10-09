import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from src.data import build_scaled as data
from src.data.config import NETWORK_FEATURES
from src.data.loaders import DataRoot
from src.fl.local import local_train
from src.fl.params import get_params
from src.models.lstm_ids import build_model
from src.experiment_config import RunConfig
from src.fl import simulation


class Float64Pipeline(unittest.TestCase):
    def test_sequential_training_and_final_evaluation_in_double(self):
        class Root:
            feature_dtype = 'float64'
            name = 'double-software-fixture'
            def num_features(self): return 3
            def num_classes(self): return 2
            def n_clients(self): return 2
            def client_rows(self, c): return 16
            def label_map(self): return {'BenignTraffic': 0, 'DDoS-UDP_Flood': 1}
            def load_client(self, c, **kwargs):
                return np.random.default_rng(c).normal(size=(16, 3)), np.arange(16)%2
            def test_loader(self, **kwargs):
                x, y = self.load_client(99)
                return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=8)
        cfg = RunConfig(model='mlp', hidden=8, rounds=1, num_clients=2, batch_size=8,
                        optimizer='sgd', lr=.001, eval_on_subset=False, device='cpu')
        with patch.object(simulation.loaders, 'DataRoot', return_value=Root()):
            result, model = simulation.run_federated(cfg, verbose=False, return_model=True)
        self.assertEqual(next(model.parameters()).dtype, torch.float64)
        self.assertEqual(result['final']['evaluated_rows'], 16)
        self.assertTrue(np.isfinite(result['final_accuracy']))

    def test_storage_loaders_and_close_values_preserve_precision(self):
        values = 1 + np.arange(500)*.001
        values[-1] = 1e12
        raw = pd.DataFrame({c: values for c in NETWORK_FEATURES})
        raw[data.LABEL_COLUMN] = data.ATTACK_LABELS[0]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'data'
            with patch.object(data, '_chunks', return_value=iter([(0, raw)])):
                data.build(out, max_files=1, feature_dtype='float64', keep_interim=True, quiet=True)
            root = DataRoot(out)
            x, _ = root.load_client(0)
            self.assertEqual(x.dtype, np.dtype('float64'))
            self.assertEqual(root.load_client_chunk(0, 0, 3)[0].dtype, x.dtype)
            self.assertEqual(root.load_client_chunk(0, 0, 100000)[0].dtype, x.dtype)
            held = next(iter(root.stream_test()))[0]
            self.assertEqual(held.dtype, torch.float64)
            self.assertEqual(next(iter(root.test_loader()))[0].dtype, torch.float64)
            self.assertGreater(len(np.unique(held[:, 0].numpy())), len(np.unique(held[:, 0].numpy().astype('float32'))))

    def test_private_and_nonprivate_training_stays_double(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(1)
        x = rng.normal(size=(16, 3)).astype('float64')
        params = get_params(build_model('mlp', 3, 2, hidden=8).double())
        for dp in (False, True):
            new, count, info = local_train(global_params=params, x=x, y=np.arange(16)%2,
                n_features=3, n_classes=2, model_name='mlp', hidden=8, epochs=1,
                batch_size=8, lr=.001, device=torch.device('cpu'), dp=dp,
                noise_multiplier=1.3, optimizer_name='sgd', momentum=0)
            self.assertEqual(count, 16)
            self.assertTrue(all(p.dtype == np.float64 and np.isfinite(p).all() for p in new))
            if dp: self.assertEqual(info['steps'], 2)
        with self.assertRaisesRegex(ValueError, 'precision must agree'):
            local_train(global_params=params, x=x.astype('float32'), y=np.arange(16)%2,
                n_features=3, n_classes=2, model_name='mlp', hidden=8, epochs=1,
                batch_size=8, lr=.001, device=torch.device('cpu'))
