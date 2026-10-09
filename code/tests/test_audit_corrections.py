"""Fast regression tests for the pre-publication audit corrections."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.data import build_scaled as data
from src.models.lstm_ids import build_model
from src.privacy import adaptive_eps
from src.privacy.dp_engine import epsilon_from_schedule
from src.utils.reporting import row_from_result
from src.fl.aggregation import aggregate, krum_scores, multi_krum


class AuditCorrections(unittest.TestCase):
    def test_equal_weight_control_is_distinct_from_weighted_mean(self):
        updates=[[np.array([0.])],[np.array([10.])],[np.array([20.])]]
        weighted,_=aggregate('fedavg',updates,weights=[8,1,1])
        uniform,_=aggregate('uniformavg',updates,weights=[8,1,1])
        np.testing.assert_allclose(weighted[0],[3.])
        np.testing.assert_allclose(uniform[0],[10.])
        for weights in ([0,0,0],[-1,1,1],[float('nan'),1,1]):
            with self.assertRaises(ValueError):
                aggregate('fedavg',updates,weights=weights)

    def test_krum_rejects_invalid_counts_without_clamping(self):
        updates=[[np.array([float(i)])] for i in range(5)]
        for f in (-1,3,.5):
            with self.assertRaises(ValueError):
                krum_scores(updates,f)
        for m in (0,6,1.5):
            with self.assertRaises(ValueError):
                multi_krum(updates,1,n_selected=m)

    def test_holdout_is_batch_boundary_invariant(self):
        ids = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0])
        full = data._test_mask(ids, np.zeros(34, dtype=int))
        seen = np.zeros(34, dtype=int)
        split = np.concatenate([data._test_mask(x, seen) for x in (ids[:3], ids[3:8], ids[8:])])
        np.testing.assert_array_equal(full, split)

    def test_streaming_scaler_never_fits_test_outliers(self):
        # First and sixth rows of this class are held out, even across batches.
        values = np.array([100000, 1, 2, 3, 4, -100000, 5, 6, 7, 8], dtype=np.float32)
        frame = pd.DataFrame({c: values for c in data.ALL_FEATURES})
        frame['label_id'] = np.zeros(10, dtype=np.int16)
        frame['category_id'] = np.zeros(10, dtype=np.int8)
        chunks = [(0, frame.iloc[:3]), (1, frame.iloc[3:])]
        with tempfile.TemporaryDirectory() as tmp:
            interim = Path(tmp)/'interim.parquet'
            with patch.object(data, '_chunks', return_value=iter(chunks)), patch.object(data, '_prepare', side_effect=lambda raw,k,seed: raw):
                stats = data.pass1(interim, None, 3, 42, quiet=True)
            self.assertEqual(stats['scaler_n'], 8)
            np.testing.assert_allclose(stats['scaler_mean'], 4.5)
            root = data._Root(Path(tmp)/'new_data')
            data.pass2(interim, stats, root, 42, quiet=True)
            self.assertEqual(int(root.n_client.sum()), 8)
            self.assertEqual(root.n_test, 2)
            with self.assertRaises(FileExistsError):
                data.build(Path(tmp)/'new_data', quiet=True)

    def test_zero_noise_is_not_zero_privacy_cost(self):
        self.assertTrue(np.isinf(epsilon_from_schedule([(0, .1, 10)])))
        self.assertEqual(epsilon_from_schedule([]), 0)
        with self.assertRaises(ValueError):
            epsilon_from_schedule([(1, 1.1, 10)])

    def test_raw_precision_survives_prepare_interim_and_scaling(self):
        values = 83000000.0 + np.arange(10, dtype=np.float64)
        self.assertLess(len(np.unique(values.astype(np.float32))), len(values))
        raw = pd.DataFrame({c: values for c in data.ALL_FEATURES})
        raw[data.LABEL_COLUMN] = data.ATTACK_LABELS[0]
        with patch.object(data, 'clean', side_effect=lambda x: x), patch.object(
                data, 'add_ran_features', side_effect=lambda x, rng: x):
            prepared = data._prepare(raw, 0, 42)
        np.testing.assert_array_equal(prepared[data.ALL_FEATURES[0]], values)
        with tempfile.TemporaryDirectory() as tmp:
            interim = Path(tmp)/'interim.parquet'
            with patch.object(data, '_chunks', return_value=iter([(0, prepared)])), patch.object(
                    data, '_prepare', side_effect=lambda raw,k,seed: raw):
                stats = data.pass1(interim, None, 10, 42, quiet=True)
            saved = data.pq.read_table(interim).to_pandas()
            np.testing.assert_array_equal(saved[data.ALL_FEATURES[0]], values)
            self.assertEqual(saved[data.ALL_FEATURES[0]].dtype, np.dtype('float64'))
            root = data._Root(Path(tmp)/'new_data')
            data.pass2(interim, stats, root, 42, quiet=True)
            arrays = [data.pq.read_table(p).column(data.ALL_FEATURES[0]).to_numpy()
                      for p in root.parts.glob('client_*.parquet')]
            arrays.append(data.pq.read_table(root.parts/'global_test.parquet').column(
                data.ALL_FEATURES[0]).to_numpy())
            scaled = np.concatenate(arrays)
            self.assertEqual(scaled.dtype, np.dtype('float32'))
            self.assertEqual(len(np.unique(scaled)), 10)

    def test_previous_precision_version_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Incompatible preprocessing'):
            data.pass2(Path('unused'), {'preprocessing_version': 'train-only-scaler-v2'},
                       None, 42, quiet=True)

    def test_calibration_returns_safe_side(self):
        # Deterministic monotonic stand-in isolates the bisection contract.
        with patch.object(adaptive_eps, 'epsilon_from_schedule', side_effect=lambda s,delta: 1/s[0][0]):
            sched = adaptive_eps.build('static', rounds=2, target_epsilon=3,
                                       sample_rate=.1, steps_per_round=10)
            self.assertLessEqual(1/sched[0], 3)
            limited = adaptive_eps.build('static', rounds=2, target_epsilon=.3,
                                         sample_rate=.1, steps_per_round=10, max_iter=1)
            self.assertLessEqual(1/limited[0], .3)

    def test_transformer_common_factory_arguments(self):
        model = build_model('transformer',53,34,hidden=64)
        self.assertEqual(tuple(model(torch.zeros(2,53)).shape), (2,34))

    def test_summary_uses_full_test_and_rejects_failed_runs(self):
        run = {'config': {'name':'test','aggregator':'fedavg','dp':False,
                          'byzantine_fraction':0,'attack':'none'},
               'final': {'accuracy':.8,'macro_f1':.4,'fpr':.2},
               'final_accuracy':.9, 'timeline':[]}
        self.assertEqual(row_from_result(run)['accuracy'], .8)
        run['timeline']=[{'round':1,'loss':float('nan')}]
        with self.assertRaises(ValueError):
            row_from_result(run)


if __name__ == '__main__':
    unittest.main()
