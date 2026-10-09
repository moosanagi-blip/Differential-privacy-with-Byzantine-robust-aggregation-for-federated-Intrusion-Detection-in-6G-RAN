"""Partition-only variation must preserve the prepared learning problem."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from src.data import build_scaled as data
from src.data.config import NETWORK_FEATURES


class PartitionSeed(unittest.TestCase):
    def test_reuse_changes_assignments_not_features_scaler_or_holdout(self):
        n = 500
        raw = pd.DataFrame({c: np.arange(n, dtype=float)+1 for c in NETWORK_FEATURES})
        raw[data.LABEL_COLUMN] = data.ATTACK_LABELS[0]
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            interim = base/'interim'/'corpus.parquet'
            first, second, legacy = [base/name for name in ('p42', 'p43', 'default')]
            with patch.object(data, '_chunks', return_value=iter([(0, raw)])):
                data.build(first, max_files=1, seed=42, partition_seed=42,
                           interim=interim, keep_interim=True, quiet=True)
            original_interim = interim.read_bytes()
            with patch.object(data, 'pass1', side_effect=AssertionError('Must reuse same augmentation')):
                for out, seed in ((second, 43), (legacy, None)):
                    data.build(out, max_files=1, seed=42, partition_seed=seed,
                               interim=interim, reuse_interim=True, keep_interim=True, quiet=True)
            self.assertEqual(original_interim, interim.read_bytes())
            for name in ('global_test.parquet', 'eval_subset.parquet'):
                a = data.pq.read_table(first/'partitions'/name).to_pandas()
                for out in (second, legacy):
                    pd.testing.assert_frame_equal(a, data.pq.read_table(out/'partitions'/name).to_pandas())
            scalers = [data.joblib.load(out/'artifacts'/'scaler.joblib') for out in (first, second, legacy)]
            for s in scalers[1:]:
                for attr in ('mean_', 'var_', 'scale_', 'n_samples_seen_'):
                    np.testing.assert_array_equal(getattr(scalers[0], attr), getattr(s, attr))
            def train(out):
                frames = []
                for i, p in enumerate(sorted((out/'partitions').glob('client_*.parquet'))):
                    f = data.pq.read_table(p).to_pandas()
                    f['client'] = i
                    frames.append(f)
                return pd.concat(frames).sort_values(data.ALL_FEATURES[0]).reset_index(drop=True)
            a, b, old = map(train, (first, second, legacy))
            pd.testing.assert_frame_equal(a.drop(columns='client'), b.drop(columns='client'))
            self.assertTrue((a.client != b.client).any())
            pd.testing.assert_frame_equal(a, old)
            manifest = json.loads((second/'partitions'/'partition_manifest.json').read_text())
            self.assertEqual(manifest['augmentation_seed'], 42)
            self.assertEqual(manifest['partition_seed'], 43)
            self.assertEqual(manifest['n_train'], scalers[0].n_samples_seen_)
            with self.assertRaisesRegex(ValueError, 'Interim configuration differs'):
                data.build(base/'bad', max_files=1, seed=44, partition_seed=43,
                           interim=interim, reuse_interim=True, keep_interim=True, quiet=True)
