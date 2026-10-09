import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data import config, schema39
from src.data.loaders import DataRoot, _frame_to_xy


class DatasetSchema(unittest.TestCase):
    def test_invalid_features_rejected_before_radio_masking(self):
        for invalid in (np.nan, np.inf, -np.inf, 1e300):
            frame = pd.DataFrame({c: [1.] for c in schema39.ALL_FEATURES})
            frame['label_id'] = [0]
            frame[schema39.ALL_FEATURES[-1]] = invalid
            with self.subTest(value=invalid), self.assertRaises(FloatingPointError):
                _frame_to_xy(frame, mask_ran_features=True, features=schema39.ALL_FEATURES)

    def create(self, folder, new):
        root = Path(folder)
        (root/'partitions').mkdir()
        (root/'artifacts').mkdir()
        features = schema39.ALL_FEATURES if new else config.ALL_FEATURES
        if new:
            (root/'artifacts/feature_schema.json').write_text(json.dumps(schema39.schema_manifest()))
        frame = pd.DataFrame({c: np.arange(5, dtype=np.float32) + i + 1 for i,c in enumerate(features)})
        frame['label_id'] = np.arange(5) % 2
        for name in ('client_00', 'global_test'):
            frame.to_parquet(root/f'partitions/{name}.parquet', index=False, row_group_size=2)
        return DataRoot(root)

    def test_both_schemas_full_chunk_empty_and_stream(self):
        for new, size in [(False, 53), (True, 46)]:
            with self.subTest(new=new), tempfile.TemporaryDirectory() as tmp:
                root = self.create(tmp, new)
                self.assertEqual(root.num_features(), size)
                full, y = root.load_client(0)
                self.assertEqual(full.shape, (5, size))
                chunk, _ = root.load_client_chunk(0, 1, 2)
                np.testing.assert_array_equal(chunk, full[2:])
                empty, _ = root.load_client_chunk(0, 0, 10)
                self.assertEqual(empty.shape, (0, size))
                streamed = np.concatenate([x.numpy() for x,_ in root.stream_test(batch_size=2)])
                np.testing.assert_array_equal(streamed, full)
                masked, _ = root.load_client(0, mask_ran_features=True)
                np.testing.assert_array_equal(masked[:, :-7], full[:, :-7])
                self.assertFalse(masked[:, -7:].any())
                self.assertEqual(next(iter(root.test_loader(batch_size=2)))[0].shape[1], size)

    def test_bad_manifest_and_missing_manifest_reject_new_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.create(tmp, True)
            path = Path(tmp)/'artifacts/feature_schema.json'
            manifest = schema39.schema_manifest()
            manifest['feature_order'] = list(reversed(manifest['feature_order']))
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'feature_order'):
                DataRoot(tmp)
            path.unlink()
            with self.assertRaises(KeyError):
                DataRoot(tmp).load_client(0)


if __name__ == '__main__':
    unittest.main()
