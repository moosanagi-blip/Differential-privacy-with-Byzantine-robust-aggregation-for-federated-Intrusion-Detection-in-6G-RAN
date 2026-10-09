import unittest

import numpy as np
import pandas as pd

from src.data import config, schema39 as schema


class Schema39(unittest.TestCase):
    def frame(self, labels):
        out = pd.DataFrame({c: np.arange(len(labels), dtype=float) + 1
                            for c in schema.NETWORK_FEATURES})
        out['Label'] = labels
        return out

    def test_legacy_schema_unchanged_and_new_manifest_explicit(self):
        self.assertEqual(len(config.ALL_FEATURES), 53)
        self.assertEqual(len(schema.NETWORK_FEATURES), 39)
        self.assertEqual(len(set(schema.ALL_FEATURES)), 46)
        self.assertEqual(schema.schema_manifest()['n_features'], 46)

    def test_label_normalization_and_float64_preservation(self):
        frame = self.frame(['BENIGN', 'DDOS-PSHACK_FLOOD', 'MIRAI-GREIP_FLOOD'])
        frame['Header_Length'] = [83000000., 83000001., 83000002.]
        accepted, counts = schema.validate_chunk(frame)
        self.assertEqual(accepted['label'].tolist(),
                         ['BenignTraffic', 'DDoS-PSHACK_Flood', 'Mirai-greip_flood'])
        self.assertEqual(counts['accepted_rows'], 3)
        self.assertEqual(accepted['Header_Length'].nunique(), 3)
        self.assertEqual(accepted['Header_Length'].dtype, np.dtype('float64'))

    def test_rejections_counted_without_imputation_or_double_counting(self):
        frame = self.frame(['BENIGN', 'UNKNOWN', None, 'BENIGN', 'UNKNOWN'])
        frame.loc[2, 'Rate'] = np.nan
        frame.loc[3, 'Rate'] = np.inf
        frame.loc[4, 'Rate'] = -np.inf
        accepted, counts = schema.validate_chunk(frame)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(counts['rejected_non_numeric_or_nonfinite'], 3)
        self.assertEqual(counts['rejected_unknown_or_missing_label'], 1)
        self.assertEqual(counts['input_rows'], 5)

    def test_directory_labels_and_header_contract(self):
        frame = self.frame(['BENIGN']).drop(columns='Label')
        accepted, _ = schema.validate_chunk(frame, directory_label='Benign_Final')
        self.assertEqual(accepted['label'].iloc[0], 'BenignTraffic')
        with self.assertRaises(ValueError):
            schema.validate_chunk(frame, directory_label='unrecognized')
        with self.assertRaises(ValueError):
            schema.validate_chunk(frame)
        with self.assertRaises(ValueError):
            schema.validate_chunk(frame.iloc[:, ::-1], directory_label='Benign_Final')

    def test_synthetic_augmentation_deterministic_and_network_unchanged(self):
        accepted, _ = schema.validate_chunk(self.frame(['BENIGN', 'DOS-UDP_FLOOD']))
        first = schema.augment_validated(accepted, seed=42)
        pd.testing.assert_frame_equal(first, schema.augment_validated(accepted, seed=42))
        pd.testing.assert_frame_equal(first[schema.NETWORK_FEATURES], accepted[schema.NETWORK_FEATURES])
        self.assertEqual(list(first.columns), schema.ALL_FEATURES + ['label'])


if __name__ == '__main__':
    unittest.main()
