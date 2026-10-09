import json
import unittest
from unittest.mock import patch
import numpy as np

from src.evaluation import metrics


class ClassMetrics(unittest.TestCase):
    labels = {'BenignTraffic': 0, 'DDoS-UDP_Flood': 1, 'XSS': 2, 'SqlInjection': 3}

    def test_counts_rates_and_absent_classes(self):
        report = metrics.class_metrics(np.array([0,0,1,1,2]), np.array([0,1,1,2,2]), self.labels)
        self.assertEqual(report['confusion_matrix'], [[1,1,0,0],[0,1,1,0],[0,0,1,0],[0,0,0,0]])
        self.assertEqual(report['evaluated_rows'], 5)
        self.assertEqual(report['per_class']['DDoS-UDP_Flood']['f1'], .5)
        self.assertEqual(report['per_class']['XSS']['precision'], .5)
        self.assertEqual(report['per_class']['XSS']['recall'], 1.)
        self.assertIsNone(report['per_class']['SqlInjection']['recall'])
        self.assertIsNone(report['per_class']['SqlInjection']['precision'])
        self.assertEqual(sum(c['support'] for c in report['per_class'].values()), 5)
        json.dumps(report, allow_nan=False)

    def test_invalid_ids_shapes_and_maps_fail(self):
        for true,pred in [([0],[4]), ([-1],[0]), ([0.5],[0]), ([0,1],[0]), ([],[])]:
            with self.subTest(true=true,pred=pred), self.assertRaises(ValueError):
                metrics.class_metrics(np.array(true),np.array(pred),self.labels)
        with self.assertRaises(ValueError):
            metrics.class_metrics(np.array([0]),np.array([0]),{'one':0,'two':0})

    def test_evaluation_uses_explicit_dataset_map_not_global_map(self):
        # Deliberately swap benign/class IDs to reveal global-map dependence.
        labels={'DDoS-UDP_Flood':0,'BenignTraffic':1}
        with patch.object(metrics, 'predict', return_value=(np.array([0,1,1]),np.array([0,0,1]))), patch.object(
                metrics.loaders,'load_label_map',side_effect=AssertionError('global map accessed')):
            report=metrics.evaluate_full(None,None,None,label_map=labels)
        self.assertEqual(report['fpr'], .5)
        self.assertEqual(report['per_category']['Benign']['support'],2)
        self.assertEqual(report['per_class']['BenignTraffic']['recall'],.5)


if __name__ == '__main__':
    unittest.main()
