import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.fl.aggregation import krum_scores
from src.evaluation.metrics import predict
from src.training import evaluate, train_epochs


class NumericalGuards(unittest.TestCase):
    def test_finite_scores_unchanged(self):
        updates = [[np.array([float(i)])] for i in range(5)]
        np.testing.assert_allclose(krum_scores(updates, 1), [5, 2, 2, 2, 5])

    def test_krum_rejects_nan_and_inf(self):
        for value in (float('nan'), float('inf')):
            with self.assertRaises(FloatingPointError):
                krum_scores([[np.array([value])], [np.array([1.])], [np.array([2.])]], 0)

    def test_evaluation_and_training_reject_invalid_model(self):
        model = torch.nn.Linear(1, 2)
        with torch.no_grad():
            model.weight.fill_(float('nan'))
        loader = DataLoader(TensorDataset(torch.ones(2, 1), torch.zeros(2, dtype=torch.long)), batch_size=2)
        for fn in (predict, evaluate):
            with self.assertRaises(FloatingPointError):
                fn(model, loader, device=torch.device('cpu'))
        with self.assertRaises(FloatingPointError):
            train_epochs(model, loader, device=torch.device('cpu'))

    def test_valid_predictions_still_work(self):
        model = torch.nn.Linear(1, 2)
        loader = DataLoader(TensorDataset(torch.ones(2, 1), torch.zeros(2, dtype=torch.long)), batch_size=2)
        truth, predicted = predict(model, loader, torch.device('cpu'))
        self.assertEqual(len(truth), 2)
        self.assertEqual(len(predicted), 2)


if __name__ == '__main__':
    unittest.main()
