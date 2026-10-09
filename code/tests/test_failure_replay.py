import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from replay_revision_run import execute
from src.experiment_config import RunConfig


class FailureReplay(unittest.TestCase):
    def test_failure_keeps_context_without_final_metrics(self):
        def fake(cfg, verbose, on_event):
            on_event('round_end', dict(round=1, loss=2., accuracy=.5))
            on_event('client_start', dict(client=4, round=2))
            raise FloatingPointError('Non-finite training loss')
        result = execute(RunConfig(), runner=fake)
        self.assertEqual(result['status'], 'numerical_failure')
        self.assertEqual(result['completed_round_count'], 1)
        self.assertEqual(result['failure_context']['client'], 4)
        self.assertIsNone(result['final'])
        json.dumps(result, allow_nan=False)

    def test_valid_result_kept(self):
        def fake(cfg, verbose, on_event):
            return {'final': {'accuracy': .8}}
        result = execute(RunConfig(), runner=fake)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['result']['final']['accuracy'], .8)

    def test_unrelated_errors_not_labelled_numerical_failure(self):
        def fake(cfg, verbose, on_event):
            raise FileNotFoundError('missing dataset')
        with self.assertRaises(FileNotFoundError):
            execute(RunConfig(), runner=fake)


if __name__ == '__main__':
    unittest.main()
