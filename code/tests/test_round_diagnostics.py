import json
import unittest
import numpy as np
from src.fl.diagnostics import round_diagnostics


class RoundDiagnostics(unittest.TestCase):
    def test_geometry_scores_and_global_ids(self):
        updates = [[np.array([x])] for x in (0., 2., 5.)]
        before = [u[0].copy() for u in updates]
        d = round_diagnostics(updates, [4, 8, 9], {9}, [0, 1], [1., 2., 6.])
        self.assertEqual(d['honest_rms_radius'], 1.)
        self.assertEqual(d['malicious_rms_distance_to_honest_centroid'], 4.)
        self.assertEqual(d['malicious_distance_over_honest_radius'], 4.)
        self.assertEqual(d['malicious_selected_count'], 0)
        s = d['score_selection']
        self.assertEqual(s['boundary_gap'], 4.)
        self.assertEqual(s['clients'][2]['signed_margin'], -4.)
        self.assertEqual(s['clients'][2]['client'], 9)
        json.dumps(d, allow_nan=False)
        for u, old in zip(updates, before):
            np.testing.assert_array_equal(u[0], old)

    def test_ties_absence_and_zero_spread(self):
        u = [[np.array([0.])]] * 3
        d = round_diagnostics(u, [0, 1, 2], {2}, [0, 2], [1., 1., 1.])
        self.assertIsNone(d['malicious_distance_over_honest_radius'])
        self.assertEqual(d['malicious_selected_count'], 1)
        self.assertEqual(d['score_selection']['boundary_gap'], 0.)
        self.assertFalse(d['score_selection']['clients'][1]['selected'])
        for malicious in (set(), {0, 1, 2}):
            d = round_diagnostics(u, [0, 1, 2], malicious, [0, 1, 2])
            self.assertIsNone(d['malicious_rms_distance_to_honest_centroid'])
            self.assertIsNone(d['score_selection'])
            json.dumps(d, allow_nan=False)

    def test_invalid_data(self):
        u = [[np.array([0.])]] * 2
        for ids, chosen, scores in (([0, 0], [0], None), ([0, 1], [2], None),
                                    ([0, 1], [0, 0], None), ([0, 1], [0], [1.])):
            with self.assertRaises(ValueError):
                round_diagnostics(u, ids, set(), chosen, scores)
        with self.assertRaises(FloatingPointError):
            round_diagnostics([[np.array([np.nan])]], [0], set(), [0])
