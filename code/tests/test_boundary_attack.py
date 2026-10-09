import unittest
import numpy as np
from src.attacks.boundary import craft_boundary, perturbation_metrics
from src.fl.aggregation import multi_krum

class BoundaryTests(unittest.TestCase):
    def test_actual_selection_and_no_mutation(self):
        updates=[[np.array([1.+i*.03,.1*i],dtype=np.float64)] for i in range(10)]
        before=[u[0].copy() for u in updates]
        out,info=craft_boundary(updates,[np.zeros(2)], [0,6,9],3,5.)
        feasible=[c for c in info['candidates'] if c['all_attackers_selected']]
        self.assertEqual(len(info['candidates']),41)
        if feasible:
            self.assertEqual(info['chosen_scale'],max(c['scale'] for c in feasible))
            self.assertTrue({0,6,9}.issubset(multi_krum(out,3)[1]))
        for a,b in zip(before,updates): np.testing.assert_array_equal(a,b[0])

    def test_exact_displacement(self):
        before=[[np.array([0.,0.])],[np.array([2.,0.])],[np.array([1.,0.])]]
        after=before[:2]+[[np.array([4.,4.])]]
        m=perturbation_metrics(before,after,[np.zeros(2)],[2])
        self.assertEqual(m['honest_rms_radius'],1.)
        self.assertEqual(m['clients'][0]['perturbation_l2'],5.)
        self.assertEqual(m['clients'][0]['perturbation_over_pre_attack_update'],5.)

    def test_zero_denominators(self):
        u=[[np.zeros(2)] for _ in range(4)]
        m=perturbation_metrics(u,u,[np.zeros(2)],[0])
        self.assertIsNone(m['clients'][0]['perturbation_over_honest_radius'])
        self.assertIsNone(m['clients'][0]['perturbation_over_pre_attack_update'])

    def test_no_feasible_candidate_preserves_payloads(self):
        updates=[[np.array([float(i),1.])] for i in range(10)]
        # f=7 selects only three clients, so four attackers cannot all be selected.
        result,info=craft_boundary(updates,[np.zeros(2)],[0,1,2,3],7,2.)
        self.assertIs(result,updates)
        self.assertFalse(info['feasible_candidate_found'])
        self.assertIsNone(info['chosen_scale'])
        self.assertTrue(all(not c['all_attackers_selected'] for c in info['candidates']))

    def test_ties_use_actual_selector(self):
        updates=[[np.zeros(2)] for _ in range(10)]
        result,info=craft_boundary(updates,[np.zeros(2)],[0,6,9],3,2.)
        selected=multi_krum(updates,3)[1]
        expected={0,6,9}.issubset(selected)
        self.assertTrue(all(c['all_attackers_selected']==expected for c in info['candidates']))
        self.assertEqual(info['feasible_candidate_found'],expected)
