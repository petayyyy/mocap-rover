import unittest
from simulation.acceptance_matrix import run_matrix
class SimulationS15Matrix(unittest.TestCase):
 def test_fault_and_replay_matrix_is_reproducible(self):
  a=run_matrix(7); b=run_matrix(7); self.assertEqual(a,b); self.assertTrue(a['all_pass']); self.assertFalse(a['truth_used_by_runtime']); self.assertEqual(len(a['scenarios']),11); self.assertTrue(next(x for x in a['scenarios'] if x['scenario'].startswith('clock_drift'))['pass']); self.assertTrue(next(x for x in a['scenarios'] if x['scenario'].startswith('clock_drift_jump'))['new_session']); self.assertTrue(all('observed_delay_ms' in x for x in a['scenarios'] if x['scenario'].startswith('delay_')))
