import unittest
from simulation.pipeline import run
class SimulationAcceptance(unittest.TestCase):
 def test_nominal_rates_and_truth_boundary(self):
  r=run(2); self.assertGreaterEqual(r['friendly_hz'],20); self.assertGreaterEqual(r['opponent_hz'],10); self.assertFalse(r['ground_truth_used_by_runtime']); self.assertEqual(r['lost_publishes'],0)
 def test_single_channel_drop_degrades_without_global_stop(self):
  r=run(2,drop_probability=.4); self.assertGreater(r['friendly_output'],0); self.assertGreater(r['dropped_frames'],0); self.assertFalse(r['ground_truth_used_by_runtime'])
