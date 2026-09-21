import unittest
from simulation.benchmark import run,soak,image_path
class SimulationS14(unittest.TestCase):
 def test_bounded_reproducible_report(self):
  r=run(.02); self.assertEqual(r['cameras'],6); self.assertEqual(r['max_queue_depth'],0); self.assertFalse(r['hardware_verified'])
 def test_thirty_minute_simulation_soak_is_bounded(self):
  r=soak(30,drop_period=17); self.assertEqual(r['simulation_minutes'],30); self.assertTrue(r['bounded']); self.assertGreater(r['drops'],0); self.assertFalse(r['raw_frames_retained'])
 def test_image_path_profile_is_not_empty_loop(self):
  r=image_path(.02); self.assertGreater(r['frames'],0); self.assertEqual(r['detected'],r['accepted']); self.assertFalse(r['raw_frames_retained']); self.assertGreater(r['latency_ms_p95'],0)
