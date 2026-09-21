import unittest
from simulation.benchmark import run
class SimulationS14(unittest.TestCase):
 def test_bounded_reproducible_report(self):
  r=run(.02); self.assertEqual(r['cameras'],6); self.assertEqual(r['max_queue_depth'],0); self.assertFalse(r['hardware_verified'])
