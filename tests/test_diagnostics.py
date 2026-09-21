import unittest
from localization_contracts.diagnostics import *
class SimulationS13(unittest.TestCase):
 def test_measured_filtered_predicted_and_metrics(self):
  d=Diagnostics(); d.record('measured',1,latency_ms=2); d.record('predicted',2,latency_ms=4); d.channel('camera_1',1,2,10); r=d.report(); self.assertEqual(r['latency_ms']['p95'],2); self.assertFalse(r['truth_visible'])
