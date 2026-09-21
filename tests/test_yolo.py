import unittest
from localization_contracts.yolo import runtime_status, YoloDetector

class SimulationS11Yolo(unittest.TestCase):
 def test_missing_runtime_is_explicit_and_never_synthetic(self):
  status=runtime_status('/tmp/does-not-exist-yolo.pt')
  self.assertFalse(status.available); self.assertEqual(status.hardware_verified,False)
  with self.assertRaises(RuntimeError): YoloDetector('/tmp/does-not-exist-yolo.pt')

if __name__=='__main__': unittest.main()
