import unittest
from localization_contracts.apriltag import *

class SimulationS08(unittest.TestCase):
    def setUp(self): self.o=AprilTagObserver(TagConfig(calibration_version="c1"),500,500,320,240)
    def d(self, tag=0, side=100): return Detection("camera_1",1,tag,((270,190),(270+side,190),(270+side,190+side),(270,190+side)),100,110,120)
    def test_observation_identity_geometry_and_quality(self):
        x=self.o.observe(self.d()); self.assertEqual(x.marker_id,0); self.assertEqual(x.calibration_version,"c1"); self.assertGreater(x.position_m[2],1)
        self.assertFalse(x.pose_6d_valid); self.assertEqual(x.attitude_state, "unknown")
    def test_wrong_id_missing_and_tiny_marker_rejected(self):
        self.assertIsNone(self.o.observe(self.d(9))); self.assertIsNone(self.o.observe(self.d(0,1)))

if __name__ == "__main__": unittest.main()
