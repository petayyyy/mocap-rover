import unittest
import numpy as np
from localization_contracts.apriltag import *

class SimulationS08(unittest.TestCase):
    def setUp(self): self.o=AprilTagObserver(TagConfig(calibration_version="c1"),500,500,320,240)
    def d(self, tag=0, side=100): return Detection("camera_1",1,tag,((270,190),(270+side,190),(270+side,190+side),(270,190+side)),100,110,120)
    def test_observation_identity_geometry_and_quality(self):
        x=self.o.observe(self.d()); self.assertEqual(x.marker_id,0); self.assertEqual(x.calibration_version,"c1"); self.assertGreater(x.position_m[2],1)
        self.assertFalse(x.pose_6d_valid); self.assertEqual(x.attitude_state, "unknown")
    def test_wrong_id_missing_and_tiny_marker_rejected(self):
        self.assertIsNone(self.o.observe(self.d(9))); self.assertIsNone(self.o.observe(self.d(0,1)))

    def test_pnp_transform_and_metric_pose(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV unavailable")
        K=np.array([[500.,0,320],[0,500.,240],[0,0,1.]])
        object_points=np.array([[-.2,.2,0],[.2,.2,0],[.2,-.2,0],[-.2,-.2,0]], dtype=float)
        corners,_=cv2.projectPoints(object_points, np.zeros((3,1)), np.array([[.1],[-.05],[2.]]), K, np.zeros((5,1)))
        observer=PnpAprilTagObserver(TagConfig(calibration_version="pnp"),K,[0,0,0,0,0],
            {"rotation":np.eye(3),"translation":[0,0,0]}, {"rotation":np.eye(3),"translation":[0,0,.2254]})
        d=Detection("camera_1",2,0,tuple(map(tuple,corners.reshape(4,2))),100,110,120)
        result=observer.observe(d)
        self.assertIsNotNone(result); self.assertTrue(result.pose_6d_valid)
        self.assertEqual(result.attitude_state,"valid")
        self.assertAlmostEqual(result.position_m[0], .1, places=3)
        self.assertAlmostEqual(result.position_m[1], -.05, places=3)
        self.assertAlmostEqual(result.position_m[2], 2-.2254, places=3)

if __name__ == "__main__": unittest.main()
