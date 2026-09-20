import json, unittest
import numpy as np
from localization_contracts import Observation, TrackStatus, CameraStatus, CalibrationSet, ConfigStore, load_config
from localization_contracts.geometry import rotate_twist_and_covariance
from localization_contracts.adapters import Frame, ReplayAdapter

class ContractsTest(unittest.TestCase):
    def setUp(self): self.config = load_config("config/contracts.json")
    def test_config_stage_apply_rollback_and_digest(self):
        s = ConfigStore(self.config); old = s.active
        s.stage({"capture": {"nominal_fps": 30}}); self.assertEqual(s.active["capture"]["nominal_fps"], 15)
        s.apply(); self.assertEqual(s.active["capture"]["nominal_fps"], 30); s.rollback(); self.assertEqual(s.active, old)
        with self.assertRaises(ValueError): s.stage({"capture": {"width": 0}})
        self.assertEqual(len(ConfigStore.digest(old)), 64)
    def test_observation_validation_and_ground_truth_absent(self):
        o = Observation("1.0", "camera_1", 1, "m1", "friendly", 100, "sim", 2, 0, 110, 120, "cal-1", "arena", (1,2,0), (0.01,0.01,0.1), .9, "apriltag", marker_family="tag36h11", marker_id=0)
        self.assertIs(o.validate(), o); self.assertNotIn("ground_truth", o.__dataclass_fields__)
        with self.assertRaises(ValueError): Observation(**{**o.__dict__, "schema_version":"9.0"}).validate()
    def test_status_calibration_and_units(self):
        cs = CameraStatus("1.0","camera_1",True,"sim",None,"rgb",1600,1200,15,0,"cal-1","s1")
        self.assertIs(cs.validate(), cs)
        c = CalibrationSet("1.0", "cal-1", "a"*64, "arena", {"camera_1":{"K":[[1,0,0],[0,1,0],[0,0,1]],"position_world":[0,0,1]}}, 1)
        self.assertIs(c.validate(), c)
    def test_twist_and_covariance_rotate_together(self):
        r = np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        v,w,c = rotate_twist_and_covariance([1,0,0],[0,0,2],np.eye(6),r)
        np.testing.assert_allclose(v,[0,1,0]); np.testing.assert_allclose(w,[0,0,2]); np.testing.assert_allclose(c,np.eye(6))
    def test_replay_common_boundary(self):
        a = ReplayAdapter([Frame("camera_1", 0, 1, "sim", b"image")]); self.assertEqual(list(a.frames())[0].camera_id,"camera_1"); self.assertNotIn("ground_truth", a.frames().__repr__())

if __name__ == "__main__": unittest.main()
