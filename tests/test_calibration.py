import tempfile, unittest
from pathlib import Path
from localization_contracts.calibration import *

class SimulationS06(unittest.TestCase):
    def setUp(self): self.c=Intrinsics("pinhole",100,80,[50,0,50,0,50,40,0,0,1],[.1,-.02,0,0,0],focal_length_mm=2)
    def test_distortion_roundtrip_and_resize(self):
        x,y=distort_point(.2,-.1,self.c.D); u,v=undistort_point(x,y,self.c.D)
        self.assertAlmostEqual(u,.2,places=6); self.assertAlmostEqual(v,-.1,places=6)
        self.assertEqual(scale_intrinsics(self.c,200,160).status,"stale")
    def test_json_camera_info_bad_file_and_report(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"c.json"; save_json(p,self.c); self.assertEqual(load_json(p).K,self.c.K)
            p.write_text("broken");
            with self.assertRaises(ValueError): load_json(p)
        self.assertEqual(calibration_report("camera_1",self.c,.2)["focal_length_mm_role"],"metadata_only")
    def test_camera_info_yaml_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'camera.yaml'; save_camera_info_yaml(p,self.c); loaded=load_camera_info_yaml(p); self.assertEqual(loaded.K,self.c.K); self.assertEqual(loaded.D,self.c.D); self.assertEqual(loaded.width,100)

if __name__ == "__main__": unittest.main()
