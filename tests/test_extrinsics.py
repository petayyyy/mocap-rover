import unittest
import numpy as np
from localization_contracts.extrinsics import *

def pose(cid, q=0.9): return CameraPose(cid,(1,2,3),(1,0,0,0,1,0,0,0,1),quality=q,intrinsics_version="i1")

class SimulationS07(unittest.TestCase):
    def test_connected_graph_and_atomic_activation(self):
        g=CalibrationGraph(["camera_1","camera_2"]); g.add_observation("camera_1","board_a",range(10)); g.add_observation("camera_2","board_a",range(10))
        out=g.solve({"camera_1":pose("camera_1"),"camera_2":pose("camera_2")})
        a=CalibrationActivation(); a.activate(out); self.assertEqual(a.active["camera_1"].quality,.9)
        with self.assertRaises(ValueError): a.activate({"camera_1":pose("camera_1",0)})
        self.assertEqual(a.active["camera_2"].quality,.9)
    def test_disconnected_or_sparse_graph_rejected(self):
        g=CalibrationGraph(["camera_1","camera_2"]); g.add_observation("camera_1","board",range(20))
        with self.assertRaises(ValueError): g.solve({"camera_1":pose("camera_1"),"camera_2":pose("camera_2")})

    def test_image_derived_pnp_pose_and_reprojection(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV unavailable")
        g=CalibrationGraph(["camera_1"]); K=np.array([[500.,0,320],[0,500.,240],[0,0,1.]])
        obj=np.array([[-.2,-.2,2],[.2,-.2,2],[.2,.2,2],[-.2,.2,2],[0,0,2.1],[.1,.1,2.2]],float)
        img,_=cv2.projectPoints(obj,np.zeros((3,1)),np.zeros((3,1)),K,np.zeros((5,1)))
        g.add_image_observation("camera_1","board",obj,img.reshape(-1,2),K)
        out=g.solve_image_observations(); self.assertAlmostEqual(out["camera_1"].position_m[2],0.,places=3)
        self.assertGreater(out["camera_1"].quality,.9)

    def test_bundle_adjustment_refines_all_image_observations(self):
        from simulation.calibration_experiment import run
        result=run(noise_px=0.15); self.assertLess(result['bundle_translation_error_m'],.01)

if __name__ == "__main__": unittest.main()
