import unittest
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

if __name__ == "__main__": unittest.main()
