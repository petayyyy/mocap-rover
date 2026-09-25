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

    def test_bottom_tag_transform_preserves_base_position(self):
        import cv2
        K=np.array([[500.,0,320],[0,500.,240],[0,0,1.]])
        Rbase=np.diag([1.,-1.,-1.]); Tbase=np.eye(4); Tbase[:3,:3]=np.eye(3); Tbase[:3,3]=[.12,-.08,2.]
        Ttag=Tbase.copy(); Ttag[:3,:3]=Tbase[:3,:3]@Rbase; Ttag[:3,3]=Tbase[:3,3]+Tbase[:3,:3]@np.array([0,0,.0146])
        rv,_=cv2.Rodrigues(Ttag[:3,:3]); obj=np.array([[-.2,.2,0],[.2,.2,0],[.2,-.2,0],[-.2,-.2,0]],float)
        corners,_=cv2.projectPoints(obj,rv,Ttag[:3,3],K,np.zeros((5,1)))
        observer=PnpAprilTagObserver(TagConfig(ids=(0,1),calibration_version='bottom'),K,[0]*5,
            {'rotation':np.eye(3),'translation':[0,0,0]}, {'rotation':Rbase,'translation':[0,0,.0146]})
        result=observer.observe(Detection('camera_1',3,1,tuple(map(tuple,corners.reshape(4,2))),1,2,3))
        self.assertIsNotNone(result); self.assertTrue(result.pose_6d_valid)
        self.assertAlmostEqual(result.position_m[0],.12,places=2); self.assertAlmostEqual(result.position_m[1],-.08,places=2); self.assertAlmostEqual(result.position_m[2],2.,places=2)

    def test_ippe_ambiguity_prefers_planar_rover_branch(self):
        import cv2
        K=np.array([[488.31,0,647.85],[0,493.63,513.41],[0,0,1.]])
        D=np.array([-.3113,.1543,-.00149,.00222,-.05947])
        Rwc=np.array([[0,-1,0],[-1,0,0],[0,0,-1.]],float)
        camera=np.array([3.,3.,2.9]); base=np.array([1.,1.,.14]); tag=base+[0,0,.2254]
        obj=np.array([[-.2,.2,0],[.2,.2,0],[.2,-.2,0],[-.2,-.2,0]],float)
        Rct=Rwc.T; tct=Rwc.T@(tag-camera); rv,_=cv2.Rodrigues(Rct)
        corners,_=cv2.projectPoints(obj,rv,tct,K,D)
        corners=corners.reshape(4,2)+np.random.default_rng(1).normal(0,.15,(4,2))
        observer=PnpAprilTagObserver(TagConfig(calibration_version='ambiguous'),K,D,
            {'rotation':Rwc,'translation':camera},
            {'rotation':np.eye(3),'translation':[0,0,.2254]})
        result=observer.observe(Detection('camera_3',4,0,tuple(map(tuple,corners)),1,2,3))
        self.assertIsNotNone(result)
        self.assertLess(np.linalg.norm(np.asarray(result.position_m)-base),.02)
        self.assertLess(result.pixel_features['planar_tilt_deg'],2.)
        self.assertEqual(result.pixel_features['pnp_candidate_count'],2)

if __name__ == "__main__": unittest.main()
