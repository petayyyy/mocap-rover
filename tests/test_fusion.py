import unittest
from localization_contracts.apriltag import *
from localization_contracts.fusion import *
class SimulationS09(unittest.TestCase):
 def obs(self,n,t,x): return AprilTagObserver(TagConfig(calibration_version="c"),500,500,320,240).observe(Detection("c",n,0,((300+x,200),(400+x,200),(400+x,300),(300+x,300)),t,t+1,t+2))
 def test_async_motion_dedup_coasting_lost(self):
  f=PlanarFusion(); self.assertTrue(f.update(self.obs(1,0,0))); self.assertFalse(f.update(self.obs(1,0,0))); f.update(self.obs(2,100_000_000,10)); self.assertEqual(f.publish(150_000_000)["tracking_state"],"COASTING"); self.assertEqual(f.publish(400_000_000)["tracking_state"],"LOST")
 def test_out_of_order_rejected(self):
  f=PlanarFusion(); f.update(self.obs(2,100,0)); self.assertFalse(f.update(self.obs(1,50,0)))
 def test_calibration_change_resets_session(self):
  f=PlanarFusion(); first=self.obs(1,0,0); f.update(first); old=f.session; changed=self.obs(2,100,0)
  from dataclasses import replace
  self.assertTrue(f.update(replace(changed,calibration_version='new-cal'))); self.assertGreater(f.session,old); self.assertEqual(f.measurements,1)
 def test_neighbor_camera_quality_hysteresis(self):
  s=ObservationSelector(.15); a=self.obs(1,0,0); b=self.obs(2,0,1)
  from dataclasses import replace
  a=replace(a,quality=.8,camera_id='camera_1'); b=replace(b,quality=.9,camera_id='camera_2'); self.assertEqual(s.select([a,b]).camera_id,'camera_2'); b2=replace(b,quality=.85); self.assertEqual(s.select([a,b2]).camera_id,'camera_2')
if __name__=='__main__': unittest.main()
