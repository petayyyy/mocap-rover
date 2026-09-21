import unittest
from localization_contracts.apriltag import *
from localization_contracts.fusion import *
class SimulationS09(unittest.TestCase):
 def obs(self,n,t,x): return AprilTagObserver(TagConfig(calibration_version="c"),500,500,320,240).observe(Detection("c",n,0,((300+x,200),(400+x,200),(400+x,300),(300+x,300)),t,t+1,t+2))
 def test_async_motion_dedup_coasting_lost(self):
  f=PlanarFusion(); self.assertTrue(f.update(self.obs(1,0,0))); self.assertFalse(f.update(self.obs(1,0,0))); f.update(self.obs(2,100_000_000,10)); self.assertEqual(f.publish(150_000_000)["tracking_state"],"COASTING"); self.assertEqual(f.publish(400_000_000)["tracking_state"],"LOST")
 def test_out_of_order_rejected(self):
  f=PlanarFusion(); f.update(self.obs(2,100,0)); self.assertFalse(f.update(self.obs(1,50,0)))
if __name__=='__main__': unittest.main()
