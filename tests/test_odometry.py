import unittest
from localization_contracts.apriltag import AprilTagObserver, Detection, TagConfig
from localization_contracts.fusion import PlanarFusion
from localization_contracts.odometry import TimedOdometryPublisher

class SimulationS09Odometry(unittest.TestCase):
 def obs(self,t,n):
  return AprilTagObserver(TagConfig(calibration_version='c'),500,500,320,240).observe(Detection('c',n,0,((270,190),(370,190),(370,290),(270,290)),t,t+1,t+2))
 def test_independent_30hz_publisher_from_15hz_measurements(self):
  f=PlanarFusion(dropout_horizon_ms=200); p=TimedOdometryPublisher(f,30)
  samples=[]
  for i in range(10):
   t=i*66_666_666; f.update(self.obs(t,i)); x=p.step(t); 
   if x: samples.append(x)
   x=p.step(t+33_333_333)
   if x: samples.append(x)
  self.assertGreaterEqual(len(samples),18); self.assertTrue(all(x.frame_id=='arena' and x.child_frame_id=='base_link' for x in samples)); self.assertGreaterEqual(samples[-1].output_hz,20)
 def test_publisher_preserves_lost_state(self):
  f=PlanarFusion(dropout_horizon_ms=100); f.update(self.obs(0,0)); p=TimedOdometryPublisher(f,30); p.step(0); x=p.step(200_000_000); self.assertEqual(x.tracking_state,'LOST'); self.assertFalse(x.valid)

if __name__=='__main__': unittest.main()
