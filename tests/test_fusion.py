import unittest
from localization_contracts.apriltag import *
from localization_contracts.fusion import *
class SimulationS09(unittest.TestCase):
 def obs(self,n,t,x): return AprilTagObserver(TagConfig(calibration_version="c"),500,500,320,240).observe(Detection("c",n,0,((300+x,200),(400+x,200),(400+x,300),(300+x,300)),t,t+1,t+2))
 def test_async_motion_dedup_coasting_lost(self):
  f=PlanarFusion(); self.assertTrue(f.update(self.obs(1,0,0))); self.assertFalse(f.update(self.obs(1,0,0))); f.update(self.obs(2,100_000_000,10)); self.assertEqual(f.publish(150_000_000)["tracking_state"],"COASTING"); self.assertEqual(f.publish(400_000_000)["tracking_state"],"LOST")
 def test_out_of_order_rejected(self):
  f=PlanarFusion(); f.update(self.obs(2,100,0)); self.assertFalse(f.update(self.obs(1,50,0)))
 def test_same_timestamp_does_not_create_infinite_velocity(self):
  f=PlanarFusion(); f.update(self.obs(1,1_000_000_000,0)); f.update(self.obs(2,1_000_000_000,10))
  self.assertLess(abs(f.state.vx), 100.0)
 def test_rates_are_session_relative(self):
  f=PlanarFusion(); f.update(self.obs(1,10_000_000_000,0)); f.publish(10_100_000_000)
  self.assertGreater(f.publish(10_200_000_000)["output_hz"], 1.0)
 def test_innovation_gate_rejects_implausible_jump(self):
  f=PlanarFusion(gate_sigma=3.0); self.assertTrue(f.update(self.obs(1,0,0)))
  self.assertFalse(f.update(self.obs(2,100_000_000,1000))); self.assertEqual(f.rejected,1); self.assertEqual(f.measurements,1)
 def test_calibration_change_resets_session(self):
  f=PlanarFusion(); first=self.obs(1,0,0); f.update(first); old=f.session; changed=self.obs(2,100,0)
  from dataclasses import replace
  self.assertTrue(f.update(replace(changed,calibration_version='new-cal'))); self.assertGreater(f.session,old); self.assertEqual(f.measurements,1)
 def test_neighbor_camera_quality_hysteresis(self):
  s=ObservationSelector(.15); a=self.obs(1,0,0); b=self.obs(2,0,1)
  from dataclasses import replace
  a=replace(a,quality=.8,camera_id='camera_1'); b=replace(b,quality=.9,camera_id='camera_2'); self.assertEqual(s.select([a,b]).camera_id,'camera_2'); b2=replace(b,quality=.85); self.assertEqual(s.select([a,b2]).camera_id,'camera_2')
 def test_synchronous_arbiter_is_quality_not_thread_order(self):
  from dataclasses import replace
  base=self.obs(1,100,0)
  weak=replace(base,camera_id='camera_1',measurement_id='weak',quality=.3,
               pixel_features={**base.pixel_features,'reprojection_error_px':1.0,'side_px':40})
  strong=replace(base,camera_id='camera_2',measurement_id='strong',quality=.7,
                 pixel_features={**base.pixel_features,'reprojection_error_px':.3,'side_px':60})
  arb=SynchronousObservationArbiter(('camera_1','camera_2'))
  self.assertEqual(arb.report('camera_1',100,[weak]),[])
  ready=arb.report('camera_2',100,[strong])
  self.assertEqual(ready[0][1].measurement_id,'strong')
  self.assertEqual([x.measurement_id for x in ready[0][2]],['weak'])
 def test_synchronous_arbiter_advancing_camera_closes_skipped_frame(self):
  from dataclasses import replace
  base=self.obs(1,100,0); base=replace(base,camera_id='camera_1')
  arb=SynchronousObservationArbiter(('camera_1','camera_2'))
  arb.report('camera_1',100,[base]); first=arb.report('camera_2',90,[])
  ready=arb.report('camera_2',110,[])
  self.assertEqual(first[0][0],90)
  self.assertEqual(ready[0][0],100)
  self.assertEqual(ready[0][1].camera_id,'camera_1')
 def test_synchronous_arbiter_bounds_stalled_camera_and_drops_late_report(self):
  from dataclasses import replace
  arb=SynchronousObservationArbiter(('camera_1','camera_2'),max_pending=2)
  base=self.obs(1,100,0)
  self.assertEqual(arb.report('camera_1',100,[replace(base,camera_id='camera_1')]),[])
  arb.report('camera_1',200,[replace(base,camera_id='camera_1',capture_time_ns=200)])
  ready=arb.report('camera_1',300,[replace(base,camera_id='camera_1',capture_time_ns=300)])
  self.assertEqual(ready[0][0],100)
  self.assertEqual(arb.report('camera_2',100,[]),[])
  self.assertEqual(arb.late_reports,1)
 def test_common_track_status_contract(self):
  f=PlanarFusion(); f.update(self.obs(1,0,0)); s=f.status(10); self.assertEqual(s.tracking_state,'COASTING'); self.assertEqual(s.pose_frame,'arena'); self.assertEqual(s.calibration_version,'c')
if __name__=='__main__': unittest.main()
