import unittest
from localization_contracts.opponent import *

class SimulationS11(unittest.TestCase):
 def test_confirmation_rate_and_timeout(self):
  t=OpponentTracker(); self.assertTrue(t.update(Detection2D('c',1,0,(1,2,3,4),.8))); self.assertFalse(t.publish(0).confirmed); t.update(Detection2D('c',2,100000000,(2,2,4,4),.8)); self.assertTrue(t.publish(100000000).confirmed); self.assertTrue(t.publish(500000000).lost)
 def test_low_confidence_not_track(self): self.assertFalse(OpponentTracker().update(Detection2D('c',1,0,(0,0,1,1),.1)))
 def test_multicamera_single_selection_and_switch_count(self):
  a=MultiCameraAssociator(); d1=Detection2D('c1',1,100,(0,0,10,10),.7); d2=Detection2D('c2',1,100,(1,0,11,10),.9)
  self.assertEqual(a.select([d1,d2]).camera_id,'c2'); self.assertEqual(a.select([Detection2D('c1',2,150,(1,0,11,10),.8)]).camera_id,'c1'); self.assertEqual(a.id_switches,0)
 def test_height_aware_contact_projection(self):
  d=Detection2D('c',1,0,(40,40,60,60),.9); p=project_contact_point(d,100,100,50,50,(0,0,2.0),((1,0,0),(0,1,0),(0,0,-1)),height_uncertainty_m=.1); self.assertAlmostEqual(p['position_m'][0],0); self.assertAlmostEqual(p['position_m'][2],0); self.assertEqual(p['height_uncertainty_m'],.1)
 def test_metric_tracker_survives_camera_handoff_without_pixel_state(self):
  t=OpponentTracker(); self.assertTrue(t.update(Detection2D('c1',1,0,(0,0,1,1),.9,position_m=(2.,3.))))
  self.assertTrue(t.update(Detection2D('c2',2,100000000,(0,0,1,1),.9,position_m=(2.1,3.))))
  self.assertAlmostEqual(t.publish(100000000).x,2.1); self.assertEqual(t.id_switches,0)
 def test_metric_association_does_not_compare_pixels_between_cameras(self):
  a=MultiCameraAssociator(max_metric_jump_m=.5)
  self.assertIsNotNone(a.select([Detection2D('c1',1,0,(0,0,10,10),.8,position_m=(1,1))]))
  self.assertIsNotNone(a.select([Detection2D('c2',2,50000000,(900,900,910,910),.8,position_m=(1.1,1))]))

if __name__=='__main__': unittest.main()
