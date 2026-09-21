import unittest
from localization_contracts.opponent import *
class SimulationS11(unittest.TestCase):
 def test_confirmation_rate_and_timeout(self):
  t=OpponentTracker(); self.assertTrue(t.update(Detection2D('c',1,0,(1,2,3,4),.8))); self.assertFalse(t.publish(0).confirmed); t.update(Detection2D('c',2,100000000,(2,2,4,4),.8)); self.assertTrue(t.publish(100000000).confirmed); self.assertTrue(t.publish(500000000).lost)
 def test_low_confidence_not_track(self): self.assertFalse(OpponentTracker().update(Detection2D('c',1,0,(0,0,1,1),.1)))
