import unittest
from localization_contracts.config import load_config
from localization_contracts.settings import *
class SimulationS12(unittest.TestCase):
 def test_atomic_apply_ack_and_hardware_restriction(self):
  b=SettingsBackend(load_config('config/contracts.json'),{'trigger':False}); b.stage({'capture':{'width':800}}); self.assertTrue(b.apply().ok); self.assertEqual(b.revision,1)
  with self.assertRaises(ConfigError): b.stage({'timing':{'trigger':True}})
  self.assertTrue(b.rollback().ok)
