import unittest
from localization_contracts.config import load_config
from localization_contracts.settings import *
class SimulationS12(unittest.TestCase):
 def test_atomic_apply_ack_and_hardware_restriction(self):
  b=SettingsBackend(load_config('config/contracts.json'),{'trigger':False}); b.stage({'capture':{'width':800}}); self.assertTrue(b.apply().ok); self.assertEqual(b.revision,1)
  with self.assertRaises(ConfigError): b.stage({'timing':{'trigger':True}})
  self.assertTrue(b.rollback().ok)
 def test_stale_revision_returns_error_ack(self):
  b=SettingsBackend(load_config('config/contracts.json')); b.stage({'capture':{'width':900}}); self.assertTrue(b.apply(expected_revision=0).ok); b.stage({'capture':{'width':901}}); self.assertFalse(b.apply(expected_revision=0).ok); self.assertEqual(b.last_ack.error,'stale config revision')
 def test_runtime_binding_receives_apply_and_rollback(self):
  b=SettingsBackend(load_config('config/contracts.json')); seen=[]; b.bind_runtime(lambda c:seen.append(('apply',c['capture']['width'])),lambda c:seen.append(('rollback',c['capture']['width'])))
  b.stage({'capture':{'width':901}}); self.assertTrue(b.apply().ok); self.assertEqual(seen[-1],('apply',901)); self.assertTrue(b.rollback().ok); self.assertEqual(seen[-1][0],'rollback')
