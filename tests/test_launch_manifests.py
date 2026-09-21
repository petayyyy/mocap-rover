import json, unittest
from pathlib import Path
class SimulationLaunchManifests(unittest.TestCase):
 def test_sim_and_replay_are_virtual_and_share_registry(self):
  root=Path(__file__).parents[1]; sim=json.loads((root/'launch/simulation.launch.json').read_text()); replay=json.loads((root/'launch/replay.launch.json').read_text()); self.assertFalse(sim['physical_devices']); self.assertFalse(replay['hardware_verified']); self.assertEqual(sim['registry'],replay['registry']); self.assertTrue(replay['seek_creates_new_session'])
