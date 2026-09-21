import json, subprocess, sys, tempfile, unittest
from pathlib import Path
class SimulationAcceptanceRunner(unittest.TestCase):
 def test_manifest_contains_hash_seed_and_gate(self):
  with tempfile.NamedTemporaryFile(suffix='.json') as f:
   recording=f.name+'.replay.json'; subprocess.run([sys.executable,'scripts/run_acceptance.py','--seed','9','--output',f.name,'--recording',recording],check=True,stdout=subprocess.DEVNULL)
   with open(f.name,encoding='utf-8') as h: d=json.load(h)
   self.assertEqual(d['seed'],9); self.assertFalse(d['hardware_verified']); self.assertFalse(d['sim_accepted']); self.assertTrue(d['config_digest']); self.assertEqual(d['calibration_path'],'config/cameras.json'); self.assertEqual(len(d['calibration_digest']),64); self.assertEqual(d['world_path'],'worlds/mocap_arena.sdf'); self.assertEqual(len(d['world_digest']),64); self.assertTrue(d['git_revision'])
   with open(recording,encoding='utf-8') as h: replay=json.load(h)
   self.assertGreater(d['recording_frames'],0); self.assertTrue(replay['header']['config_hash'])
   self.assertEqual(d['scenario_manifest'],str(Path(f.name).with_suffix('.episodes.json'))); self.assertEqual(len(d['scenario_manifest_digest']),64)
   self.assertEqual([x['seed'] for x in d['seed_sweep']],[42,7,123]); self.assertTrue(all(x['opponent_hz']>=10 for x in d['seed_sweep']))
