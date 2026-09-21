import json, subprocess, sys, tempfile, unittest
class SimulationAcceptanceRunner(unittest.TestCase):
 def test_manifest_contains_hash_seed_and_gate(self):
  with tempfile.NamedTemporaryFile(suffix='.json') as f:
   recording=f.name+'.replay.json'; subprocess.run([sys.executable,'scripts/run_acceptance.py','--seed','9','--output',f.name,'--recording',recording],check=True,stdout=subprocess.DEVNULL)
   with open(f.name,encoding='utf-8') as h: d=json.load(h)
   self.assertEqual(d['seed'],9); self.assertFalse(d['hardware_verified']); self.assertFalse(d['sim_accepted']); self.assertTrue(d['config_digest'])
   with open(recording,encoding='utf-8') as h: replay=json.load(h)
   self.assertGreater(d['recording_frames'],0); self.assertTrue(replay['header']['config_hash'])
