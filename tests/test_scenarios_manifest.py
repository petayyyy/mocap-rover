import unittest
from simulation.scenarios import manifest
class SimulationScenarioManifest(unittest.TestCase):
 def test_independent_trajectory_style_and_label_manifest(self):
  m=manifest((42,7),seconds=1); self.assertEqual(len(m['episodes']),8); self.assertEqual(len(m['episodes'][0]['points']),15); self.assertEqual({e['name'] for e in m['episodes']},{'rest','straight','circle','eight'}); self.assertEqual(m['truth_role'],'labels_and_evaluation_only'); self.assertTrue(all(e['style'] in {'steel','colored','dark','bright','striped'} for e in m['episodes']))
