import unittest
from simulation.scenarios import manifest
class SimulationScenarioManifest(unittest.TestCase):
 def test_independent_trajectory_style_and_label_manifest(self):
  m=manifest((42,7),seconds=1); self.assertEqual(len(m['episodes']),40); self.assertEqual(len(m['episodes'][0]['points']),15); self.assertEqual({e['name'] for e in m['episodes']},{'rest','straight','circle','eight'}); self.assertEqual({e['style'] for e in m['episodes']},{'steel','colored','dark','bright','striped'}); self.assertEqual(m['truth_role'],'labels_and_evaluation_only')
