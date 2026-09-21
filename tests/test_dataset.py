import unittest
from simulation.dataset import *
class SimulationS10(unittest.TestCase):
 def test_deterministic_session_split_and_truth_boundary(self):
  m=make_manifest([{'session':'s2','frame':2},{'session':'s1','frame':1}]); self.assertEqual(m['splits'][0]['split'],'test'); self.assertEqual(m['truth_role'],'labels_and_evaluation_only'); self.assertEqual(digest(m),digest(make_manifest([{'session':'s1','frame':1},{'session':'s2','frame':2}])))
if __name__=='__main__': unittest.main()
