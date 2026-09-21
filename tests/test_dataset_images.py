import tempfile
import unittest
from pathlib import Path
from simulation.dataset import build_image_dataset

class SimulationS10Images(unittest.TestCase):
 def test_reproducible_session_split_and_labels(self):
  with tempfile.TemporaryDirectory() as d:
   m=build_image_dataset(d,[{'seed':40,'trajectory':'straight'},{'seed':41,'trajectory':'circle'}],frames_per_session=2)
   self.assertEqual(len(m['records']),4); self.assertEqual({r['split'] for r in m['records']},{'test','val'})
   self.assertEqual(len({r['session'] for r in m['records']}),2)
   for r in m['records']:
    self.assertTrue((Path(d)/r['image']).is_file()); self.assertTrue((Path(d)/r['label']).read_text().startswith('0 '))
   self.assertIsNone(m['weights']); self.assertEqual(m['training_status'],'not_run')

if __name__=='__main__': unittest.main()
