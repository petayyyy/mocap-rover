import unittest
from simulation.calibration_experiment import run

class SimulationS07CalibrationExperiment(unittest.TestCase):
 def test_image_derived_extrinsic_recovers_hidden_mount(self):
  result=run(); self.assertEqual(result['independent_points'],80); self.assertFalse(result['solver_reads_truth']); self.assertLess(result['translation_error_m'],.02); self.assertLess(result['held_out_reprojection_px'],5.); self.assertGreater(result['reprojection_quality'],.5)

if __name__=='__main__': unittest.main()
