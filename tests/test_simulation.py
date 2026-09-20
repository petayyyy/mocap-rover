import json, tempfile, unittest
from pathlib import Path
from simulation.faults import FaultConfig, FaultInjector
from simulation.evaluator import evaluate

class SimulationS03(unittest.TestCase):
    def test_faults_are_seeded_and_preserve_capture_time(self):
        frames=[{"capture_time_ns":i*1000000,"sequence":i} for i in range(20)]
        cfg=FaultConfig(seed=7,drop_probability=.2,max_delay_ms=4,reorder_probability=.3,clock_offset_ms=2,clock_drift_ppm=10)
        a=FaultInjector(cfg).apply(frames); b=FaultInjector(cfg).apply(frames)
        self.assertEqual(a,b); self.assertTrue(all(x["receive_time_ns"] >= x["capture_time_ns"] for x in a))
        self.assertTrue(all(x["capture_time_ns"] % 1000000 == 0 for x in a))

    def test_evaluator_is_separate_from_runtime(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'e.json').write_text(json.dumps([{"stamp_ns":1,"position_m":[1,2,0]}]))
            (p/'t.json').write_text(json.dumps([{"stamp_ns":1,"position_m":[1.03,2,0]}]))
            result=evaluate(p/'e.json',p/'t.json')
            self.assertEqual(result["matched"],1); self.assertTrue(result["truth_used_only_here"])

if __name__ == '__main__': unittest.main()
