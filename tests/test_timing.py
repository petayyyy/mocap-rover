import tempfile, unittest
from pathlib import Path
from localization_contracts.timing import *

class SimulationS05(unittest.TestCase):
    def sample(self, n=1, receive=110):
        return TimingMetadata(n, 100, receive, 120, "sim", 1_000_000, 2_000_000).validate()
    def test_timestamps_clock_quality_and_skew(self):
        self.assertEqual(ClockModel(10, 10).correct(1_000_000), 999_980)
        self.assertTrue(assess_quality([1_000_000, 2_000_000]).synchronized)
        self.assertFalse(assess_quality([0, 20_000_000]).synchronized)
        with self.assertRaises(ValueError): TimingMetadata(1, 100, 110, 120, "sim", 1, 2, "bad").validate()
    def test_replay_roundtrip_and_reset_session(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.json"; log = ReplayLog([self.sample()], "c", "k", "m"); log.dump(p)
            loaded = ReplayLog.load(p)
            self.assertEqual(loaded.frames[0].capture_time_ns, 100)
            self.assertTrue(loaded.verify_headers('c','k','m'))
            with self.assertRaises(ValueError): loaded.verify_headers('wrong','k','m')
            self.assertEqual(ReplayScheduler(loaded).schedule(reset=True)[0][0], "new_session")

if __name__ == "__main__": unittest.main()
