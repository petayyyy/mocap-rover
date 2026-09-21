import unittest
from localization_contracts.registry import CameraRegistry, PreviewMetadata, SourceBinding

class SimulationS04(unittest.TestCase):
    def setUp(self): self.r = CameraRegistry.virtual_default()

    def test_six_unique_bindings_and_preview(self):
        self.assertEqual(set(self.r.bindings), {f"camera_{i}" for i in range(1, 7)})
        self.assertEqual(self.r.preview("camera_1", PreviewMetadata(1600, 1200))["width"], 1600)
        self.assertFalse(self.r.binding("camera_1").capabilities.hardware_verified)

    def test_runtime_id_permutation_and_invalid_duplicates(self):
        old = self.r.binding("camera_1")
        self.r.replace_source("camera_1", SourceBinding("replay-7", "/r/7", "r7", "m7", old.calibration_version))
        self.assertEqual(self.r.binding("camera_1").runtime_id, "replay-7")
        with self.assertRaises(ValueError):
            self.r.replace_source("camera_2", SourceBinding("replay-7", "/r/8", "r8", "m8", old.calibration_version))
        with self.assertRaises(ValueError):
            self.r.replace_source("camera_2", SourceBinding("replay-8", "/r/8", "r8", "m8", "other"))

    def test_bounded_queue_and_independent_channels(self):
        r = CameraRegistry.virtual_default(); r.queue_size = 2
        for i in range(3): r.push("camera_1", "image", i)
        self.assertEqual([r.pop("camera_1", "image"), r.pop("camera_1", "image")], [1, 2])
        r.set_channel("camera_1", "image", False)
        self.assertFalse(r.push("camera_1", "image", 9)); self.assertTrue(r.push("camera_1", "markers", 9))

    def test_disconnect_backoff_and_one_channel_failure(self):
        self.r.disconnect("camera_1", "image", 0)
        self.assertFalse(self.r.reconnect("camera_1", "image", 50_000_000))
        self.assertTrue(self.r.reconnect("camera_1", "image", 100_000_000))
        self.r.disconnect("camera_1", "markers", 0)
        self.assertTrue(self.r.push("camera_1", "image", "ok"))
        self.assertFalse(self.r.push("camera_1", "markers", "lost"))
        self.assertTrue(self.r.push("camera_2", "image", "other"))

if __name__ == "__main__": unittest.main()
