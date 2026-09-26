import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localization_contracts.camera_model import CameraModel, infer_model

ROOT = Path(__file__).resolve().parents[1]
WORLDS = sorted(ROOT.glob("worlds/*.sdf"))


def world_cameras(path):
    root = ET.parse(path).getroot()
    for sensor in root.iter("sensor"):
        if sensor.get("type") != "camera":
            continue
        camera = sensor.find("camera")
        if camera is None:
            continue
        image = camera.find("image")
        intrinsics = camera.find("lens/intrinsics")
        if intrinsics is None or image is None:
            continue
        distortion = camera.find("distortion")
        d = [0.0] * 5
        if distortion is not None:
            d = [float(distortion.findtext(tag))
                 for tag in ("k1", "k2", "p1", "p2", "k3")]
        K = [[float(intrinsics.findtext("fx")), 0.0, float(intrinsics.findtext("cx"))],
             [0.0, float(intrinsics.findtext("fy")), float(intrinsics.findtext("cy"))],
             [0.0, 0.0, 1.0]]
        yield sensor.get("name", "?"), K, d, [int(image.findtext("width")),
                                              int(image.findtext("height"))]


class WorldCameraModels(unittest.TestCase):
    def test_worlds_exist(self):
        self.assertTrue(WORLDS)

    def test_every_rendered_camera_has_a_usable_projection_model(self):
        for world in WORLDS:
            for name, K, d, size in world_cameras(world):
                with self.subTest(world=world.name, camera=name):
                    report = CameraModel(K, d, size, name=name).validate()
                    self.assertTrue(
                        report.ok,
                        f"{world.name}/{name}: {'; '.join(report.reasons)}",
                    )

    def test_scale_to_hfov_never_overrides_the_declared_focal_length(self):
        # scale_to_hfov rescales the rendered image, so the effective focal
        # length stops matching <intrinsics> and every PnP range is biased.
        for world in WORLDS:
            root = ET.parse(world).getroot()
            for lens in root.iter("lens"):
                scale = lens.find("scale_to_hfov")
                if scale is None or lens.find("intrinsics") is None:
                    continue
                with self.subTest(world=world.name):
                    self.assertEqual((scale.text or "").strip().lower(), "false")


if __name__ == "__main__":
    unittest.main()


class WorldRates(unittest.TestCase):
    """set_world_rates has to reach every limit that caps a high-speed run."""

    def patched(self, **kwargs):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "set_world_rates", ROOT / "scripts" / "set_world_rates.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        options = dict(camera_hz=90.0, truth_hz=200.0, max_linear=12.0,
                       max_angular=3.0, lidar_hz=None, step_size=None)
        options.update(kwargs)
        return module.patch(ROOT / "worlds" / "mocap_arena_l2.sdf", **options)

    def test_the_shipped_world_caps_the_rover_below_walking_pace_of_40kmh(self):
        root = ET.parse(ROOT / "worlds" / "mocap_arena_l2.sdf").getroot()
        limits = [float(p.findtext("max_linear_velocity"))
                  for p in root.iter("plugin")
                  if (p.get("name") or "").endswith("DiffDrive")]
        self.assertTrue(limits)
        # 11.11 m/s is 40 km/h; the world ships at 1.
        self.assertTrue(all(v < 11.11 for v in limits),
                        "world already allows 40 km/h; update this test")

    def test_patching_lifts_every_cap_that_matters(self):
        tree, changes = self.patched()
        root = tree.getroot()
        for plugin in root.iter("plugin"):
            name = plugin.get("name") or ""
            if name.endswith("DiffDrive"):
                self.assertGreaterEqual(float(plugin.findtext("max_linear_velocity")), 11.11)
                self.assertLessEqual(float(plugin.findtext("min_linear_velocity")), -11.11)
            elif name.endswith("PosePublisher"):
                self.assertGreaterEqual(float(plugin.findtext("update_frequency")), 200)
        for sensor in root.iter("sensor"):
            if sensor.get("type") == "camera":
                self.assertEqual(float(sensor.findtext("update_rate")), 90.0)
            if sensor.get("type") == "gpu_lidar":
                # Left alone unless asked: the L2 really does run at 5.55 Hz.
                self.assertAlmostEqual(float(sensor.findtext("update_rate")), 5.55)
        self.assertTrue(changes)

    def test_patching_is_idempotent(self):
        tree, _ = self.patched()
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as handle:
            tree.write(handle.name, encoding="utf-8", xml_declaration=True)
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "set_world_rates", ROOT / "scripts" / "set_world_rates.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _, again = module.patch(
                Path(handle.name), camera_hz=90.0, truth_hz=200.0,
                max_linear=12.0, max_angular=3.0, lidar_hz=None, step_size=None)
        self.assertEqual(again, [])
