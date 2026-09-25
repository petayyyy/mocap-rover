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
