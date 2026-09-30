import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def generate(out, *extra):
    subprocess.run([sys.executable, str(ROOT / "scripts/generate_world.py"), "--profile", "imx219_160",
                    "--lidar", "airy", "--ideal-cameras", "--world-name", "w", "--output-dir", str(out),
                    *extra], check=True, capture_output=True)
    return (json.loads((out / "config/w/runtime_cameras.json").read_text()),
            json.loads((out / "config/w/scenario.json").read_text()),
            (out / "worlds/w.sdf").read_text())


class Mounts5LayoutTest(unittest.TestCase):
    def test_cameras_sit_on_the_five_mounts_looking_down(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, scenario, _ = generate(Path(tmp), "--layout", "mounts5", "--centre-spacing", "0.4")
        positions = [c["position_world"][:2] for c in config["cameras"]]
        self.assertEqual(positions, [[3, 3], [9, 3], [3, 9], [9, 9], [5.8, 6], [6.2, 6]])
        for cam in config["cameras"]:
            self.assertAlmostEqual(cam["R_world_optical"][2][2], -1.0)   # optical axis straight down
        rolls = [round(c["layout"]["roll_deg"]) for c in config["cameras"]]
        self.assertEqual(rolls, [0, 0, 0, 0, 0, 90])
        self.assertEqual(scenario["centre_spacing_m"], 0.4)

    def test_corner_tilt_leans_toward_the_arena_centre(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = generate(Path(tmp), "--layout", "mounts5", "--layout-variant", "corner_tilt_deg=10")
        for cam in config["cameras"][:4]:
            axis = [row[2] for row in cam["R_world_optical"]]
            x, y, _ = cam["position_world"]
            self.assertAlmostEqual(math.degrees(math.acos(-axis[2])), 10, places=6)
            self.assertGreater(axis[0] * (6 - x) + axis[1] * (6 - y), 0)

    def test_centre_tilt_out_leans_the_pair_apart(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = generate(Path(tmp), "--layout", "mounts5", "--layout-variant", "centre=tilt_out_15")
        left, right = config["cameras"][4:]
        self.assertLess(left["R_world_optical"][0][2], 0)
        self.assertGreater(right["R_world_optical"][0][2], 0)

    def test_lidar_mesh_is_resolved_through_the_resource_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, scenario, world = generate(Path(tmp), "--layout", "final")
        self.assertIn("<uri>model://unitree_l2/meshes/unitree_l2_base.obj</uri>", world)
        self.assertNotIn("file://", world)
        fractions = scenario["coverage_fractions"]
        self.assertEqual([f["height_m"] for f in fractions], [0.3654, 0.483])
        self.assertGreaterEqual(fractions[0]["at_least_1"], fractions[0]["at_least_2"])

    def test_final_layout_reproduces_the_shipped_imx219_world(self):
        """final + set_world_rates gives worlds/mocap_arena_imx219.sdf except the two mesh URIs."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            subprocess.run([sys.executable, str(ROOT / "scripts/generate_world.py"), "--profile", "imx219_160",
                            "--layout", "final", "--lidar", "airy", "--ideal-cameras",
                            "--world-name", "mocap_arena_imx219", "--output-dir", str(out)],
                           check=True, capture_output=True)
            world = out / "worlds/mocap_arena_imx219.sdf"
            subprocess.run([sys.executable, str(ROOT / "scripts/set_world_rates.py"), str(world), "--camera-hz", "80",
                            "--truth-hz", "200", "--lidar-hz", "10", "--max-linear", "12"],
                           check=True, capture_output=True)
            new = world.read_text().splitlines()
            runtime = json.loads((out / "config/mocap_arena_imx219/runtime_cameras.json").read_text())
        old = (ROOT / "worlds/mocap_arena_imx219.sdf").read_text().splitlines()
        self.assertEqual(len(new), len(old))
        differing = [(a.strip(), b.strip()) for a, b in zip(new, old) if a != b]
        self.assertEqual([a for a, _ in differing],
                         ["<uri>model://unitree_l2/meshes/unitree_l2_base.obj</uri>",
                          "<uri>model://unitree_l2/meshes/unitree_l2_rotor.obj</uri>"])
        self.assertTrue(all(b.endswith(a.replace("<uri>model://", "/models/")) for a, b in differing))
        shipped = json.loads((ROOT / "config/mocap_arena_imx219/runtime_cameras.json").read_text())
        self.assertEqual(runtime, shipped)

    def test_variant_needs_mounts5(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(subprocess.CalledProcessError):
                generate(Path(tmp), "--layout", "final", "--layout-variant", "centre=xx")


if __name__ == "__main__":
    unittest.main()
