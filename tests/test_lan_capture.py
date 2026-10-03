"""The LAN camera node and the laptop receiver, end to end without hardware.

A real ``CameraNode`` runs in a thread with the synthetic sensor, whose pixel
(r, c) of frame ``seq`` is ``(7r + 3c + seq) & 0xFF``, so every window that
arrives can be checked against the coordinates and frame its header claims.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from localization_contracts import lan_capture  # noqa: E402
from pi_cam import flash_analysis, lan_protocol as proto  # noqa: E402
from pi_cam.camera_node import (  # noqa: E402
    CameraNode, Frame, NodeConfig, PtpMonitor, SyntheticSensor,
    realtime_minus_boottime_ns, stamp_reference_correction_ns)

WIDTH, HEIGHT = 328, 246


def start_node(**overrides):
    cfg = NodeConfig(camera_id=overrides.pop("camera_id", "cam_t"), port=overrides.pop("port", 0),
                     sensor="synthetic",
                     synthetic_width=WIDTH, synthetic_height=HEIGHT, synthetic_fps=100.0,
                     status_period_s=0.2, ptp_enabled=False, **overrides)
    node = CameraNode(cfg)
    node.start()
    return node


def expected_window(frame):
    rows = np.arange(frame.row0, frame.row0 + frame.height)[:, None]
    cols = np.arange(frame.col0, frame.col0 + frame.width)[None, :]
    return ((7 * rows + 3 * cols + frame.frame_seq) & 0xFF).astype(np.uint8)


class Protocol(unittest.TestCase):
    def test_frame_header_roundtrip_and_size(self):
        header = proto.FrameHeader("camera_3", 12345, 1_700_000_000_123_456_789, 800_000, 9452,
                                   12_048_192, 100, 200, 480, 480, proto.FORMAT_JPEG, 1, 2,
                                   1640, 1232, 7, 1_700_000_000_130_000_000, 555, -42, None)
        raw = header.pack()
        self.assertEqual(len(raw), proto.FRAME_HEADER_SIZE)
        back = proto.FrameHeader.unpack(raw)
        self.assertEqual(back, header)
        self.assertIsNone(back.ptp_offset_ns)
        with_ptp = proto.FrameHeader.unpack(
            proto.FrameHeader(**{**header.__dict__, "ptp_offset_ns": -12345}).pack())
        self.assertEqual(with_ptp.ptp_offset_ns, -12345)

    def test_stream_parser_reassembles_split_messages(self):
        header = proto.FrameHeader("c", 1, 2, 3, 4, 5, 0, 0, 4, 2, proto.FORMAT_Y8, 0, 1,
                                   4, 2, 0, 0)
        payload = bytes(range(8))
        stream = b"".join(proto.encode_frame(header, payload)) + proto.encode_json(
            proto.MSG_STATUS, {"sensor_fps": 83.0}) + b"".join(proto.encode_frame(header, payload))
        parser = proto.StreamParser()
        messages = []
        for i in range(0, len(stream), 7):        # feed in awkward slices
            messages.extend(parser.feed(stream[i:i + 7]))
        self.assertEqual([m.msg_type for m in messages],
                         [proto.MSG_FRAME, proto.MSG_STATUS, proto.MSG_FRAME])
        self.assertEqual(messages[0].frame, header)
        self.assertEqual(bytes(messages[0].data), payload)
        self.assertEqual(messages[1].meta, {"sensor_fps": 83.0})

    def test_frame_parse_costs_microseconds(self):
        header = proto.FrameHeader("camera_1", 1, 2, 3, 4, 5, 0, 0, 480, 480, proto.FORMAT_Y8,
                                   0, 2, 1640, 1232, 1, 2)
        payload = bytes(480 * 480)
        message = b"".join(proto.encode_frame(header, payload))
        parser = proto.StreamParser()
        begin = time.perf_counter()
        n = 500
        for _ in range(n):
            (msg,) = parser.feed(message)
            msg.frame
        per_message_us = (time.perf_counter() - begin) / n * 1e6
        # 500 messages per second must cost a negligible share of the second.
        self.assertLess(per_message_us, 500.0)

    def test_bad_magic_is_rejected(self):
        with self.assertRaises(proto.ProtocolError):
            proto.StreamParser().feed(b"\0" * 32)

    def test_window_normalization_clamps_and_aligns(self):
        w = proto.normalize_window((-5, 1500, 480, 480, "jpeg"), 1640, 1232)
        self.assertEqual(w, {"row0": 0, "col0": 1160, "w": 480, "h": 480, "format": "jpeg"})
        w = proto.normalize_window({"row0": 101, "col0": 7, "w": 2000, "h": 10}, 1640, 1232)
        self.assertEqual((w["row0"], w["col0"], w["w"]), (100, 0, 1640))
        w = proto.normalize_window({"row0": 101, "col0": 7, "w": 20, "h": 10}, 1640, 1232)
        self.assertEqual((w["row0"], w["col0"], w["w"]), (100, 6, 20))
        with self.assertRaises(proto.ProtocolError):
            proto.normalize_window((0, 0, 10, 10, "png"), 1640, 1232)


class NodeAndSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.node = start_node()
        cls.source = lan_capture.LanCameraSource([("127.0.0.1", cls.node.port)])
        assert cls.source.wait_connected(5.0) == ["cam_t"]

    @classmethod
    def tearDownClass(cls):
        cls.source.close()
        cls.node.stop()

    def drain(self):
        """Throw away the backlog an earlier test may have built up."""
        while self.source.take("cam_t", 0.02) is not None:
            pass

    def take_group(self, predicate=lambda g: True, tries=120):
        for _ in range(tries):
            group = self.source.take("cam_t", 0.5)
            if group is not None and predicate(group):
                return group
        self.fail("no matching frame group arrived")

    def test_hello_carries_identity_from_config_not_address(self):
        hello = self.source.hello("cam_t")
        self.assertEqual(hello["camera_id"], "cam_t")
        self.assertEqual((hello["sensor_width"], hello["sensor_height"]), (WIDTH, HEIGHT))
        self.assertEqual(hello["line_time_ns"], 9452)

    def test_windows_of_one_frame_share_the_stamp_and_match_pixels(self):
        ack = self.source.request_windows("cam_t", [(10, 20, 64, 48), (100, 200, 64, 48, "jpeg")])
        self.assertTrue(ack["ok"], ack)
        group = self.take_group(lambda g: len(g) == 2 and g[0].request_id == ack["request_id"])
        a, b = group
        self.assertEqual(a.stamp_ns, b.stamp_ns)
        self.assertEqual(a.frame_seq, b.frame_seq)
        self.assertEqual((a.row0, a.col0, a.width, a.height, a.format), (10, 20, 64, 48, "y8"))
        self.assertEqual((b.row0, b.col0, b.width, b.height, b.format), (100, 200, 64, 48, "jpeg"))
        np.testing.assert_array_equal(a.array, expected_window(a))
        # JPEG q90 of a smooth ramp: close, not exact.
        self.assertLess(np.abs(b.array.astype(int) - expected_window(b).astype(int)).mean(), 6.0)
        camera_id, array, stamp, row0, col0, line_time, exposure, receive = a
        self.assertEqual((camera_id, row0, col0, line_time, exposure), ("cam_t", 10, 20, 9452, 800_000))
        self.assertIs(array, a.array)
        self.assertGreater(receive, stamp)
        self.assertEqual(a.row_stamp_ns(5, exposure_centre=False), stamp + 15 * 9452)

    def test_stamp_is_exposure_start_on_the_realtime_scale(self):
        self.source.request_windows("cam_t", [(0, 0, 8, 8)])
        group = self.take_group(lambda g: len(g) == 1 and g[0].width == 8)
        frame = group[0]
        # readout_start reference: stamp = SensorTimestamp - exposure, moved to REALTIME.
        expected = frame.sensor_stamp_ns + frame.clock_offset_ns - frame.exposure_ns
        self.assertEqual(frame.stamp_ns, expected)
        self.assertLess(abs(frame.clock_offset_ns - realtime_minus_boottime_ns()), 5_000_000)
        latency_ms = (frame.receive_ns - frame.stamp_ns) / 1e6
        self.assertGreater(latency_ms, 0.0)
        self.assertLess(latency_ms, 200.0)

    def test_full_frame_on_request_has_sensor_size(self):
        self.source.request_windows("cam_t", [(0, 0, 8, 8)])
        self.drain()
        ack = self.source.request_full("cam_t")
        group = self.take_group(lambda g: any(f.is_full for f in g))
        full = [f for f in group if f.is_full][0]
        self.assertEqual(full.array.shape, (HEIGHT, WIDTH))
        self.assertEqual(full.request_id, ack["request_id"])
        # The full frame and the window are the same sensor frame.
        window = [f for f in group if not f.is_full][0]
        self.assertEqual(window.stamp_ns, full.stamp_ns)
        np.testing.assert_array_equal(full.array, expected_window(full))
        np.testing.assert_array_equal(window.array, full.array[:8, :8])

    def test_slow_consumer_keeps_the_latest_and_counts_drops(self):
        self.source.request_windows("cam_t", [(0, 0, 16, 16)])
        self.take_group(lambda g: g[0].width == 16)
        before = self.source.stats()
        time.sleep(0.5)                       # ~50 frames arrive, nobody takes them
        group = self.take_group(lambda g: g[0].width == 16)
        after = self.source.stats()
        self.assertGreater(after["dropped"]["cam_t"] - before["dropped"]["cam_t"], 20)
        self.assertGreater(after["received"]["cam_t"] - before["received"]["cam_t"], 30)
        # What we got is the newest frame, not a stale one.
        newer = self.take_group(lambda g: g[0].width == 16)
        self.assertLess(newer[0].frame_seq - group[0].frame_seq, 10)

    def test_status_reports_sensor_rate_and_counters(self):
        deadline = time.monotonic() + 3.0
        status = None
        while time.monotonic() < deadline:
            status = self.source.status("cam_t")
            if status and status["frames_captured"] > 10:
                break
            time.sleep(0.05)
        self.assertIsNotNone(status)
        self.assertAlmostEqual(status["sensor_fps"], 100.0, delta=1.0)
        for key in ("frames_missed", "frames_dropped_queue", "capture_to_send_ms", "ptp",
                    "clock_offset_ns", "line_time_ns", "exposure_ns", "throttled", "soc_temp_c"):
            self.assertIn(key, status)
        self.assertEqual(status["ptp"]["state"], "disabled")
        ack = self.source.request_status("cam_t")
        self.assertTrue(ack["ok"])

    def test_configure_changes_exposure_in_the_headers(self):
        ack = self.source.configure("cam_t", exposure_us=500)
        self.assertTrue(ack["ok"], ack)
        self.source.request_windows("cam_t", [(0, 0, 4, 4)])
        group = self.take_group(lambda g: g[0].width == 4 and g[0].exposure_ns == 500_000)
        self.assertEqual(group[0].exposure_ns, 500_000)
        self.source.configure("cam_t", exposure_us=800)

    def test_unknown_command_is_refused_not_fatal(self):
        link = self.source._link("cam_t")
        answer = link.send_command({"cmd": "explode"})
        self.assertFalse(answer["ok"])
        self.assertTrue(self.source.request_status("cam_t")["ok"])


class HardwareRegressions(unittest.TestCase):
    """Defects found on a live CM5; each would pass silently in isolation."""

    def test_wait_connected_waits_for_the_hello_not_the_first_frame(self):
        # A node whose windows are still set from an earlier client starts
        # sending frames the moment the next client connects.  A consumer that
        # returned on the first frame found hello() empty and crashed
        # (record_lan_dataset did, on hardware).
        node = start_node(camera_id="cam_h")
        first = lan_capture.LanCameraSource([("127.0.0.1", node.port)])
        try:
            first.wait_connected(5.0)
            first.request_windows("cam_h", [(0, 0, 32, 32)])
            self.assertIsNotNone(first.take("cam_h", 1.0))
            first.close()                      # windows stay set on the node
            second = lan_capture.LanCameraSource([("127.0.0.1", node.port)])
            try:
                self.assertEqual(second.wait_connected(5.0), ["cam_h"])
                hello = second.hello("cam_h")
                self.assertIsNotNone(hello, "wait_connected returned before the hello")
                self.assertEqual(hello["camera_id"], "cam_h")
                self.assertIn("cam_h", second.announced)
            finally:
                second.close()
        finally:
            node.stop()

    def test_frame_release_drops_the_buffer_view_before_the_camera_stops(self):
        # picamera2's DMA allocator refuses to close while a numpy array still
        # exports the mapped buffer: "cannot close exported pointers exist".
        closed = []
        array = np.zeros((4, 4), dtype=np.uint8)
        frame = Frame(array, 1, 2, 3, 1.0, 0, release=lambda: closed.append(True))
        frame.release()
        self.assertIsNone(frame.y, "the buffer view outlives release()")
        self.assertEqual(closed, [True])
        frame.release()                        # idempotent
        self.assertEqual(closed, [True])

    def test_payload_bytes_is_the_wire_size_not_the_decoded_size(self):
        node = start_node(camera_id="cam_b")
        source = lan_capture.LanCameraSource([("127.0.0.1", node.port)])
        try:
            source.wait_connected(5.0)
            source.request_windows("cam_b", [(0, 0, 64, 64), (0, 0, 64, 64, "jpeg")])
            for _ in range(60):
                group = source.take("cam_b", 0.5)
                if group and len(group) == 2:
                    break
            else:
                self.fail("no two-window group arrived")
            y8 = [f for f in group if f.format == "y8"][0]
            jpeg = [f for f in group if f.format == "jpeg"][0]
            self.assertEqual(y8.payload_bytes, 64 * 64)
            self.assertGreater(jpeg.payload_bytes, 0)
            self.assertLess(jpeg.payload_bytes, jpeg.array.size,
                            "a JPEG window must count its compressed size")
        finally:
            source.close()
            node.stop()

    def test_ptp_command_uses_an_absolute_pmc_and_optional_sudo(self):
        # pmc lives in /usr/sbin, absent from a service PATH, and it cannot bind
        # its reply socket as an unprivileged user.
        cfg = NodeConfig()
        self.assertTrue(cfg.ptp_pmc.startswith("/"), cfg.ptp_pmc)
        self.assertEqual(PtpMonitor(cfg).command()[:2], ["sudo", "-n"])
        plain = PtpMonitor(NodeConfig(ptp_pmc_sudo=False)).command()
        self.assertEqual(plain[0], cfg.ptp_pmc)
        self.assertIn("GET PORT_DATA_SET", plain)

    def test_ptp_monitor_reports_unavailable_when_pmc_says_nothing(self):
        monitor = PtpMonitor(NodeConfig(ptp_pmc="/bin/true", ptp_pmc_sudo=False))
        monitor.poll_once()
        snapshot = monitor.snapshot()
        self.assertEqual(snapshot["state"], "unavailable")
        self.assertIsNone(snapshot["offset_ns"])

    def test_ptp_monitor_parses_a_pmc_answer(self):
        monitor = PtpMonitor(NodeConfig(ptp_pmc="/bin/echo", ptp_pmc_sudo=False))
        monitor.command = lambda: ["/bin/printf", "%s\n",
                                   "offsetFromMaster -1234.0", "portState SLAVE"]
        monitor.poll_once()
        snapshot = monitor.snapshot()
        self.assertEqual(snapshot["offset_ns"], -1234)
        self.assertEqual(snapshot["state"], "SLAVE")


class NodeSurvivesClients(unittest.TestCase):
    def test_capture_continues_without_a_client_and_after_reconnect(self):
        node = start_node(camera_id="cam_r")
        try:
            time.sleep(0.3)
            captured_alone = node.frames_captured
            self.assertGreater(captured_alone, 10)
            source = lan_capture.LanCameraSource([("127.0.0.1", node.port)])
            source.wait_connected(5.0)
            source.request_windows("cam_r", [(0, 0, 8, 8)])
            self.assertIsNotNone(source.take("cam_r", 1.0))
            source.close()
            time.sleep(0.3)
            self.assertGreater(node.frames_captured, captured_alone + 20)
            self.assertIsNone(node.client)
            again = lan_capture.LanCameraSource([("127.0.0.1", node.port)])
            self.assertEqual(again.wait_connected(5.0), ["cam_r"])
            again.request_windows("cam_r", [(0, 0, 8, 8)])
            self.assertIsNotNone(again.take("cam_r", 1.0))
            again.close()
        finally:
            node.stop()

    def test_source_reconnects_when_the_node_comes_back(self):
        node = start_node(camera_id="cam_q")
        port = node.port
        source = lan_capture.LanCameraSource([("127.0.0.1", port)])
        try:
            self.assertEqual(source.wait_connected(5.0), ["cam_q"])
            source.request_windows("cam_q", [(0, 0, 8, 8)])
            node.stop()
            time.sleep(0.5)
            while source.take("cam_q", 0.05) is not None:
                pass                                   # drain what arrived before the stop
            restarted = time.monotonic_ns()
            node = start_node(camera_id="cam_q", port=port)
            deadline = time.monotonic() + 8.0
            group = None
            while time.monotonic() < deadline:
                group = source.take("cam_q", 0.5)
                if group is not None and group[0].receive_mono_ns > restarted:
                    break
                group = None
            self.assertIsNotNone(group, "no frames after the node restarted")
            self.assertEqual(group[0].width, 8)       # the window list was restored
            self.assertGreaterEqual(source.stats()["links"][f"127.0.0.1:{port}"]["reconnects"], 1)
        finally:
            source.close()
            node.stop()


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is needed to write FFV1")
class RecordLanDataset(unittest.TestCase):
    def test_recording_has_the_gazebo_dataset_format(self):
        import record_lan_dataset
        from replay_dataset import FrameSource, read_jsonl
        node = start_node(camera_id="camera_1")
        tmp = tempfile.TemporaryDirectory()
        try:
            root = Path(tmp.name)
            config = {"role": "image_calibrated", "calibration_version": "test",
                      "cameras": [{"name": "camera_1", "image_size": [WIDTH, HEIGHT]},
                                  {"name": "camera_2", "image_size": [WIDTH, HEIGHT]}]}
            (root / "runtime_cameras.json").write_text(json.dumps(config))
            out = root / "dataset"
            meta = record_lan_dataset.main([
                "--nodes", f"127.0.0.1:{node.port}", "--config", str(root / "runtime_cameras.json"),
                "--output", str(out), "--seconds", "1.5", "--divisor", "5", "--allow-missing"])
            self.assertEqual(set(meta["cameras"]), {"camera_1"})
            for name in ("meta.json", "runtime_cameras.json", "camera_1.mkv", "camera_1.jsonl",
                         "truth.jsonl", "clock.jsonl"):
                self.assertTrue((out / name).exists(), name)
            self.assertEqual((out / "truth.jsonl").read_text(), "")
            rows = read_jsonl(out / "camera_1.jsonl")
            self.assertGreater(len(rows), 10)
            self.assertEqual([r["index"] for r in rows], list(range(len(rows))))
            for key in ("camera_id", "stamp_ns", "wall_ns", "exposure_ns", "line_time_ns",
                        "node_send_ns", "receive_ns"):
                self.assertIn(key, rows[0])
            self.assertEqual(meta["cameras"]["camera_1"]["frames"], len(rows))
            self.assertEqual(meta["full_frame_divisor"], 5)
            self.assertIn("fps", meta["achieved"]["camera_1"])
            clock = read_jsonl(out / "clock.jsonl")
            self.assertGreater(len(clock), 5)
            self.assertLessEqual(clock[0]["sim_ns"], rows[0]["stamp_ns"] + 1_000_000_000)
            # Frames are stored losslessly and in index order: pixel check per frame.
            source = FrameSource(out / "camera_1.mkv")
            for row in rows[:5]:
                rgb = source.read(row["index"])
                self.assertEqual(rgb.shape, (HEIGHT, WIDTH, 3))
                np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 2])
                self.assertEqual(int(rgb[3, 5, 0]), SyntheticSensor.expected_pixel(3, 5, row["frame_seq"]))
            source.close()
            probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v",
                                    "-show_entries", "stream=codec_name,pix_fmt", "-of", "csv=p=0",
                                    str(out / "camera_1.mkv")], capture_output=True, text=True)
            self.assertEqual(probe.stdout.strip(), "ffv1,bgr0")
        finally:
            node.stop()
            tmp.cleanup()


class TimingTool(unittest.TestCase):
    def test_timing_report_has_the_numbers_per_camera(self):
        import lan_camera_timing
        node = start_node(camera_id="camera_9")
        try:
            report = lan_camera_timing.main([
                "--nodes", f"127.0.0.1:{node.port}", "--seconds", "1.0", "--full-period", "0.3",
                "--windows", "0,0,32,32;40,40,32,32"])
        finally:
            node.stop()
        cam = report["cameras"]["camera_9"]
        self.assertGreater(cam["frames_received"], 50)
        self.assertGreater(cam["window_latency"]["n"], 100)
        self.assertGreaterEqual(cam["full_latency"]["n"], 2)
        self.assertLess(cam["window_latency"]["p95"] / 1e6, 100.0)
        self.assertAlmostEqual(cam["period_median_ns"] / 1e6, 10.0, delta=0.5)
        self.assertEqual(cam["node_frames_missed"], 0)


class RollingShutterArithmetic(unittest.TestCase):
    LT, EXP, H = 9452, 800_000, 1232

    def test_led_fit_recovers_line_time_and_reference(self):
        rng = np.random.default_rng(3)
        for reference, offset in (("readout_start_first_row", -self.EXP),
                                  ("exposure_start_first_row", 0),
                                  ("frame_end", -(self.EXP + self.H * self.LT))):
            samples = []
            for i in range(40):
                t0 = 10 ** 15 + i * 12_048_192
                flash = t0 + rng.uniform(self.EXP, (self.H - 1) * self.LT)
                image = flash_analysis.render_flash(self.H, 40, t0, self.LT, self.EXP, flash, rng=rng)
                band = flash_analysis.lit_band(image)
                self.assertFalse(flash_analysis.band_is_truncated(band, self.H))
                samples.append({"flash_ns": flash, "sensor_stamp_ns": t0 - offset,
                                "first_row": band[0], "last_row": band[1], "exposure_ns": self.EXP})
            fit = flash_analysis.fit_rolling_shutter(samples, self.H)
            self.assertAlmostEqual(fit["line_time_ns"], self.LT, delta=self.LT * 0.01)
            self.assertAlmostEqual(fit["stamp_offset_ns"], offset, delta=self.LT * 2)
            self.assertEqual(fit["reference"], reference)
            self.assertAlmostEqual(fit["line_time_from_band_ns"], self.LT, delta=self.LT * 0.03)

    def test_flash_time_from_two_frames_agrees(self):
        # The same flash seen by two synchronized cameras with different phases.
        flash = 10 ** 15 + 5_000_000
        estimates = []
        for phase in (0, 3_000_000, 5_777_777):
            t0 = 10 ** 15 - phase
            image = flash_analysis.render_flash(self.H, 8, t0, self.LT, self.EXP, flash)
            band = flash_analysis.lit_band(image)
            estimates.append(flash_analysis.flash_time_ns(t0, self.LT, self.EXP, *band))
        for estimate in estimates:
            self.assertLess(abs(estimate - flash), self.LT)

    def test_truncated_band_is_detected(self):
        image = flash_analysis.render_flash(100, 8, 0, self.LT, self.EXP, 99 * self.LT + 10)
        band = flash_analysis.lit_band(image)
        self.assertTrue(flash_analysis.band_is_truncated(band, 100))
        self.assertIsNone(flash_analysis.lit_band(np.full((50, 8), 20, np.uint8)))

    def test_reference_correction_signs(self):
        self.assertEqual(stamp_reference_correction_ns("exposure_start_first_row", 800, 9, 10), 0)
        self.assertEqual(stamp_reference_correction_ns("readout_start_first_row", 800, 9, 10), -800)
        self.assertEqual(stamp_reference_correction_ns("frame_end", 800, 9, 10), -890)

    def test_probe_dry_run_matches_the_emulated_platform(self):
        from pi_cam import led_timestamp_probe as probe
        result = probe.run(probe.parse_args(["--synthetic", "--flashes", "12", "--period-s", "0.02"]))
        fit = result["fit"]
        self.assertEqual(fit["reference"], "readout_start_first_row")
        self.assertAlmostEqual(fit["line_time_ns"], self.LT, delta=self.LT * 0.02)
        self.assertEqual(result["recommended_config"]["stamp_reference"], "readout_start_first_row")
        self.assertLess(abs(result["recommended_config"]["stamp_correction_ns"]), 50_000)


class RawBayerPath(unittest.TestCase):
    """The CM4's ISP cannot do 1640x1232 at 83 fps, so luminance comes raw."""

    def test_bayer8_is_its_own_wire_format(self):
        self.assertEqual(proto.FORMAT_NAMES[proto.FORMAT_BAYER8], "bayer8")
        self.assertIn("bayer8", proto.FORMAT_CODES)
        header = proto.FrameHeader("c", 1, 2, 3, 4, 5, 10, 20, 4, 2,
                                   proto.FORMAT_BAYER8, 0, 1, 1640, 1232, 0, 0)
        self.assertEqual(proto.FrameHeader.unpack(header.pack()).format_name, "bayer8")

    def test_a_bayer_window_decodes_and_can_be_flattened(self):
        # Green pixels twice as bright as red and blue, as a lit scene gives.
        tile = np.array([[100, 200], [200, 100]], dtype=np.uint8)
        array = np.tile(tile, (4, 4))
        header = proto.FrameHeader("cam", 1, 2, 3, 4, 5, 0, 0, 8, 8,
                                   proto.FORMAT_BAYER8, 0, 1, 1640, 1232, 0, 0)
        raw = lan_capture.decode_frame(header, array.tobytes(), 1, 1)
        self.assertEqual(raw.format, "bayer8")
        np.testing.assert_array_equal(raw.array, array)
        gains = lan_capture.estimate_bayer_gains(array)
        self.assertAlmostEqual(gains[0][0], 2.0, places=3)
        self.assertAlmostEqual(gains[0][1], 1.0, places=3)
        flat = lan_capture.decode_frame(header, array.tobytes(), 1, 1, bayer_gains=gains)
        self.assertEqual(int(flat.array.min()), 200)
        self.assertEqual(int(flat.array.max()), 200)

    def test_the_gain_map_follows_the_window_phase(self):
        gains = [[2.0, 1.0], [1.0, 2.0]]
        even = lan_capture.bayer_gain_map(gains, 2, 2, row0=100, col0=200)
        np.testing.assert_allclose(even, [[2.0, 1.0], [1.0, 2.0]])
        # An odd offset would shift the phase; normalize_window forbids it, and
        # the map follows it anyway so a caller cannot silently mis-correct.
        odd = lan_capture.bayer_gain_map(gains, 2, 2, row0=101, col0=200)
        np.testing.assert_allclose(odd, [[1.0, 2.0], [2.0, 1.0]])
        for offset in (0, 2, 4, 376):
            w = proto.normalize_window((offset, offset, 480, 480), 1640, 1232)
            self.assertEqual(w["row0"] % 2, 0)
            self.assertEqual(w["col0"] % 2, 0)

    def test_estimate_refuses_a_black_field(self):
        with self.assertRaises(ValueError):
            lan_capture.estimate_bayer_gains(np.zeros((4, 4), np.uint8))

    def test_config_accepts_raw_and_rejects_anything_else(self):
        self.assertEqual(NodeConfig().stream, "isp")
        self.assertEqual(NodeConfig.load(None, stream="raw").stream, "raw")
        self.assertEqual(NodeConfig.load(None, stream="isp").stream, "isp")
        with self.assertRaises(ValueError):
            NodeConfig.load(None, stream="bayer")

    def test_the_node_warns_when_the_pipeline_loses_frames(self):
        # The CM4 through the ISP delivers every second frame while sensor_fps
        # still reads 83, so the loss has to be shouted about, not inferred.
        import logging
        node = start_node(camera_id="cam_w")
        try:
            with self.assertLogs("camera_node", level=logging.WARNING) as captured:
                with node.stats_lock:
                    node.frames_captured += 50
                    node.frames_missed += 50
                deadline = time.monotonic() + 4.0
                while time.monotonic() < deadline and not captured.records:
                    time.sleep(0.05)
            text = "\n".join(captured.output)
            self.assertIn("lost", text)
            self.assertIn("raw", text, "the warning must name the fix")
        finally:
            node.stop()

    def test_hello_says_which_stream_the_node_reads(self):
        node = start_node(camera_id="cam_s")
        source = lan_capture.LanCameraSource([("127.0.0.1", node.port)])
        try:
            source.wait_connected(5.0)
            hello = source.hello("cam_s")
            self.assertEqual(hello["stream"], "isp")
            self.assertEqual(hello["wire_format"], "y8")
        finally:
            source.close()
            node.stop()


class TimingToolFullFrameMode(unittest.TestCase):
    def test_counters_are_reported_per_run_not_since_the_node_started(self):
        # The node's counters run from its own start, so a run following a
        # saturating one inherited thousands of drops that were not its own.
        import lan_camera_timing as tool
        self.assertEqual(tool._delta({"frames_missed": 10}, {"frames_missed": 4},
                                     "frames_missed"), 6)
        self.assertEqual(tool._delta({"frames_missed": 10}, None, "frames_missed"), 10)
        self.assertIsNone(tool._delta({}, {"frames_missed": 1}, "frames_missed"))
        self.assertIsNone(tool._delta(None, None, "frames_missed"))

    def test_stream_full_mode_measures_full_frames_and_clears_it_afterwards(self):
        import lan_camera_timing as tool
        node = start_node(camera_id="camera_f")
        try:
            report = tool.main(["--nodes", f"127.0.0.1:{node.port}", "--seconds", "1.0",
                                "--stream-full", "2", "--full-format", "y8"])
        finally:
            node.stop()
        cam = report["cameras"]["camera_f"]
        self.assertEqual(report["stream_full_divisor"], 2)
        self.assertEqual(report["windows"], [], "stream_full must not also ask for windows")
        self.assertEqual(cam["window_latency"]["n"], 0)
        self.assertGreater(cam["full_latency"]["n"], 10)
        self.assertEqual(cam["node_frames_missed"], 0)
        # Every other frame at 100 fps.
        self.assertAlmostEqual(cam["delivered_fps"], 50.0, delta=15.0)
        self.assertIsNone(node.full_stream, "the stream was left running on the node")


class TimedFullStream(unittest.TestCase):
    """Full frames by timer with a phase per node (calibration station)."""

    def collect(self, source, camera_id, seconds):
        frames = []
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            group = source.take(camera_id, 0.2)
            if group:
                frames.extend(f for f in group if f.is_full)
        return frames

    def test_frames_come_once_per_period_at_the_given_phase(self):
        node = start_node(camera_id="cam_timed")
        source = lan_capture.LanCameraSource([("127.0.0.1", node.port)])
        try:
            self.assertEqual(source.wait_connected(5.0), ["cam_timed"])
            period, phase = 50_000_000, 20_000_000
            ack = source.stream_full("cam_timed", 0, "y8", hz=20.0, phase_ns=phase)
            self.assertTrue(ack["ok"], ack)
            self.assertEqual((ack["hz"], ack["phase_ns"]), (20.0, phase))
            self.collect(source, "cam_timed", 0.3)
            frames = self.collect(source, "cam_timed", 1.5)
        finally:
            source.close()
            node.stop()
        self.assertAlmostEqual(len(frames) / 1.5, 20.0, delta=4.0)
        ticks = [(f.stamp_ns - phase) // period for f in frames]
        self.assertEqual(len(set(ticks)), len(ticks), "two frames in one period")
        # Each frame is the first sensor frame (10 ms apart) after its tick.
        offsets = [(f.stamp_ns - phase) % period for f in frames]
        self.assertLess(max(offsets), 12_000_000)

    def test_gray_turns_a_raw_mosaic_into_luminance_before_jpeg(self):
        from pi_cam.camera_node import demosaic_full_gray
        node = start_node(camera_id="cam_raw")
        node.wire_format = "bayer8"           # behave like a CM4 reading the raw stream
        source = lan_capture.LanCameraSource([("127.0.0.1", node.port)])
        try:
            source.wait_connected(5.0)
            source.stream_full("cam_raw", 0, "jpeg", hz=10.0, gray=True)
            self.collect(source, "cam_raw", 0.3)
            gray = self.collect(source, "cam_raw", 0.6)
            source.stream_full("cam_raw", 0, "y8", hz=10.0)
            self.collect(source, "cam_raw", 0.3)
            raw = self.collect(source, "cam_raw", 0.6)
        finally:
            source.close()
            node.stop()
        self.assertTrue(gray and raw)
        self.assertEqual({f.format for f in gray}, {"jpeg"})
        self.assertEqual({f.format for f in raw}, {"bayer8"}, "without gray the mosaic travels raw")
        frame = gray[-1]
        mosaic = np.asarray([[SyntheticSensor.expected_pixel(r, c, frame.frame_seq)
                              for c in range(WIDTH)] for r in range(HEIGHT)], np.uint8)
        expected = demosaic_full_gray(mosaic).astype(float)
        self.assertLess(np.abs(frame.array.astype(float) - expected)[4:-4, 4:-4].mean(), 4.0)

    def test_divisor_stream_is_unchanged(self):
        node = start_node(camera_id="cam_div")
        source = lan_capture.LanCameraSource([("127.0.0.1", node.port)])
        try:
            source.wait_connected(5.0)
            ack = source.stream_full("cam_div", 4)
            self.assertNotIn("hz", ack)
            self.assertEqual(node.full_stream["divisor"], 4)
            source.stream_full("cam_div", 0)
            time.sleep(0.2)
            self.assertIsNone(node.full_stream)
        finally:
            source.close()
            node.stop()


class ExposureSweep(unittest.TestCase):
    """The probe that decided the timestamp reference without an LED."""

    def test_fit_reads_a_flat_sweep_as_readout_start(self):
        from pi_cam import exposure_sweep_probe as probe
        # Measured on the CM5: the delay does not move with the exposure.
        rows = [(0.5, 0.49, 14.18, 14.9), (1.0, 0.99, 14.18, 14.9),
                (2.0, 1.99, 14.17, 14.4), (4.0, 4.00, 14.18, 14.2)]
        fit = probe.fit(rows, frame_duration_ms=12.04)
        self.assertEqual(fit["reference"], "readout_start_first_row")
        self.assertLess(abs(fit["slope_ms_per_ms"]), 0.01)
        self.assertTrue(fit["conclusive"])
        self.assertAlmostEqual(fit["intercept_ms"], 14.18, delta=0.02)

    def test_a_noisy_but_flat_sweep_is_still_conclusive(self):
        from pi_cam import exposure_sweep_probe as probe
        # The real CM5 sweep: a 0.7 ms step against a 3.5 ms exposure change.
        rows = [(0.5, 0.49, 14.495, 14.86), (1.0, 0.99, 14.501, 14.87),
                (2.0, 1.99, 13.804, 14.44), (4.0, 4.00, 13.803, 14.17)]
        fit = probe.fit(rows, frame_duration_ms=12.04)
        self.assertEqual(fit["reference"], "readout_start_first_row")
        self.assertTrue(fit["conclusive"], fit)
        self.assertLess(fit["delay_span_ms"], fit["exposure_span_ms"] / 4)
        self.assertIn("not in the timestamp", fit["verdict"])

    def test_a_slope_between_the_hypotheses_is_refused(self):
        from pi_cam import exposure_sweep_probe as probe
        rows = [(0.5, 0.5, 14.0, 14.0), (4.0, 4.0, 16.0, 16.0)]   # slope 0.57
        fit = probe.fit(rows)
        self.assertEqual(fit["reference"], "unknown")
        self.assertFalse(fit["conclusive"])
        with self.assertRaises(ValueError):
            probe.fit(rows, tolerance=0.6)

    def test_fit_reads_a_rising_sweep_as_exposure_start(self):
        from pi_cam import exposure_sweep_probe as probe
        rows = [(0.5, 0.5, 12.5, 12.6), (1.0, 1.0, 13.0, 13.1),
                (2.0, 2.0, 14.0, 14.1), (4.0, 4.0, 16.0, 16.1)]
        fit = probe.fit(rows, frame_duration_ms=12.04)
        self.assertEqual(fit["reference"], "exposure_start_first_row")
        self.assertAlmostEqual(fit["slope_ms_per_ms"], 1.0, delta=0.01)
        self.assertTrue(fit["conclusive"])

    def test_long_exposures_are_dropped_from_the_fit(self):
        from pi_cam import exposure_sweep_probe as probe
        # The 8 ms point of a 12.04 ms frame jumped 1.5 ms on hardware and must
        # not drag the slope.
        rows = [(0.5, 0.49, 14.18, 14.2), (1.0, 0.99, 14.18, 14.2),
                (2.0, 1.99, 14.17, 14.2), (4.0, 4.00, 14.18, 14.2),
                (8.0, 8.00, 15.67, 15.7)]
        fit = probe.fit(rows, frame_duration_ms=12.04)
        self.assertEqual(fit["points_dropped"], 1)
        self.assertEqual(fit["reference"], "readout_start_first_row")
        self.assertGreater(abs(probe.fit(rows)["slope_ms_per_ms"]), 0.1)   # unfiltered

    def test_probe_runs_end_to_end_on_the_synthetic_sensor(self):
        from pi_cam import exposure_sweep_probe as probe
        result = probe.run(probe.parse_args(
            ["--synthetic", "--exposures-us", "500,1000,2000", "--frames", "5"]))
        self.assertEqual(len(result["rows"]), 3)
        # The synthetic sensor stamps the exposure start and delivers one period
        # later, so the delay must not follow the exposure.
        self.assertEqual(result["fit"]["reference"], "readout_start_first_row")
        self.assertIn("stamp_reference", result["config_line"])


class NodeConfigFile(unittest.TestCase):
    def test_example_config_loads_and_rejects_unknown_keys(self):
        cfg = NodeConfig.load(ROOT / "pi_cam" / "node_config.example.json")
        self.assertEqual(cfg.computed_line_time_ns(), 9452)
        self.assertEqual(cfg.line_time_source(), "register_model")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({"camera_id": "x", "bogus": 1}, handle)
        with self.assertRaises(ValueError):
            NodeConfig.load(handle.name)
        Path(handle.name).unlink()

    def test_service_starts_the_node_with_the_config(self):
        unit = (ROOT / "pi_cam" / "camera_node.service").read_text()
        self.assertIn("camera_node.py --config", unit)
        self.assertIn("Restart=always", unit)


if __name__ == "__main__":
    unittest.main()
