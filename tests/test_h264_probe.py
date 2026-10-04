"""The H.264 probe without the CM4: V4L2 ABI numbers and the timing logic
against a stand-in encoder that either releases each frame or holds one."""
from __future__ import annotations

import threading
import time

import pytest

from pi_cam import h264_probe as hp
from pi_cam.camera_node import boottime_ns


def test_ioctl_numbers_match_the_kernel_headers():
    # Printed by a C program built against <linux/videodev2.h> (x86_64; the
    # same LP64 layout as the CM4's aarch64).
    assert hp.VIDIOC_QUERYCAP == 2154321408
    assert hp.VIDIOC_S_FMT == 3234878981
    assert hp.VIDIOC_REQBUFS == 3222558216
    assert hp.VIDIOC_QUERYBUF == 3227014665
    assert hp.VIDIOC_QBUF == 3227014671
    assert hp.VIDIOC_DQBUF == 3227014673
    assert hp.VIDIOC_STREAMON == 1074026002
    assert hp.VIDIOC_STREAMOFF == 1074026003
    assert hp.VIDIOC_S_PARM == 3234616854
    assert hp.VIDIOC_S_CTRL == 3221771804
    assert hp.CID_BITRATE == 10029519 and hp.CID_H264_I_PERIOD == 10029670
    assert hp.CID_H264_LEVEL == 10029671 and hp.CID_H264_PROFILE == 10029675
    assert hp.CID_REPEAT_SEQ_HEADER == 10029538 and hp.CID_BITRATE_MODE == 10029518
    assert hp.PIX_FMT_YUV420 == 842093913 and hp.PIX_FMT_H264 == 875967048


def test_parse_case():
    assert hp.parse_case("640x480@30") == (640, 480, 30.0)
    assert hp.parse_case("full@all") == (1640, 1232, 0.0)
    assert hp.parse_case("full") == (1640, 1232, 0.0)


class FakeEncoder:
    """Encodes in ``delay_s``; with ``hold`` it gives frame k out only when
    frame k+1 comes in, like an encoder that keeps one frame back."""

    delay_s = 0.004
    hold = False

    def __init__(self, device, width, height, fps, bitrate, gop, profile="high", level="4.2",
                 rate_mode="vbr", keep_data=False, on_output=None, **_):
        self.on_output = on_output
        self.gop = gop
        self.count = 0
        self.held = None
        self.lock = threading.Lock()
        self.info = {"device": "fake", "card": "fake", "stride": width}
        self.controls = {"bitrate": bitrate, "i_period": gop}

    def _emit(self, frame_id, n):
        def later():
            time.sleep(self.delay_s)
            self.on_output(frame_id, boottime_ns(), 5000, n % self.gop == 0, b"x")
        threading.Thread(target=later, daemon=True).start()

    def submit(self, gray, frame_id):
        queued = boottime_ns()
        with self.lock:
            n, self.count = self.count, self.count + 1
            if self.hold:
                if self.held is not None:
                    self._emit(*self.held)
                self.held = (frame_id, n)
            else:
                self._emit(frame_id, n)
        return queued

    def close(self):
        pass


def run(cases, hold, tmp_path):
    FakeEncoder.hold = hold
    out = tmp_path / "r.json"
    return hp.main(["--synthetic", "--cases", *cases, "--seconds", "1.0",
                    "--hold-gap-s", "0.05", "--output", str(out)],
                   make_encoder=FakeEncoder), out


@pytest.mark.parametrize("hold", [False, True])
def test_hold_is_detected(hold, tmp_path):
    results, out = run(["640x480@30"], hold, tmp_path)
    r = results["cases"][0]
    assert "error" not in r, r.get("error")
    assert r["hold_test"]["holds_a_frame"] is hold
    if hold:
        assert r["released_before_next_input"] < 0.1
    else:
        assert r["released_before_next_input"] > 0.9
        assert 3.0 < r["latency_ms"]["encoder"]["p50"] < 20.0
    assert out.exists()


def test_rates_and_counts(tmp_path):
    results, _ = run(["640x480@30", "full@all"], False, tmp_path)
    small, full = results["cases"]
    assert 25 <= small["input_fps"] <= 32
    assert full["input_fps"] > 60
    for r in (small, full):
        assert r["lost_in_encoder"] == 0
        assert r["encoded"] == r["submitted"]
        assert r["latency_ms"]["total_exposure_to_encoded"]["p50"] > r["latency_ms"]["encoder"]["p50"]
        assert r["frame_kbytes"]["keyframes"]["n"] > 0
