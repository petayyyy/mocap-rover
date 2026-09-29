#!/usr/bin/env python3
"""Record a replayable dataset from live CM4/CM5 camera nodes over LAN.

Writes the format of ``scripts/record_dataset.py`` so that
``scripts/replay_dataset.py`` and ``scripts/evaluate_recording.py`` read it
unchanged::

    <output>/
      meta.json                 nodes, counts, drops, achieved rate, time span
      runtime_cameras.json      calibration copied from --config
      camera_N.mkv              lossless FFV1 (bgr0, as the Gazebo recorder), grey replicated
      camera_N.jsonl            index, stamp_ns, wall_ns + exposure_ns, line_time_ns,
                                node_send_ns, receive_ns, frame_seq, frame_duration_ns
      truth.jsonl               empty: a live recording has no ground truth
      clock.jsonl               laptop CLOCK_REALTIME samples every 100 ms

A full frame at the sensor rate does not fit 1 GbE, so every node is switched
to ``stream_full`` with a divisor: the sensor keeps its 83 fps timing and the
node sends every N-th full frame.  The rate the channel actually carried is
measured and written to ``meta.json`` (``achieved``), together with the node's
own drop counters.  ``stamp_ns`` is the common PTP scale, exposure start of
the first row; ``wall_ns`` is the laptop's monotonic clock at receipt, as in
the Gazebo recorder, and ``receive_ns`` the laptop's CLOCK_REALTIME.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from localization_contracts.lan_capture import LanCameraSource, LanNotConnected  # noqa: E402


class VideoWriter(threading.Thread):
    """Feed Y8 frames to one ffmpeg process; index rows go to a jsonl."""

    def __init__(self, path: Path, width, height, fps, codec_args):
        super().__init__(daemon=True)
        self.path = path
        self.queue: queue.Queue = queue.Queue()
        self.index = path.with_suffix(".jsonl").open("w")
        self.count = 0
        self.bytes = 0
        self.width, self.height = width, height
        self.process = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
             "-pix_fmt", "gray", "-s", f"{width}x{height}", "-r", str(fps),
             "-i", "-", "-vf", "format=rgb24", *codec_args, str(path)],
            stdin=subprocess.PIPE)

    def run(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            frame, row = item
            self.process.stdin.write(frame)
            row["index"] = self.count
            self.index.write(json.dumps(row) + "\n")
            self.count += 1
            self.bytes += len(frame)
        self.process.stdin.close()
        self.process.wait()
        self.index.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nodes", nargs="+", required=True, help="host:port of every camera node")
    p.add_argument("--config", required=True,
                   help="runtime_cameras.json (image_calibrated) of the live arena")
    p.add_argument("--output", required=True)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--divisor", type=int, default=8,
                   help="send every N-th sensor frame in full (83/8 ~ 10 fps per camera)")
    p.add_argument("--format", choices=("y8", "jpeg"), default="y8",
                   help="wire format of full frames; jpeg is lossy and only for a slow link")
    p.add_argument("--codec", choices=("ffv1", "x264-lossless"), default="ffv1")
    p.add_argument("--connect-timeout", type=float, default=10.0)
    p.add_argument("--allow-missing", action="store_true",
                   help="record the cameras that answered even if the config lists more")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    cfg = json.loads(Path(a.config).read_text())
    cams = {c["name"]: c for c in cfg["cameras"]}
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; choose a new --output")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.config, out / "runtime_cameras.json")

    source = LanCameraSource(a.nodes)
    seen = source.wait_connected(a.connect_timeout)
    missing = sorted(set(cams) - set(seen))
    unknown = sorted(set(seen) - set(cams))
    if unknown:
        print(f"nodes announced cameras not in the config, ignored: {unknown}", flush=True)
    if missing and not a.allow_missing:
        source.close()
        raise SystemExit(f"no node for cameras {missing} (seen {seen}); use --allow-missing")
    recorded = [cid for cid in cams if cid in seen]
    if not recorded:
        source.close()
        raise SystemExit("no camera node answered")

    codec = (["-c:v", "ffv1", "-level", "3", "-threads", "2"] if a.codec == "ffv1"
             else ["-c:v", "libx264", "-qp", "0", "-preset", "veryfast",
                   "-pix_fmt", "rgb24", "-threads", "2"])
    hello = {cid: source.hello(cid) for cid in recorded}
    for cid in recorded:
        expected = tuple(cams[cid]["image_size"])
        actual = (hello[cid]["sensor_width"], hello[cid]["sensor_height"])
        if actual != expected:
            source.close()
            raise SystemExit(f"{cid}: node sensor {actual} != config image_size {expected}")
    nominal_fps = max(1.0, 83.0 / a.divisor)
    writers = {}
    for cid in recorded:
        w, h = cams[cid]["image_size"]
        writers[cid] = VideoWriter(out / f"{cid}.mkv", w, h, round(nominal_fps, 3), codec)
        writers[cid].start()

    for cid in recorded:
        source.request_windows(cid, [])
        source.stream_full(cid, a.divisor, a.format)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    (out / "truth.jsonl").write_text("")
    clock_file = (out / "clock.jsonl").open("w")
    counts = {"frames": dict.fromkeys(recorded, 0), "size_errors": 0, "windows_ignored": 0}
    wire_bytes = dict.fromkeys(recorded, 0)
    first_stamp = {}
    last_stamp = {}
    wall_start = time.monotonic()
    start_realtime = time.clock_gettime_ns(time.CLOCK_REALTIME)
    last_clock = 0
    last_print = wall_start
    print(f"recording {recorded} to {out}; divisor {a.divisor}, format {a.format}", flush=True)
    while not stop.is_set():
        now_mono = time.monotonic()
        realtime = time.clock_gettime_ns(time.CLOCK_REALTIME)
        if realtime - last_clock >= 100_000_000:
            last_clock = realtime
            clock_file.write(json.dumps({"sim_ns": realtime, "wall_ns": time.monotonic_ns()}) + "\n")
        if now_mono - wall_start >= a.seconds:
            break
        item = source.take_any(0.05)
        if item is None:
            continue
        cid, group = item
        if cid not in writers:
            continue
        for frame in group:
            if not frame.is_full:
                counts["windows_ignored"] += 1
                continue
            if (frame.width, frame.height) != tuple(cams[cid]["image_size"]):
                counts["size_errors"] += 1
                continue
            row = {"camera_id": cid, "stamp_ns": frame.stamp_ns, "wall_ns": frame.receive_mono_ns,
                   "exposure_ns": frame.exposure_ns, "line_time_ns": frame.line_time_ns,
                   "node_send_ns": frame.node_send_ns, "receive_ns": frame.receive_ns,
                   "frame_seq": frame.frame_seq, "frame_duration_ns": frame.frame_duration_ns,
                   "format": frame.format}
            writers[cid].queue.put((frame.array.tobytes(), row))
            counts["frames"][cid] += 1
            wire_bytes[cid] += frame.payload_bytes
            first_stamp.setdefault(cid, frame.stamp_ns)
            last_stamp[cid] = frame.stamp_ns
        if now_mono - last_print >= 1.0:
            last_print = now_mono
            elapsed = now_mono - wall_start
            stats = source.stats()
            per_cam = " ".join(f"{cid}:{counts['frames'][cid] / max(elapsed, 1e-9):.1f}fps"
                               for cid in recorded)
            mbit = sum(l["bytes"] for l in stats["links"].values()) * 8 / max(elapsed, 1e-9) / 1e6
            print(f"wall {elapsed:6.1f}s {per_cam} link {mbit:.0f} Mbit/s laptop drops "
                  f"{sum(stats['dropped'].values())} queued "
                  f"{sum(w.queue.qsize() for w in writers.values())}", flush=True)

    for cid in recorded:
        try:
            source.stream_full(cid, 0, a.format, timeout=1.0)
        except (LanNotConnected, TimeoutError):
            pass
    statuses = {cid: source.status(cid) for cid in recorded}
    stats = source.stats()
    source.close()
    for w in writers.values():
        w.queue.put(None)
    for w in writers.values():
        w.join()
    clock_file.close()
    wall_seconds = time.monotonic() - wall_start
    achieved = {}
    for cid in recorded:
        n = writers[cid].count
        span = (last_stamp.get(cid, 0) - first_stamp.get(cid, 0)) / 1e9 if n > 1 else 0.0
        achieved[cid] = {
            "frames": n, "fps": n / wall_seconds if wall_seconds else 0.0,
            "fps_by_stamps": (n - 1) / span if span > 0 else None,
            "mbit_s_decoded": writers[cid].bytes * 8 / wall_seconds / 1e6 if wall_seconds else 0.0,
            "mbit_s_wire": wire_bytes[cid] * 8 / wall_seconds / 1e6 if wall_seconds else 0.0,
            "laptop_dropped": stats["dropped"].get(cid, 0),
            "node_frames_dropped_queue": (statuses[cid] or {}).get("frames_dropped_queue"),
            "node_frames_missed": (statuses[cid] or {}).get("frames_missed"),
            "node_sensor_fps": (statuses[cid] or {}).get("sensor_fps"),
        }
    meta = {
        "world": None, "source": "lan_camera_nodes",
        "calibration_version": cfg.get("calibration_version"),
        "nodes": list(a.nodes),
        "cameras": {cid: {"image_size": cams[cid]["image_size"], "video": f"{cid}.mkv",
                          "index": f"{cid}.jsonl", "frames": writers[cid].count}
                    for cid in recorded},
        "lidar": None, "truth_rows": 0,
        "image_size_errors": counts["size_errors"], "windows_ignored": counts["windows_ignored"],
        "sim_start_ns": start_realtime, "sim_end_ns": time.clock_gettime_ns(time.CLOCK_REALTIME),
        "wall_seconds": wall_seconds, "codec": a.codec, "pixel_format": "rgb24",
        "wire_format": a.format, "full_frame_divisor": a.divisor,
        "nominal_fps": nominal_fps, "achieved": achieved,
        "node_hello": hello, "node_status_end": statuses,
        "link_stats": stats["links"], "time_scale": "CLOCK_REALTIME via PTP",
        "stamp_reference": "exposure_start_first_row",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"achieved": achieved, "wall_seconds": wall_seconds}, indent=2), flush=True)
    return meta


if __name__ == "__main__":
    main()
    os._exit(0)
