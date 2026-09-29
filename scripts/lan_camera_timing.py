#!/usr/bin/env python3
"""Measure the camera nodes in their match configuration and print the numbers.

Runs the nodes as the tracker will: two windows on every frame plus one full
frame every ``--full-period`` seconds, for ``--seconds``.  Prints per camera

* exposure -> laptop latency P50/P95/max, windows and full frames separately
  (``receive_ns - stamp_ns``, both on the common PTP scale, so a clock error
  between node and laptop shows up here as well);
* node-side latency exposure -> send and the network share;
* frame period jitter from ``stamp_ns`` (deviation from the median period);
* drops: sensor frames missed by the node, frames the node dropped because
  the link was behind, frames the laptop dropped because the consumer was;
* the node's own status: sensor fps, PTP offset and state, SoC temperature,
  throttling flags, CPU load; and the laptop's CPU load.

The whole output is the report; ``--json`` also saves everything raw.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from localization_contracts.lan_capture import LanCameraSource, LanNotConnected  # noqa: E402


def summary(values):
    if not values:
        return {"n": 0, "p50": None, "p95": None, "max": None, "mean": None}
    a = np.asarray(values, dtype=float)
    return {"n": int(a.size), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max()), "mean": float(a.mean())}


def fmt(s, scale=1.0, unit=""):
    if not s or s["n"] == 0:
        return "n=0"
    return (f"n={s['n']} P50 {s['p50'] / scale:.2f}{unit} P95 {s['p95'] / scale:.2f}{unit} "
            f"max {s['max'] / scale:.2f}{unit}")


class LaptopCpu:
    def __init__(self):
        self.last = self.read()

    @staticmethod
    def read():
        try:
            v = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
            return sum(v), v[3] + v[4]
        except (OSError, ValueError, IndexError):
            return None

    def percent(self):
        now = self.read()
        if now is None or self.last is None:
            return None
        total, idle = now[0] - self.last[0], now[1] - self.last[1]
        self.last = now
        return None if total <= 0 else 100.0 * (total - idle) / total


def parse_windows(spec, size):
    if spec:
        out = []
        for item in spec.split(";"):
            r, c, w, h = (int(v) for v in item.split(","))
            out.append((r, c, w, h))
        return out
    # Two windows away from each other, as two rovers would be.
    return [(120, 160, size, size), (620, 1000, size, size)]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nodes", nargs="+", required=True)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--windows", help='"row0,col0,w,h;row0,col0,w,h" (default: two 480x480)')
    p.add_argument("--window-size", type=int, default=480)
    p.add_argument("--window-format", choices=("y8", "jpeg"), default="y8")
    p.add_argument("--full-period", type=float, default=2.0)
    p.add_argument("--full-format", choices=("y8", "jpeg"), default="y8")
    p.add_argument("--stream-full", type=int, metavar="DIVISOR",
                   help="measure a continuous stream of FULL frames instead of windows: "
                        "every DIVISOR-th sensor frame (1 = every frame, the maximum the "
                        "sensor and the link can carry)")
    p.add_argument("--json", help="write the raw measurements here")
    p.add_argument("--connect-timeout", type=float, default=10.0)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    windows = [(r, c, w, h, a.window_format) for r, c, w, h in parse_windows(a.windows, a.window_size)]
    source = LanCameraSource(a.nodes)
    cams = source.wait_connected(a.connect_timeout)
    if not cams:
        source.close()
        raise SystemExit("no camera node answered")
    if a.stream_full:
        windows = []
        print(f"cameras: {cams}; FULL frames every {a.stream_full} sensor frame(s) "
              f"({a.full_format}), no windows; {a.seconds}s", flush=True)
        for cid in cams:
            source.request_windows(cid, [])
            source.stream_full(cid, a.stream_full, a.full_format)
    else:
        print(f"cameras: {cams}; windows {windows}; full frame every {a.full_period}s "
              f"({a.full_format}); {a.seconds}s", flush=True)
        for cid in cams:
            source.request_windows(cid, windows)

    per = {cid: {"window_latency_ns": [], "full_latency_ns": [], "node_latency_ns": [],
                 "net_latency_ns": [], "period_dev_ns": [], "stamps": [], "seqs": [],
                 "statuses": [], "window_bytes": 0, "full_bytes": 0} for cid in cams}
    cpu = LaptopCpu()
    cpu_samples = []
    start = time.monotonic()
    next_full = start
    next_cpu = start + 1.0
    while time.monotonic() - start < a.seconds:
        now = time.monotonic()
        if not a.stream_full and now >= next_full:
            next_full = now + a.full_period
            for cid in cams:
                try:
                    source.request_full(cid, a.full_format, wait=False)
                except LanNotConnected:
                    pass
        if now >= next_cpu:
            next_cpu = now + 1.0
            value = cpu.percent()
            if value is not None:
                cpu_samples.append(value)
            for cid in cams:
                status = source.status(cid)
                if status is not None and (not per[cid]["statuses"]
                                           or per[cid]["statuses"][-1] is not status):
                    per[cid]["statuses"].append(status)
        item = source.take_any(0.05)
        if item is None:
            continue
        cid, group = item
        d = per.get(cid)
        if d is None:
            continue
        for f in group:
            latency = f.receive_ns - f.stamp_ns
            # Wire bytes, so a JPEG window counts its compressed size.
            if f.is_full:
                d["full_latency_ns"].append(latency)
                d["full_bytes"] += f.payload_bytes
            else:
                d["window_latency_ns"].append(latency)
                d["window_bytes"] += f.payload_bytes
            d["node_latency_ns"].append(f.node_send_ns - f.stamp_ns)
            d["net_latency_ns"].append(f.receive_ns - f.node_send_ns)
        d["stamps"].append(group[0].stamp_ns)
        d["seqs"].append(group[0].frame_seq)

    final_status = {cid: source.status(cid) for cid in cams}
    stats = source.stats()
    if a.stream_full:
        for cid in cams:
            try:
                source.stream_full(cid, 0, a.full_format, timeout=1.0)
            except (LanNotConnected, TimeoutError):
                pass
    source.close()
    elapsed = time.monotonic() - start

    report = {"seconds": elapsed, "windows": windows, "full_period_s": a.full_period,
              "stream_full_divisor": a.stream_full, "full_format": a.full_format,
              "laptop_cpu_percent": summary(cpu_samples), "cameras": {}}
    for cid in cams:
        d = per[cid]
        stamps, seqs = np.asarray(d["stamps"], dtype=np.int64), np.asarray(d["seqs"], dtype=np.int64)
        period_dev = []
        median_period = None
        if stamps.size > 2:
            dstamp, dseq = np.diff(stamps), np.diff(seqs)
            ok = dseq > 0
            per_frame = dstamp[ok] / dseq[ok]
            median_period = float(np.median(per_frame))
            period_dev = np.abs(per_frame - median_period).tolist()
        statuses = d["statuses"]
        s_end = final_status[cid] or (statuses[-1] if statuses else {})
        ptp = [s["ptp"]["offset_ns"] for s in statuses if s.get("ptp", {}).get("offset_ns") is not None]
        temps = [s["soc_temp_c"] for s in statuses if s.get("soc_temp_c") is not None]
        node_cpu = [s["cpu_percent"] for s in statuses if s.get("cpu_percent") is not None]
        fps = [s["sensor_fps"] for s in statuses if s.get("sensor_fps")]
        throttled = sorted({flag for s in statuses for flag in (s.get("throttled") or {}).get("flags", [])})
        report["cameras"][cid] = {
            "frames_received": int(stamps.size), "laptop_dropped": stats["dropped"].get(cid, 0),
            "window_latency": summary(d["window_latency_ns"]),
            "full_latency": summary(d["full_latency_ns"]),
            "node_latency": summary(d["node_latency_ns"]),
            "net_latency": summary(d["net_latency_ns"]),
            "period_median_ns": median_period, "period_jitter": summary(period_dev),
            "delivered_fps": stamps.size / elapsed if elapsed else None,
            "window_mbit_s": d["window_bytes"] * 8 / elapsed / 1e6 if elapsed else None,
            "full_mbit_s": d["full_bytes"] * 8 / elapsed / 1e6 if elapsed else None,
            "sensor_fps": summary(fps), "ptp_offset_ns": summary(ptp),
            "ptp_offset_abs_ns": summary([abs(v) for v in ptp]),
            "ptp_state": s_end.get("ptp", {}).get("state"),
            "soc_temp_c": summary(temps), "node_cpu_percent": summary(node_cpu),
            "throttled_flags": throttled,
            "node_frames_missed": s_end.get("frames_missed"),
            "node_frames_dropped_queue": s_end.get("frames_dropped_queue"),
            "node_capture_to_send_ms": s_end.get("capture_to_send_ms"),
            "line_time_ns": s_end.get("line_time_ns"), "exposure_ns": s_end.get("exposure_ns"),
            "link": stats["links"],
        }

    print()
    mode = (f"FULL frames every {a.stream_full} sensor frame(s) ({a.full_format})"
            if a.stream_full else f"windows {windows}, full every {a.full_period}s")
    print(f"=== lan_camera_timing: {elapsed:.1f}s, {mode} ===")
    print(f"laptop CPU: {fmt(report['laptop_cpu_percent'], 1, '%')}")
    for cid, r in report["cameras"].items():
        print(f"\n[{cid}] frames {r['frames_received']} ({r['delivered_fps']:.1f}/s), "
              f"laptop drops {r['laptop_dropped']}, node missed {r['node_frames_missed']}, "
              f"node dropped {r['node_frames_dropped_queue']}")
        print(f"  sensor fps       {fmt(r['sensor_fps'], 1)}")
        print(f"  window latency   {fmt(r['window_latency'], 1e6, 'ms')}")
        print(f"  full latency     {fmt(r['full_latency'], 1e6, 'ms')}")
        print(f"  node exp->send   {fmt(r['node_latency'], 1e6, 'ms')}")
        print(f"  send->receive    {fmt(r['net_latency'], 1e6, 'ms')}")
        median = r["period_median_ns"]
        print(f"  period           {median / 1e6 if median else float('nan'):.3f} ms, "
              f"jitter {fmt(r['period_jitter'], 1e3, 'us')}")
        print(f"  link             windows {r['window_mbit_s']:.1f} Mbit/s, full {r['full_mbit_s']:.1f} Mbit/s")
        print(f"  ptp              state {r['ptp_state']}, |offset| {fmt(r['ptp_offset_abs_ns'], 1e3, 'us')}")
        print(f"  node             temp {fmt(r['soc_temp_c'], 1, 'C')}, cpu {fmt(r['node_cpu_percent'], 1, '%')}, "
              f"throttled {r['throttled_flags'] or 'none'}")
        print(f"  line_time {r['line_time_ns']} ns, exposure {r['exposure_ns']} ns")
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nraw measurements: {a.json}")
    return report


if __name__ == "__main__":
    main()
