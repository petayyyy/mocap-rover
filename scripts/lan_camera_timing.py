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

Target profile (the hybrid): ``--small 640x480@30 --small-stagger`` adds every
node's small stream, the whole frame reduced on the node, with the nodes'
frames spread over the period; ``--window-nodes 3 --window-size 320
--full-period 0`` keeps the marker windows on three nodes and drops the full
frames.  The small stream gets its own latency, rate, drops and the node's
reduction time.

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


def estimate_clock_offset(source, camera_id, rounds=9):
    """Node clock minus laptop clock, by the NTP estimator over the command path.

    Every cross-machine latency here is ``receive_ns - stamp_ns`` with the two
    ends read on different machines, so without PTP it carries the clock offset
    whole.  Unsynchronised boards are tens of milliseconds apart -- a CM4
    measured 25 ms off, which turned a real 27 ms latency into an apparent
    2.6 ms.  A status request gives the node's own clock inside a round trip we
    time locally, and the round with the smallest round trip bounds the offset
    to half of it.
    """
    best = None
    for _ in range(rounds):
        before = time.clock_gettime_ns(time.CLOCK_REALTIME)
        try:
            source.request_status(camera_id, timeout=2.0)
        except (LanNotConnected, TimeoutError):
            continue
        status = None
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            candidate = source.status(camera_id)
            if candidate is not None and candidate.get("status_time_ns", 0) > before:
                status = candidate
                break
            time.sleep(0.002)
        after = time.clock_gettime_ns(time.CLOCK_REALTIME)
        if status is None:
            continue
        rtt = after - before
        offset = status["status_time_ns"] - (before + after) // 2
        if best is None or rtt < best["rtt_ns"]:
            best = {"offset_ns": offset, "rtt_ns": rtt, "uncertainty_ns": rtt // 2}
    return best


def _delta(end, start, key):
    """Counter difference over this run, or the raw value if no baseline exists."""
    if not end or end.get(key) is None:
        return None
    if not start or start.get(key) is None:
        return end[key]
    return end[key] - start[key]


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
    p.add_argument("--full-period", type=float, default=2.0, help="0 = no full frames")
    p.add_argument("--window-nodes", type=int, default=0,
                   help="only the first N nodes get windows (0 = all)")
    p.add_argument("--no-windows", action="store_true",
                   help="no windows on any node (e.g. the small stream alone)")
    p.add_argument("--small", metavar="WxH@HZ",
                   help="every node's small stream, e.g. 640x480@30")
    p.add_argument("--small-format", choices=("y8", "jpeg"), default="y8")
    p.add_argument("--small-stagger", action="store_true",
                   help="spread the nodes' small frames evenly over the period (needs PTP)")
    p.add_argument("--clock-mode", choices=("estimate", "ptp"), default="estimate",
                   help="ptp: node and laptop share the PTP time scale, no offset is applied; "
                        "estimate: offset from the status round trip")
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
        for i, cid in enumerate(cams):
            if a.no_windows or (a.window_nodes and i >= a.window_nodes):
                source.request_windows(cid, [])
            else:
                source.request_windows(cid, windows)
    small = None
    if a.small:
        size, hz = a.small.split("@")
        sw, sh = (int(v) for v in size.split("x"))
        small = (sw, sh, float(hz))
        period_ns = int(round(1e9 / small[2]))
        for i, cid in enumerate(cams):
            phase = i * period_ns // len(cams) if a.small_stagger else 0
            source.stream_small(cid, sw, sh, small[2], phase, a.small_format)
        print(f"small stream {sw}x{sh} @ {small[2]:g} Hz ({a.small_format}), "
              f"{'staggered' if a.small_stagger else 'all nodes in phase'}", flush=True)

    # Before anything is timed: without this every cross-machine latency below
    # is off by the clock difference, and on an unsynchronised board that is
    # bigger than the latency itself.
    if a.clock_mode == "ptp":
        clock_offset = {cid: {"offset_ns": 0, "rtt_ns": 0, "uncertainty_ns": 0} for cid in cams}
    else:
        clock_offset = {cid: estimate_clock_offset(source, cid) for cid in cams}
    for cid, off in clock_offset.items():
        if off is None:
            print(f"  {cid}: clock offset UNKNOWN", flush=True)
        else:
            print(f"  {cid}: node clock {off['offset_ns'] / 1e6:+.2f} ms vs this laptop "
                  f"(+-{off['uncertainty_ns'] / 1e6:.2f} ms)", flush=True)

    per = {cid: {"window_latency_ns": [], "full_latency_ns": [], "node_latency_ns": [],
                 "net_latency_ns": [], "period_dev_ns": [], "stamps": [], "seqs": [],
                 "statuses": [], "window_bytes": 0, "full_bytes": 0,
                 "small_latency_ns": [], "small_bytes": 0, "small_stamps": [],
                 # The node's clock reads OFFSET ahead of ours, so its stamp is
                 # that much too large: add the offset back to compare with our
                 # receive time.
                 "offset_ns": ((clock_offset.get(cid) or {}).get("offset_ns") or 0)}
           for cid in cams}
    # The node's counters run from ITS start, not from ours, so a run that
    # follows a saturating one would inherit its drops.  Take a baseline and
    # report differences; ask for a status so we do not wait for the next tick.
    baseline = {}
    for cid in cams:
        try:
            source.request_status(cid, timeout=2.0)
        except (LanNotConnected, TimeoutError):
            pass
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and len(baseline) < len(cams):
        for cid in cams:
            status = source.status(cid)
            if cid not in baseline and status is not None:
                baseline[cid] = status
        if len(baseline) < len(cams):
            time.sleep(0.05)
    cpu = LaptopCpu()
    cpu_samples = []
    start = time.monotonic()
    next_full = start
    next_cpu = start + 1.0
    while time.monotonic() - start < a.seconds:
        now = time.monotonic()
        if not a.stream_full and a.full_period > 0 and now >= next_full:
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
        if small:
            for cid in cams:
                f = source.take_small(cid)
                if f is not None and cid in per:
                    d = per[cid]
                    d["small_latency_ns"].append(f.receive_ns - f.stamp_ns + d["offset_ns"])
                    d["small_bytes"] += f.payload_bytes
                    d["small_stamps"].append(f.stamp_ns)
        item = source.take_any(0.005 if small else 0.05)
        if item is None:
            continue
        cid, group = item
        d = per.get(cid)
        if d is None:
            continue
        for f in group:
            # Correct for the node/laptop clock difference; zero under PTP.
            latency = f.receive_ns - f.stamp_ns + d["offset_ns"]
            # Wire bytes, so a JPEG window counts its compressed size.
            if f.is_full:
                d["full_latency_ns"].append(latency)
                d["full_bytes"] += f.payload_bytes
            else:
                d["window_latency_ns"].append(latency)
                d["window_bytes"] += f.payload_bytes
            # Both ends of this one are the node's own clock: never corrected.
            d["node_latency_ns"].append(f.node_send_ns - f.stamp_ns)
            d["net_latency_ns"].append(f.receive_ns - f.node_send_ns + d["offset_ns"])
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
    if small:
        for cid in cams:
            try:
                source.stream_small(cid, hz=0, timeout=1.0)
            except (LanNotConnected, TimeoutError):
                pass
    source.close()
    elapsed = time.monotonic() - start

    report = {"seconds": elapsed, "windows": windows, "full_period_s": a.full_period,
              "window_nodes": a.window_nodes, "small": a.small, "small_stagger": a.small_stagger,
              "clock_mode": a.clock_mode,
              "stream_full_divisor": a.stream_full, "full_format": a.full_format,
              "laptop_cpu_percent": summary(cpu_samples), "clock_offset": clock_offset,
              "cameras": {}}
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
            "node_frames_missed": _delta(s_end, baseline.get(cid), "frames_missed"),
            "node_frames_dropped_queue": _delta(s_end, baseline.get(cid), "frames_dropped_queue"),
            "node_frames_missed_total": s_end.get("frames_missed"),
            "node_frames_dropped_queue_total": s_end.get("frames_dropped_queue"),
            "node_uptime_s": s_end.get("uptime_s"),
            "node_capture_to_send_ms": s_end.get("capture_to_send_ms"),
            "line_time_ns": s_end.get("line_time_ns"), "exposure_ns": s_end.get("exposure_ns"),
            "link": stats["links"],
        }
        if small:
            r = report["cameras"][cid]
            r["small_received"] = len(d["small_stamps"])
            r["small_fps"] = len(d["small_stamps"]) / elapsed if elapsed else None
            r["small_latency"] = summary(d["small_latency_ns"])
            r["small_mbit_s"] = d["small_bytes"] * 8 / elapsed / 1e6 if elapsed else None
            r["small_laptop_dropped"] = stats.get("small_dropped", {}).get(cid, 0)
            r["node_small_dropped"] = _delta(s_end, baseline.get(cid), "small_dropped")
            r["node_small_scale_ms"] = s_end.get("small_scale_ms")
            r["node_small_capture_to_send_ms"] = s_end.get("small_capture_to_send_ms")

    print()
    mode = (f"FULL frames every {a.stream_full} sensor frame(s) ({a.full_format})"
            if a.stream_full else f"windows {windows}, full every {a.full_period}s")
    print(f"=== lan_camera_timing: {elapsed:.1f}s, {mode} ===")
    print(f"laptop CPU: {fmt(report['laptop_cpu_percent'], 1, '%')}")
    for cid, off in report["clock_offset"].items():
        if off is None:
            print(f"  {cid}: clock offset UNKNOWN; cross-machine latencies are raw")
        else:
            print(f"  {cid}: node clock {off['offset_ns'] / 1e6:+.2f} ms vs this laptop "
                  f"(+-{off['uncertainty_ns'] / 1e6:.2f} ms); cross-machine latencies "
                  f"corrected for it, 'node exp->send' needs no correction")
    for cid, r in report["cameras"].items():
        print(f"\n[{cid}] frames {r['frames_received']} ({r['delivered_fps']:.1f}/s), "
              f"laptop drops {r['laptop_dropped']}, node missed {r['node_frames_missed']}, "
              f"node dropped {r['node_frames_dropped_queue']} "
              f"(node totals since its start: missed {r['node_frames_missed_total']}, "
              f"dropped {r['node_frames_dropped_queue_total']})")
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
        if small:
            sc, st = r["node_small_scale_ms"] or {}, r["node_small_capture_to_send_ms"] or {}
            print(f"  small stream     {r['small_received']} frames ({r['small_fps']:.1f}/s), "
                  f"{r['small_mbit_s']:.1f} Mbit/s, node skipped {r['node_small_dropped']}, "
                  f"laptop dropped {r['small_laptop_dropped']}")
            print(f"  small latency    {fmt(r['small_latency'], 1e6, 'ms')}")
            print(f"  node reduce      P50 {sc.get('p50') or float('nan'):.2f}ms P95 "
                  f"{sc.get('p95') or float('nan'):.2f}ms; exp->send P50 "
                  f"{st.get('p50') or float('nan'):.2f}ms P95 {st.get('p95') or float('nan'):.2f}ms")
        print(f"  line_time {r['line_time_ns']} ns, exposure {r['exposure_ns']} ns")
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nraw measurements: {a.json}")
    return report


if __name__ == "__main__":
    main()
