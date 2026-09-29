#!/usr/bin/env python3
"""Synchronization error between camera nodes from a shared LED flash.

Two (or more) nodes look at one LED that blinks briefly and irregularly
(``pi_cam/led_timestamp_probe.py --blink-only`` on any Pi, or any other
source).  Every node streams a window that contains the LED at the full
sensor rate.  For every frame that shows the flash, the flash instant on that
node's clock is ``stamp_ns + row * line_time_ns + exposure_ns / 2`` with
``row`` the centre of the lit band.  The difference between two nodes for one
flash is their synchronization error, PTP and timestamp reference included.
Repeated over ``--flashes`` flashes it gives the P50/P95/max the report needs.

Band heights also cross-check ``line_time_ns``: ``exposure / band_rows``
must agree with the header value to within a few percent.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from localization_contracts.lan_capture import LanCameraSource  # noqa: E402
from pi_cam.flash_analysis import (  # noqa: E402
    band_is_truncated, flash_time_ns, line_time_from_band, lit_band)


def parse_window(spec):
    r, c, w, h = (int(v) for v in spec.split(","))
    return r, c, w, h


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nodes", nargs="+", required=True)
    p.add_argument("--window", default="376,580,480,480",
                   help="row0,col0,w,h of the window that contains the LED, for every camera")
    p.add_argument("--camera-window", action="append", default=[],
                   help="camera_id=row0,col0,w,h to override the window per camera")
    p.add_argument("--columns", help="c0,c1 inside the window where the LED is (default: all)")
    p.add_argument("--threshold", type=float, help="row brightness threshold (default: adaptive)")
    p.add_argument("--flashes", type=int, default=50)
    p.add_argument("--seconds", type=float, default=120.0, help="give up after this long")
    p.add_argument("--pair-window-ms", type=float, default=8.0,
                   help="flashes closer than this on two cameras are the same flash")
    p.add_argument("--json")
    return p.parse_args(argv)


def pair_events(events_a, events_b, window_ns):
    """Greedy nearest pairing of two sorted event lists; returns (ta, tb) pairs."""
    pairs = []
    j = 0
    for ta, _ in events_a:
        while j < len(events_b) and events_b[j][0] < ta - window_ns:
            j += 1
        best = None
        for tb, _ in events_b[j:j + 3]:
            if abs(tb - ta) <= window_ns and (best is None or abs(tb - ta) < abs(best - ta)):
                best = tb
        if best is not None:
            pairs.append((ta, best))
    return pairs


def stats(values):
    if not values:
        return {"n": 0}
    a = np.abs(np.asarray(values, dtype=float))
    return {"n": int(a.size), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max()), "signed_median": float(np.median(values))}


def main(argv=None):
    a = parse_args(argv)
    default_window = parse_window(a.window)
    overrides = dict(item.split("=", 1) for item in a.camera_window)
    columns = tuple(int(v) for v in a.columns.split(",")) if a.columns else None
    source = LanCameraSource(a.nodes)
    cams = source.wait_connected(10.0)
    if len(cams) < 2:
        source.close()
        raise SystemExit(f"need at least two cameras, got {cams}")
    windows = {cid: parse_window(overrides[cid]) if cid in overrides else default_window for cid in cams}
    for cid in cams:
        source.request_windows(cid, [windows[cid] + ("y8",)])
    print(f"cameras {cams}, windows {windows}; waiting for {a.flashes} flashes", flush=True)

    events = {cid: [] for cid in cams}
    band_lt = {cid: [] for cid in cams}
    header_lt = {}
    start = time.monotonic()
    last_report = start
    while time.monotonic() - start < a.seconds:
        item = source.take_any(0.05)
        if item is None:
            continue
        cid, group = item
        if cid not in events:
            continue
        frame = group[0]
        header_lt[cid] = frame.line_time_ns
        band = lit_band(frame.array, columns, a.threshold)
        if band is None or band_is_truncated(band, frame.height):
            continue
        first, last = frame.row0 + band[0], frame.row0 + band[1]
        t = flash_time_ns(frame.stamp_ns, frame.line_time_ns, frame.exposure_ns, first, last)
        events[cid].append((t, {"frame_seq": frame.frame_seq, "first_row": first, "last_row": last,
                                "stamp_ns": frame.stamp_ns, "exposure_ns": frame.exposure_ns}))
        band_lt[cid].append(line_time_from_band(frame.exposure_ns, first, last))
        counts = {c: len(e) for c, e in events.items()}
        if min(counts.values()) >= a.flashes:
            break
        if time.monotonic() - last_report > 5.0:
            last_report = time.monotonic()
            print(f"flashes so far {counts}", flush=True)
    source.close()

    window_ns = int(a.pair_window_ms * 1e6)
    report = {"cameras": cams, "windows": windows, "flashes": {c: len(e) for c, e in events.items()},
              "line_time_header_ns": header_lt,
              "line_time_from_band_ns": {c: (float(np.median(v)) if v else None) for c, v in band_lt.items()},
              "pairs": {}}
    print()
    for ca, cb in itertools.combinations(cams, 2):
        pairs = pair_events(events[ca], events[cb], window_ns)
        diffs = [ta - tb for ta, tb in pairs]
        s = stats(diffs)
        report["pairs"][f"{ca}-{cb}"] = {**s, "diffs_ns": diffs}
        if s["n"]:
            print(f"{ca} - {cb}: n={s['n']} |dt| P50 {s['p50'] / 1e3:.1f} us  P95 {s['p95'] / 1e3:.1f} us  "
                  f"max {s['max'] / 1e3:.1f} us  signed median {s['signed_median'] / 1e3:+.1f} us")
        else:
            print(f"{ca} - {cb}: no paired flashes")
    for cid in cams:
        band = report["line_time_from_band_ns"][cid]
        head = header_lt.get(cid)
        if band and head:
            print(f"{cid}: line_time header {head} ns, from band {band:.0f} ns "
                  f"({(band / head - 1) * 100:+.1f} %), flashes {len(events[cid])}")
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    main()
