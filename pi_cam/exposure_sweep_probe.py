#!/usr/bin/env python3
"""Find what SensorTimestamp marks, using only the exposure control.

No LED, no GPIO, no wiring: run it on the node with the camera free.

A frame is delivered after its readout finishes.  Write the delay from the
frame's own timestamp to the moment ``capture_request()`` returns:

* if SensorTimestamp marks the **start of exposure** of the first row, that
  delay is ``exposure + readout + isp`` and grows one-for-one with exposure;
* if it marks the **start of readout** of the first row (the end of the first
  row's exposure), the delay is ``readout + isp`` and does not depend on the
  exposure at all.

So sweeping the exposure and fitting a line decides between them: slope 1
means exposure start, slope 0 means readout start.  Measured on a CM5 Lite
with IMX219, the delay held at 14.18 ms while the exposure went 0.5 -> 4 ms,
slope 0.000: SensorTimestamp excludes the exposure.

What this does NOT give: the absolute offset and ``line_time_ns``.  A
timestamp at the very end of the frame is also exposure-independent, and only
``led_timestamp_probe.py`` separates that case and measures the row period.
Use this probe first because it needs no hardware, then the LED probe.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pi_cam.camera_node import (  # noqa: E402
    NodeConfig, boottime_ns, make_sensor, stamp_reference_correction_ns)

log = logging.getLogger("exposure_sweep")

# Above this fraction of the frame period the pipeline starts stretching its
# own schedule and the point stops being comparable; measured on CM5 at 8 ms
# of a 12.04 ms frame, where the delay jumped 1.5 ms.
MAX_EXPOSURE_FRACTION = 0.5


def measure_one(sensor, exposure_us, frames, settle):
    """Return (actual_exposure_ms, min_delay_ms, p50_delay_ms) for one exposure.

    The statistic is the MINIMUM delay, not the median.  Everything that can go
    wrong between the frame being ready and this loop reading the clock only
    ever adds time: thread scheduling, another process on the core, the ISP
    queue.  The minimum over enough frames is therefore the closest thing to
    the pipeline's own latency, and unlike the median it does not drift with
    load.  On the CM5 the median moved 1.3 ms across a sweep where the minimum
    moved 0.02 ms, which was enough to fake a slope of 0.28.
    """
    sensor.set_controls(exposure_us=exposure_us)
    for _ in range(settle):                     # let the new exposure take effect
        sensor.capture().release()
    delays, actual = [], []
    for _ in range(frames):
        frame = sensor.capture()
        now = boottime_ns()
        delays.append((now - frame.sensor_stamp_ns) / 1e6)
        actual.append(frame.exposure_ns / 1e6)
        frame.release()
    return statistics.median(actual), min(delays), statistics.median(delays)


def sweep(sensor, exposures_us, frames=60, settle=25, passes=2):
    """Return one row per exposure: (asked_ms, actual_ms, delay_ms, p50_ms).

    The sweep runs forward and then backward and averages each exposure's two
    passes, so a slow drift over the run (the SoC warming up, for instance)
    cancels instead of tilting the line.
    """
    order = list(exposures_us)
    collected = {us: [] for us in order}
    actuals = {}
    for index in range(max(1, passes)):
        for exposure_us in (order if index % 2 == 0 else list(reversed(order))):
            actual, low, p50 = measure_one(sensor, exposure_us, frames, settle)
            collected[exposure_us].append((low, p50))
            actuals[exposure_us] = actual
            log.info("pass %d exposure asked %5.2f ms actual %5.2f ms  "
                     "stamp->delivery min %6.2f ms p50 %6.2f ms",
                     index + 1, exposure_us / 1000.0, actual, low, p50)
    rows = []
    for exposure_us in order:
        lows = [v[0] for v in collected[exposure_us]]
        p50s = [v[1] for v in collected[exposure_us]]
        rows.append((exposure_us / 1000.0, actuals[exposure_us],
                     sum(lows) / len(lows), sum(p50s) / len(p50s)))
        log.info("exposure %5.2f ms: delay min %6.3f ms (p50 %6.2f ms)",
                 rows[-1][0], rows[-1][2], rows[-1][3])
    return rows


def fit(rows, frame_duration_ms=None, tolerance=0.35):
    """Decide the reference from how much the delay moves with the exposure.

    The two hypotheses are a full slope apart: the delay either carries the
    exposure one-for-one (stamp before the exposure) or ignores it entirely
    (stamp after it).  Real sweeps are not perfectly flat -- on the CM5 the
    minimum delay stepped 0.7 ms between the 1 ms and 2 ms points while the
    exposure moved 3.5 ms -- so the test is which hypothesis the measured
    slope is near, with a band that leaves a clear gap between them.

    ``tolerance`` is that band; it must stay below 0.5 or the two hypotheses
    would overlap.
    """
    if not 0 < tolerance < 0.5:
        raise ValueError("tolerance must leave a gap between the two hypotheses")
    usable = rows
    if frame_duration_ms:
        limit = MAX_EXPOSURE_FRACTION * frame_duration_ms
        usable = [r for r in rows if r[1] <= limit] or rows
    xs = [r[1] for r in usable]      # actual exposure
    ys = [r[2] for r in usable]      # min delay, the load-insensitive statistic
    n = len(xs)
    if n < 2 or max(xs) == min(xs):
        raise ValueError("need at least two different exposures")
    mx, my = sum(xs) / n, sum(ys) / n
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
             / sum((x - mx) ** 2 for x in xs))
    intercept = my - slope * mx
    exposure_span = max(xs) - min(xs)
    delay_span = max(ys) - min(ys)
    near_readout = abs(slope) <= tolerance
    near_exposure = abs(slope - 1.0) <= tolerance
    reference = ("readout_start_first_row" if near_readout else
                 "exposure_start_first_row" if near_exposure else "unknown")
    return {
        "slope_ms_per_ms": slope, "intercept_ms": intercept,
        "reference": reference, "points_used": n, "points_dropped": len(rows) - n,
        "exposure_span_ms": exposure_span, "delay_span_ms": delay_span,
        # What the delay WOULD have moved if the stamp preceded the exposure.
        "delay_span_if_exposure_start_ms": exposure_span,
        "conclusive": near_readout != near_exposure,
        "verdict": ("the exposure is not in the timestamp"
                    if near_readout and not near_exposure else
                    "the timestamp precedes the exposure"
                    if near_exposure and not near_readout else
                    "inconclusive: the slope sits between the two hypotheses"),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="node_config.json; defaults to the built-in values")
    p.add_argument("--exposures-us", default="500,1000,2000,4000,8000")
    p.add_argument("--frames", type=int, default=60, help="frames measured per exposure")
    p.add_argument("--passes", type=int, default=2,
                   help="sweeps over the exposure list, alternating direction")
    p.add_argument("--synthetic", action="store_true", help="dry run without a camera")
    p.add_argument("--output", help="write the rows and the fit as JSON")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def run(a):
    cfg = NodeConfig.load(a.config)
    if a.synthetic:
        cfg.sensor = "synthetic"
        cfg.synthetic_width, cfg.synthetic_height = 64, 64
    sensor = make_sensor(cfg)
    sensor.start()
    try:
        exposures = [int(v) for v in a.exposures_us.split(",")]
        rows = sweep(sensor, exposures, frames=a.frames, passes=a.passes)
    finally:
        sensor.stop()
    period_ms = 1000.0 / cfg.fps
    result = {"rows": rows, "frame_duration_ms": period_ms, "fit": fit(rows, period_ms)}
    reference = result["fit"]["reference"]
    result["config_line"] = {
        "stamp_reference": reference,
        "correction_applied_ns": stamp_reference_correction_ns(
            reference, int(statistics.median([r[1] for r in rows]) * 1e6),
            cfg.computed_line_time_ns(), sensor.height),
    }
    return result


def main(argv=None):
    a = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    result = run(a)
    print(json.dumps({k: result[k] for k in ("fit", "config_line", "frame_duration_ms")}, indent=2))
    fit_result = result["fit"]
    print(f"exposure moved {fit_result['exposure_span_ms']:.2f} ms, delay moved "
          f"{fit_result['delay_span_ms']:.2f} ms -> {fit_result['verdict']}")
    if not fit_result["conclusive"]:
        print("WARNING: inconclusive. Widen --exposures-us, raise --frames/--passes, "
              "stop the camera service and other load, and check that AeEnable is off.",
              flush=True)
    if a.output:
        Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    main()
