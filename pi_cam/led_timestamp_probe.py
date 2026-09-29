#!/usr/bin/env python3
"""Measure what SensorTimestamp means and the row time, with an LED on a GPIO.

Run on the node itself with the camera service stopped.  An LED on ``--gpio``
points into the lens (or lights a wall in view).  The probe fires a short
pulse at a time it records on CLOCK_BOOTTIME -- the clock of SensorTimestamp
-- then finds the lit band of rows in the frame that shows it.  Over
``--flashes`` flashes with random phase the band's last row against
``flash - SensorTimestamp`` is a straight line: the slope is ``line_time_ns``,
the intercept says whether SensorTimestamp is the exposure start of the first
row, the start of readout, or the end of the frame.  The result is what goes
into ``node_config.json`` as ``stamp_reference`` / ``stamp_correction_ns`` /
``line_time_ns`` and into the report.

``--blink-only`` just blinks the LED irregularly for the two-node test
(``scripts/lan_sync_check.py``).  ``--synthetic`` renders the flash into a
synthetic frame and checks the arithmetic without hardware.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pi_cam import flash_analysis  # noqa: E402
from pi_cam.camera_node import (  # noqa: E402
    NodeConfig, PicameraSensor, SyntheticSensor, boottime_ns, stamp_reference_correction_ns)

log = logging.getLogger("led_probe")
NS = 1_000_000_000


class Gpio:
    """One output pin through lgpio, RPi.GPIO or gpiozero, whichever is present."""

    def __init__(self, pin):
        self.pin = pin
        self.backend = None
        try:
            import lgpio
            self.h = lgpio.gpiochip_open(0)
            lgpio.gpio_claim_output(self.h, pin, 0)
            self._set = lambda v: lgpio.gpio_write(self.h, pin, v)
            self.backend = "lgpio"
            return
        except Exception:  # noqa: BLE001
            pass
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
            self._set = lambda v: GPIO.output(pin, v)
            self.backend = "RPi.GPIO"
            return
        except Exception:  # noqa: BLE001
            pass
        from gpiozero import LED
        led = LED(pin)
        self._set = lambda v: led.on() if v else led.off()
        self.backend = "gpiozero"

    def pulse(self, pulse_us):
        """Fire the LED; return (t_on, t_off) on CLOCK_BOOTTIME."""
        t_on = boottime_ns()
        self._set(1)
        end = t_on + pulse_us * 1000
        while boottime_ns() < end:
            pass
        self._set(0)
        t_off = boottime_ns()
        return t_on, t_off


class SyntheticFlashSensor(SyntheticSensor):
    """Frames with a rendered flash at a known BOOTTIME instant, for dry runs.

    Emulates a platform whose SensorTimestamp is ``reference``.
    """

    def __init__(self, cfg, line_time_ns=9452, reference="readout_start_first_row"):
        super().__init__(cfg)
        self.line_time_ns = line_time_ns
        self.reference = reference
        self.flash_ns = None
        self.columns = (self.width // 2 - 20, self.width // 2 + 20)

    def capture(self):
        frame = super().capture()
        # frame.sensor_stamp_ns is the exposure start of row 0 for the synthetic
        # sensor; shift it to emulate the platform's reference.
        t0 = frame.sensor_stamp_ns
        correction = stamp_reference_correction_ns(self.reference, frame.exposure_ns,
                                                   self.line_time_ns, self.height)
        frame.sensor_stamp_ns = t0 - correction
        # A flat scene with the flash band where the rolling shutter puts it.
        frame.y = flash_analysis.render_flash(
            self.height, self.width, t0, self.line_time_ns, frame.exposure_ns,
            self.flash_ns if self.flash_ns is not None else t0 - NS, self.columns)
        return frame


class FakeGpio:
    def __init__(self, sensor: SyntheticFlashSensor):
        self.sensor = sensor
        self.backend = "synthetic"

    def pulse(self, pulse_us):
        t_on = boottime_ns()
        self.sensor.flash_ns = t_on + pulse_us * 500
        return t_on, t_on + pulse_us * 1000


def find_flash(sensor, columns, threshold, max_frames=6):
    """Capture until a frame shows the band; return (frame_info, band) or None."""
    for _ in range(max_frames):
        with sensor.capture() as frame:
            band = flash_analysis.lit_band(frame.y, columns, threshold)
            info = {"sensor_stamp_ns": frame.sensor_stamp_ns, "exposure_ns": frame.exposure_ns,
                    "frame_duration_ns": frame.frame_duration_ns, "sequence": frame.sequence}
            height = frame.y.shape[0]
        if band is not None and not flash_analysis.band_is_truncated(band, height):
            return info, band
    return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="node_config.json for exposure/gain/mode")
    p.add_argument("--gpio", type=int, default=17, help="BCM pin driving the LED")
    p.add_argument("--flashes", type=int, default=50)
    p.add_argument("--pulse-us", type=int, default=100)
    p.add_argument("--columns", help="c0,c1 columns where the LED is (default: full width)")
    p.add_argument("--threshold", type=float)
    p.add_argument("--period-s", type=float, default=0.4, help="mean time between flashes")
    p.add_argument("--blink-only", action="store_true", help="only blink, for the two-node check")
    p.add_argument("--synthetic", action="store_true", help="dry run without camera or GPIO")
    p.add_argument("--synthetic-reference", default="readout_start_first_row")
    p.add_argument("--output", help="write the samples and the fit as JSON")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def run(a):
    cfg = NodeConfig.load(a.config)
    columns = tuple(int(v) for v in a.columns.split(",")) if a.columns else None
    if a.synthetic:
        cfg.sensor = "synthetic"
        # Narrow but full height: the readout must span the frame period for a
        # random-phase flash to land inside a frame, as it does on the IMX219.
        cfg.synthetic_width, cfg.synthetic_height = 328, 1232
        sensor = SyntheticFlashSensor(cfg, reference=a.synthetic_reference)
        columns = sensor.columns
        gpio = FakeGpio(sensor)
    else:
        gpio = Gpio(a.gpio)
        sensor = None if a.blink_only else PicameraSensor(cfg)
    log.info("gpio via %s", gpio.backend)
    if a.blink_only:
        log.info("blinking on GPIO %d every ~%.2f s, pulse %d us; Ctrl-C to stop", a.gpio, a.period_s, a.pulse_us)
        n = 0
        try:
            while True:
                time.sleep(random.uniform(0.5, 1.5) * a.period_s)
                t_on, t_off = gpio.pulse(a.pulse_us)
                n += 1
                print(json.dumps({"flash": n, "t_on_boottime_ns": t_on, "t_off_boottime_ns": t_off}), flush=True)
        except KeyboardInterrupt:
            return {"flashes": n}

    sensor.start()
    try:
        for _ in range(5):          # let exposure settle
            with sensor.capture():
                pass
        samples = []
        misses = 0
        period_s = 1.0 / cfg.fps
        for i in range(a.flashes):
            # Wait by consuming frames: the camera queue stays drained, so the
            # first frame after the pulse is the one that can show it, and
            # the phase of the flash inside the frame is random.
            for _ in range(int(random.uniform(0.5, 1.5) * a.period_s / period_s)):
                with sensor.capture():
                    pass
            time.sleep(random.uniform(0.0, period_s))
            t_on, t_off = gpio.pulse(a.pulse_us)
            found = find_flash(sensor, columns, a.threshold)
            if found is None:
                misses += 1
                log.warning("flash %d not seen", i)
                continue
            info, (first, last) = found
            sample = {**info, "flash_ns": (t_on + t_off) // 2, "t_on_ns": t_on, "t_off_ns": t_off,
                      "first_row": first, "last_row": last}
            samples.append(sample)
            log.info("flash %d: rows %d..%d, flash-stamp %.3f ms", i, first, last,
                     (sample["flash_ns"] - info["sensor_stamp_ns"]) / 1e6)
    finally:
        sensor.stop()
    if not samples:
        raise SystemExit("no flash was seen; check the LED position, --columns and exposure")
    fit = flash_analysis.fit_rolling_shutter(samples, sensor.height)
    model_lt = cfg.computed_line_time_ns()
    result = {
        "samples": samples, "misses": misses, "fit": fit,
        "line_time_model_ns": model_lt,
        "line_time_error_percent": (fit["line_time_ns"] / model_lt - 1) * 100 if "line_time_ns" in fit else None,
        "recommended_config": {
            "stamp_reference": fit["reference"] if fit["reference"] != "unknown" else "exposure_start_first_row",
            "stamp_correction_ns": int(round(fit["stamp_offset_ns"] - (
                stamp_reference_correction_ns(fit["reference"], fit["exposure_ns"],
                                              fit.get("line_time_ns", model_lt), sensor.height)
                if fit["reference"] != "unknown" else 0))),
            "line_time_ns": int(round(fit.get("line_time_ns", model_lt))),
        },
    }
    return result


def main(argv=None):
    a = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    result = run(a)
    if "fit" in result:
        fit = result["fit"]
        print(json.dumps({"fit": fit, "line_time_model_ns": result["line_time_model_ns"],
                          "line_time_error_percent": result["line_time_error_percent"],
                          "recommended_config": result["recommended_config"], "misses": result["misses"]},
                         indent=2))
    if a.output:
        Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    main()
