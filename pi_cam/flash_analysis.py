"""Rolling-shutter arithmetic for LED flash tests.

Row ``r`` of a rolling-shutter frame is exposed during::

    [T0 + r * line_time,  T0 + r * line_time + exposure]

where ``T0`` is the exposure start of the first row -- the ``stamp_ns`` the
node puts on every frame.  A short flash at ``t_f`` therefore lights a band of
rows ``r`` with ``T0 + r*lt <= t_f <= T0 + r*lt + exposure``:

    first lit row  ~ (t_f - T0 - exposure) / lt
    last lit row   ~ (t_f - T0) / lt
    band height    ~ exposure / lt

So from one frame with a known flash time we get the timestamp reference of
the platform, and from the band height or two flashes we get ``line_time``.
The same arithmetic, run on two nodes that saw one flash, gives their clock
difference: that is the synchronization error the report has to quote.

Pure numpy; imported on the node (LED probe) and on the laptop (sync check).
"""
from __future__ import annotations

import math

import numpy as np


def row_profile(image, columns=None):
    """Mean brightness per row, over ``columns`` = (c0, c1) or the full width."""
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    if columns is not None:
        c0, c1 = columns
        array = array[:, int(c0):int(c1)]
    return array.astype(np.float64).mean(axis=1)


def lit_band(image, columns=None, threshold=None, min_rows=1):
    """Return ``(first_row, last_row)`` of the flash band, or None.

    ``threshold`` defaults to halfway between the darkest and brightest row.
    Rows are counted in the coordinate frame of ``image``; add the window's
    ``row0`` yourself when the image is a window of the sensor frame.
    """
    profile = row_profile(image, columns)
    if profile.size == 0:
        return None
    low, high = float(profile.min()), float(profile.max())
    if threshold is None:
        if high - low < 8.0:      # no flash in this frame
            return None
        threshold = (low + high) / 2.0
    lit = np.flatnonzero(profile >= threshold)
    if lit.size < min_rows:
        return None
    # Take the largest contiguous run: reflections elsewhere must not stretch the band.
    breaks = np.flatnonzero(np.diff(lit) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [lit.size]))
    best = int(np.argmax(ends - starts))
    return int(lit[starts[best]]), int(lit[ends[best] - 1])


def band_is_truncated(band, height):
    """A band touching the first or last row continues outside the image.

    Its centre and its last row then say nothing about the flash, so such a
    frame is skipped; the flash usually shows whole in the neighbouring frame.
    """
    first, last = band
    return first <= 0 or last >= height - 1


def flash_time_ns(stamp_ns, line_time_ns, exposure_ns, first_row, last_row):
    """Best estimate of the flash instant from a lit band, on the frame's scale.

    Uses the band centre, which is insensitive to the brightness threshold:
    ``t_f = T0 + centre*lt + exposure/2``.
    """
    centre = (first_row + last_row) / 2.0
    return int(round(stamp_ns + centre * line_time_ns + exposure_ns / 2.0))


def line_time_from_band(exposure_ns, first_row, last_row):
    """Line time implied by the band height of one flash (needs exposure >> lt)."""
    rows = last_row - first_row + 1
    return exposure_ns / rows if rows > 0 else math.nan


def fit_rolling_shutter(samples, sensor_height=None):
    """Fit line time and the SensorTimestamp reference from LED flashes.

    ``samples``: iterable of dicts with ``flash_ns`` (known flash time on the
    node clock, CLOCK_BOOTTIME), ``sensor_stamp_ns`` (raw SensorTimestamp of
    the frame that shows it), ``first_row``, ``last_row`` (sensor rows),
    ``exposure_ns``.

    With ``T0 = sensor_stamp + d`` the last lit row obeys
    ``flash - sensor_stamp = d + lt * last_row``: a straight line whose slope
    is the line time and whose intercept says what SensorTimestamp is.
    """
    rows = np.array([s["last_row"] for s in samples], dtype=float)
    delta = np.array([s["flash_ns"] - s["sensor_stamp_ns"] for s in samples], dtype=float)
    exposures = np.array([s["exposure_ns"] for s in samples], dtype=float)
    heights = np.array([s["last_row"] - s["first_row"] + 1 for s in samples], dtype=float)
    n = len(rows)
    if n == 0:
        raise ValueError("no samples")
    result = {"samples": n}
    if n >= 2 and np.ptp(rows) > 0:
        slope, intercept = np.polyfit(rows, delta, 1)
        residual = delta - (intercept + slope * rows)
        result.update(line_time_ns=float(slope), stamp_offset_ns=float(intercept),
                      residual_p50_ns=float(np.percentile(np.abs(residual), 50)),
                      residual_p95_ns=float(np.percentile(np.abs(residual), 95)),
                      residual_max_ns=float(np.abs(residual).max()))
    band_lt = exposures / heights
    result["line_time_from_band_ns"] = float(np.median(band_lt))
    result["exposure_ns"] = float(np.median(exposures))
    lt = result.get("line_time_ns", result["line_time_from_band_ns"])
    if "stamp_offset_ns" not in result:
        # One flash: use the band centre and the band's own line time.
        centres = np.array([(s["first_row"] + s["last_row"]) / 2.0 for s in samples])
        result["stamp_offset_ns"] = float(np.median(delta - centres * lt - exposures / 2.0))
    result["reference"] = classify_reference(
        result["stamp_offset_ns"], result["exposure_ns"], lt, sensor_height)
    return result


def classify_reference(stamp_offset_ns, exposure_ns, line_time_ns, sensor_height=None,
                       tolerance_ns=300_000):
    """Name what SensorTimestamp marks, from ``T0 = SensorTimestamp + offset``."""
    candidates = {
        "exposure_start_first_row": 0.0,
        "readout_start_first_row": -float(exposure_ns),
    }
    if sensor_height:
        candidates["frame_end"] = -float(exposure_ns + sensor_height * line_time_ns)
    name, value = min(candidates.items(), key=lambda kv: abs(kv[1] - stamp_offset_ns))
    if abs(value - stamp_offset_ns) <= tolerance_ns:
        return name
    return "unknown"


def render_flash(height, width, stamp_ns, line_time_ns, exposure_ns, flash_ns,
                 columns=None, background=20, brightness=220, rng=None):
    """Synthetic rolling-shutter frame with one flash, for tests and dry runs."""
    image = np.full((height, width), background, dtype=np.uint8)
    if rng is not None:
        image = np.clip(image + rng.integers(-5, 6, size=image.shape), 0, 255).astype(np.uint8)
    starts = stamp_ns + np.arange(height) * float(line_time_ns)
    lit = (starts <= flash_ns) & (flash_ns <= starts + exposure_ns)
    c0, c1 = (0, width) if columns is None else columns
    image[lit, c0:c1] = brightness
    return image
