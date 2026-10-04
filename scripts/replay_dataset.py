#!/usr/bin/env python3
"""Deterministic offline replay of a recorded dataset through the live tracker.

Reads a directory written by ``scripts/record_dataset.py`` and drives the same
contracts ``scripts/run_localization.py`` wires together -- detector, PnP
observer, ROI planner, asynchronous group buffer, IMM filter, lidar
continuation -- without Gazebo, ROS or a network.  The output directory has
the files ``run_localization.py`` writes, so ``scripts/evaluate_recording.py``
and ``scripts/conductor_check.py replay`` read it unchanged.

Time.  Everything runs on the dataset's simulated clock, never on wall time:

* a frame reaches the laptop ``--transport-ms`` after its render stamp, and
  a lidar scan ``--lidar-transport-ms`` after its scan stamp;
* the detector result reaches the filter ``--processing-ms`` after arrival
  (a fixed simulated cost, so the replay is reproducible; the measured wall
  cost of every frame is recorded separately in ``camera_frames.jsonl``);
* the ROI watchdog, the group window and the 200 Hz publication tick all read
  the same simulated clock.  The dataset clock does not start at zero.

In the output, every ``*wall_ns`` field carries that simulated delivery clock,
i.e. the replay models a laptop running at real time.  The actual wall clock of
this process is in ``replay_wall_ns`` and in ``timing.json``.

Parallelism.  Frames rendered at the same instant (all six cameras share one
render stamp) are processed together on a pool with one worker per camera,
as the six ``tag_worker`` threads do live.  The ROI plan for each frame is
taken from the filter before the pool starts, and the results are applied to
the filter in camera order afterwards, so the output does not depend on thread
scheduling: two runs of the same command give identical odometry.

Truth.  The runtime never reads truth.  The one exception is the operator's
rectangle around each rover: the replay projects the recorded cuboid of the
opponent, and of tag_rover, into the first frame that shows it and records
the box, standing in for a person drawing it before the match.  That is
``operator_box_from_truth`` and nothing else opens ``truth.jsonl``.  The
tag_rover box starts its track before the first marker and confirms the
identity for that one measurement only; the marker stays the only source that
confirms tag_rover afterwards, and yaw stays invalid until a marker gives it.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import heapq
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from localization_contracts import lidar_pipeline, roi_tracker  # noqa: E402
from localization_contracts.apriltag import marker_plane_z  # noqa: E402
from localization_contracts.camera_model import CameraModel  # noqa: E402
from localization_contracts.detector import PROFILES  # noqa: E402
from localization_contracts.marker_families import normalize_marker_family  # noqa: E402
from localization_contracts.rover_filter import (  # noqa: E402
    AsyncObservationBuffer, ImmRoverFilter, Measurement, OMEGA, POSITION, VX, VY, YAW, YAW_ONLY,
    measurement_from_observation, observation_stamp_ns,
)
from localization_contracts.contracts import Observation, SCHEMA_VERSION, FRAME_ARENA  # noqa: E402
from localization_contracts.identity import TwoRoverIdentity  # noqa: E402
from localization_contracts.opponent_camera import OpponentCamera, SILHOUETTE  # noqa: E402
from localization_contracts.camera_worker import POOLS, REACQUIRE_PERIOD_NS, scaled_camera  # noqa: E402
from localization_contracts.cuboid import localize_box  # noqa: E402
from localization_contracts.link_emulation import LinkModel, MODES as LINK_MODES  # noqa: E402
from localization_contracts.small_stream_codec import (  # noqa: E402
    CODECS as SMALL_CODECS, size_summary as small_size_summary)

# Opponent cuboid as the operator sees it: 0.9 x 0.52 m, top at 0.483 m.
OPPONENT_SIZE_M = (0.9, 0.52, 0.483)
OPERATOR_IDENTITY = "operator:opponent"
OPERATOR_TAG_IDENTITY = "operator:tag_rover"
# Only a blob this far from tag_rover may restart a lost opponent track.
REACQUIRE_CLEAR_M = 1.0
TAG_BODY_M = (0.72, 0.52, 0.40)
# Measured with scripts/bench_sam2.py on the RTX 3070 Laptop: P95 of one SAM2
# call, two cameras at 15 Hz each on their own phases, tiny, 512 px, bfloat16
# (report 08; artifacts/bench_08).
SAM2_DEFAULT_LATENCY_MS = 15.3
# Chosen on dataset_mounts5b_01 (report 08): P95 of the SAM2 readings' squared
# Mahalanobis error at scale 1 was 1.89 against 5.99 expected -> sqrt(1.89/5.99).
# Not changed for the other datasets.
SAM2_DEFAULT_SIGMA_SCALE = 0.55
# Tests replace the GPU engine with a stand-in (a callable returning one).
SAM2_ENGINE_FACTORY = None

# Event order at one instant: a measurement that arrives at t is visible to
# the tick at t, and the clock is advanced before anything reads it.
# RESULT is one camera's processed frame reaching the laptop under link
# emulation, where the arrival depends on what was sent.
# SAM2 is a SAM2 reading reaching the filter --sam2-latency-ms after its frame
# was processed.
CLOCK, ENQUEUE, RESULT, SAM2, LIDAR, CAMERA, TICK = range(7)
SAM2_MODES = ("off", "backup", "always")
# backup: the background path counts as unreliable when no silhouette was
# accepted for this long.
SAM2_SILHOUETTE_GAP_NS = 100_000_000
# Two SAM2 readings this close in time and space restart a lost opponent.
SAM2_REACQUIRE_NS = 300_000_000
SAM2_REACQUIRE_M = 0.3
SAM2_PROMPT_SILHOUETTE_NS = 50_000_000
SILHOUETTE_METHODS = ("silhouette_extent", "silhouette_reacquire", "operator_box")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path)
    p.add_argument("--config", type=Path,
                   help="runtime calibration; defaults to the dataset's runtime_cameras.json")
    p.add_argument("--output", type=Path, required=True,
                   help="new directory; an existing one is never overwritten")
    p.add_argument("--seconds", type=float, default=0.0,
                   help="replay only this many simulated seconds (0 = all)")
    p.add_argument("--cameras", nargs="+",
                   help="subset of camera names, space or comma separated")
    p.add_argument("--no-lidar", action="store_true")
    p.add_argument("--no-tag-operator-box", action="store_true",
                   help="start tag_rover only from its marker, without the operator's "
                        "rectangle (the second and last truth read)")
    p.add_argument("--transport-ms", type=float, default=15.0,
                   help="render stamp to arrival on the laptop, per frame (--link-emulation ideal)")
    p.add_argument("--link-emulation", choices=LINK_MODES, default="ideal",
                   help="ideal: every frame arrives --transport-ms after its stamp, pixels as "
                        "rendered.  cm5/cm4: what the camera node sends -- luminance (cm4: from "
                        "a demosaiced Bayer mosaic), JPEG windows encoded and decoded, and each "
                        "camera's windows arrive after the node's floor and work plus the "
                        "shared gigabit port (localization_contracts/link_emulation.py)")
    p.add_argument("--roi-jpeg-max-px", type=int, default=None,
                   help="marker windows above --roi-max-px go as JPEG up to this; beyond it a "
                        "full frame on the exhausted schedule.  Default 640 with cm4/cm5, "
                        "none (clamp) with ideal")
    p.add_argument("--opponent-jpeg-max-px", type=int, default=None,
                   help="the same for the opponent's windows above --opponent-roi-max-px")
    p.add_argument("--no-opponent-jpeg", action="store_true",
                   help="opponent windows stay lossless, clamped to --opponent-roi-max-px")
    p.add_argument("--opponent-max-cameras", type=int, default=0,
                   help="at most this many cameras get an opponent window per instant, the "
                        "nearest to the predicted opponent; 0 = no limit")
    p.add_argument("--opponent-stream", default=None, metavar="WxH@HZ",
                   help="the opponent reads a second stream, the frame resized to WxH at HZ "
                        "(e.g. 640x480@30); the marker keeps the full frames")
    p.add_argument("--opponent-stream-stagger", action="store_true",
                   help="spread the cameras' small frames evenly over the stream period")
    p.add_argument("--opponent-stream-codec", choices=SMALL_CODECS, default="y8",
                   help="how the node sends the small stream: y8 raw grey (default), jpeg "
                        "per frame, or h264 (one persistent x264 stream per camera emulating "
                        "the CM4 hardware encoder: no B-frames, zerolatency, VBR); needs "
                        "--opponent-stream and --link-emulation cm4/cm5")
    p.add_argument("--opponent-stream-bitrate", type=float, default=4.0, metavar="MBIT",
                   help="h264 target bitrate per camera, Mbit/s (ceiling 1.5x)")
    p.add_argument("--opponent-stream-gop-s", type=float, default=0.2,
                   help="h264 keyframe period, seconds (GOP = round(rate * this))")
    p.add_argument("--opponent-stream-jpeg-quality", type=int, default=90)
    p.add_argument("--opponent-stream-link-as-y8", action="store_true",
                   help="experiment: the codec changes the small frames' pixels, but the "
                        "link model still carries them as raw y8 (separates image quality "
                        "from the port-timing gain of a smaller stream)")
    p.add_argument("--full-stream-codec", choices=("none", "h264"), default="none",
                   help="h264: every camera sends ONLY its whole frame as H.264 (one persistent "
                        "x264 stream per camera, as --opponent-stream-codec h264); no marker "
                        "windows from the node, no separate small stream.  The laptop decodes, "
                        "cuts the marker windows out losslessly and reduces the frame for the "
                        "opponent (--opponent-stream WxH@HZ) like the node's scale_luma.  Only "
                        "frames on the --full-stream-fps grid exist; each reaches the tract "
                        "--full-stream-latency-ms + |N(0, jitter)| after its stamp.  Needs "
                        "--link-emulation cm4/cm5 (the sensor path); the port model is not used")
    p.add_argument("--full-stream-fps", type=float, default=50.0)
    p.add_argument("--full-stream-bitrate", type=float, default=15.0, metavar="MBIT")
    p.add_argument("--full-stream-gop-s", type=float, default=0.2)
    p.add_argument("--full-stream-latency-ms", type=float, default=68.0,
                   help="stamp to arrival in the tract, the fixed part (default with jitter "
                        "16.3: P50 79, P95 100 ms, as measured on a CM4)")
    p.add_argument("--full-stream-latency-jitter-ms", type=float, default=16.3,
                   help="sigma of the half-normal added to --full-stream-latency-ms")
    p.add_argument("--full-stream-seed", type=int, default=0)
    p.add_argument("--full-stream-color", action="store_true",
                   help="the full-frame stream in colour: the CM4 Bayer mosaic demosaiced to "
                        "colour, encoded as YUV420 H.264 at the same bitrate; the marker reads "
                        "the decoded Y plane, the opponent's background model and SAM2 the "
                        "decoded colour frame (background difference = largest per-channel "
                        "|frame - background|, same threshold)")
    p.add_argument("--tag-max-cameras", type=int, default=0,
                   help="at most this many cameras get a marker window per instant, the "
                        "nearest to the predicted tag_rover; 0 = no limit")
    p.add_argument("--acquire-full-frame-hz", type=float, default=0.0,
                   help="without a track, a full frame at most this often per camera; "
                        "0 = every frame")
    p.add_argument("--link-line-time-ns", type=float, default=9452.0,
                   help="IMX219 row time; with cm4/cm5 only reported (how much younger a "
                        "marker row is on hardware), never applied: Gazebo renders a "
                        "global shutter")
    p.add_argument("--lidar-transport-ms", type=float, default=None,
                   help="scan stamp to arrival; defaults to --transport-ms")
    p.add_argument("--line-time-ns", type=float, default=0.0,
                   help="rolling-shutter line time; 0 for Gazebo, about 9500 for IMX219")
    p.add_argument("--processing-ms", type=float, default=0.0,
                   help="simulated detector+PnP cost between arrival and the filter")
    p.add_argument("--workers", type=int, default=0,
                   help="camera worker threads with --parallel threads; 0 = one per camera")
    p.add_argument("--parallel", choices=sorted(POOLS), default="processes",
                   help="how the cameras run: one process each (default), one thread each, "
                        "or all inline on one thread; the output is identical")
    p.add_argument("--prefetch", action="store_true",
                   help="with --parallel processes: decode the next frame before it is asked "
                        "for (faster cameras, slower main thread on a power-limited laptop)")
    p.add_argument("--pin-cpus", action="store_true",
                   help="with --parallel processes: main thread alone on the fastest core, "
                        "camera processes on the others, never on the slowest cluster")
    # Everything below mirrors run_localization.py, same names and defaults.
    p.add_argument("--detector-scale", type=float, default=1.0)
    p.add_argument("--detector-profile", choices=PROFILES,
                   default="coverage", help="full-frame detector (acquire, watchdog)")
    p.add_argument("--roi-detector-profile", choices=PROFILES,
                   default=None, help="detector inside a window; default = --detector-profile")
    p.add_argument("--tag-quality-min", type=float, default=0.07)
    p.add_argument("--tag-max-reprojection-px", type=float, default=2.0)
    p.add_argument("--tag-max-planar-tilt-deg", type=float, default=40.0)
    p.add_argument("--tag-min-side-px", type=float, default=20.0)
    p.add_argument("--base-z-nominal", type=float, default=0.14)
    p.add_argument("--inverted-base-z", type=float, default=0.225)
    p.add_argument("--base-z-tolerance", type=float, default=0.25)
    p.add_argument("--max-incidence-deg", type=float, default=65.0)
    p.add_argument("--pnp-ray-disagreement", type=float, default=0.35)
    p.add_argument("--min-edge-distance-px", type=float, default=8.0)
    p.add_argument("--xy-source", choices=("ray", "pnp"), default="ray")
    p.add_argument("--publish-hz", type=float, default=200.0)
    p.add_argument("--group-window-ms", type=float, default=12.0)
    p.add_argument("--coast-ms", type=float, default=300.0)
    p.add_argument("--identity-max-age-s", type=float, default=2.0)
    p.add_argument("--lost-ms", type=float, default=1500.0)
    p.add_argument("--heading-after-ms", type=float, default=150.0,
                   help="velocity direction becomes a yaw measurement after this long "
                        "without a measured yaw; 0 = never")
    p.add_argument("--heading-min-speed", type=float, default=0.5)
    p.add_argument("--yaw-valid-s", type=float, default=1.0,
                   help="yaw_valid while the last marker heading is at most this old")
    p.add_argument("--lidar-axis-sigma-deg", type=float, default=8.0,
                   help="the lidar cluster's long axis as a yaw measurement; 0 = off")
    p.add_argument("--lidar-axis-min-elongation", type=float, default=1.5)
    p.add_argument("--heading-sigma-deg", type=float, default=8.0)
    p.add_argument("--identity-hold-max-s", type=float, default=10.0,
                   help="keep an identity older than --identity-max-age-s up to this long "
                        "while the lidar confirms the track every scan and no encounter "
                        "happened since the last confirmation; 0 = never")
    p.add_argument("--max-speed-mps", type=float, default=13.0)
    p.add_argument("--roi-min-px", type=int, default=160)
    p.add_argument("--roi-max-px", type=int, default=480)
    p.add_argument("--watchdog-period-s", type=float, default=2.0)
    p.add_argument("--no-roi-tracking", action="store_true")
    p.add_argument("--roi-exhausted-period-s", type=float, default=0.25,
                   help="after the window is exhausted, full frame at most this often; "
                        "0 = every frame (run_localization.py behaviour)")
    p.add_argument("--no-roi-visibility-gates", action="store_true",
                   help="plan a window even where the marker is beyond --max-incidence-deg "
                        "or smaller than --tag-min-side-px (run_localization.py behaviour)")
    p.add_argument("--lidar-z-band", type=float, nargs=2,
                   default=lidar_pipeline.DEFAULT_Z_BAND)
    p.add_argument("--lidar-max-radius", type=float,
                   default=lidar_pipeline.DEFAULT_MAX_USEFUL_RADIUS_M)
    p.add_argument("--lidar-max-extent", type=float,
                   default=lidar_pipeline.DEFAULT_MAX_EXTENT_M)
    p.add_argument("--lidar-min-points", type=int,
                   default=lidar_pipeline.DEFAULT_MIN_POINTS)
    p.add_argument("--lidar-sweep-s", type=float,
                   default=lidar_pipeline.DEFAULT_SWEEP_DURATION_S)
    p.add_argument("--lidar-background", default=None,
                   help="voxel map JSON; config/lidar_background.json when present")
    p.add_argument("--no-lidar-background", action="store_true")
    p.add_argument("--camera-background", type=Path, default=None,
                   help="empty-arena dataset directory; enables the opponent track "
                        "(camera background model + silhouettes + lidar)")
    p.add_argument("--no-opponent", action="store_true",
                   help="tag_rover only even when --camera-background is given")
    p.add_argument("--gain", type=float, default=1.0,
                   help="multiply every decoded frame by this (lighting robustness check)")
    p.add_argument("--opponent-size", type=float, nargs=3, default=(0.9, 0.52, 0.483),
                   metavar=("LENGTH", "WIDTH", "TOP"),
                   help="measured once before the match")
    p.add_argument("--opponent-roi-min-px", type=int, default=240)
    p.add_argument("--opponent-roi-max-px", type=int, default=900)
    p.add_argument("--opponent-max-incidence-deg", type=float, default=75.0)
    p.add_argument("--opponent-min-size-px", type=float, default=40.0)
    p.add_argument("--opponent-gate-m", type=float, default=0.8)
    p.add_argument("--opponent-lidar-max-z", type=float, default=0.55)
    p.add_argument("--opponent-lidar-slab", type=float, default=0.14,
                   help="top slab for the opponent cluster centre (cabin sits aft)")
    p.add_argument("--opponent-extent-plane", type=float, default=None,
                   help="height of the plane the silhouette extent is read on; "
                        "default half the body height")
    p.add_argument("--silhouette-along-sigma", type=float, default=0.20,
                   help="silhouette sigma along the camera ray = this * tan(incidence)")
    p.add_argument("--background-threshold", type=float, default=12.0)
    p.add_argument("--background-alpha", type=float, default=0.02)
    p.add_argument("--background-stride", type=int, default=3,
                   help="use every Nth frame of the empty-arena clip")
    p.add_argument("--identity-close-m", type=float, default=1.0)
    p.add_argument("--lidar-exclusion-margin", type=float, default=0.12,
                   help="margin around the other rover's body cut out of a lidar scan")
    p.add_argument("--lidar-exclusion-max-sigma", type=float, default=0.10,
                   help="cut the other rover out only while its track is this certain; "
                        "negative disables the cut")
    p.add_argument("--lidar-range-background", type=Path, default=None,
                   help="per-ray RangeBackground npz from scripts/build_lidar_background.py")
    # SAM2 as a second opinion on the opponent (localization_contracts/sam2_opponent.py).
    p.add_argument("--sam2-mode", choices=SAM2_MODES, default="off",
                   help="off: no SAM2.  backup: SAM2 runs (keeps its memory) but its reading "
                        "reaches the filter only while the background path is unreliable -- "
                        "rovers closer than --identity-close-m, no accepted silhouette for "
                        "100 ms, or the opponent track lost.  always: every reading")
    p.add_argument("--sam2-max-cameras", type=int, default=2,
                   help="SAM2 runs on this many cameras, the nearest to the predicted opponent")
    p.add_argument("--sam2-hz", type=float, default=15.0, help="SAM2 frames per camera per second")
    p.add_argument("--sam2-checkpoint", type=Path, default=ROOT / "models/sam2/sam2.1_hiera_tiny.pt")
    p.add_argument("--sam2-config", default=None,
                   help="sam2 config name; default by the checkpoint's file name")
    p.add_argument("--sam2-image-size", type=int, default=512,
                   help="square the small frame is resized to for SAM2 (published 1024)")
    p.add_argument("--sam2-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--sam2-latency-ms", type=float, default=None,
                   help="simulated delay from the frame's arrival to its SAM2 reading in the "
                        "filter, so the replay stays deterministic; default: the P95 measured "
                        "on the RTX 3070 Laptop (docs/dataset_tz/reports/08_sam2_report.md).  "
                        "Real GPU times go to timing.json")
    p.add_argument("--sam2-score-threshold", type=float, default=0.5)
    p.add_argument("--sam2-sigma-scale", type=float, default=None,
                   help="SAM2 reading covariance = this^2 * the silhouette observer's; fixed on "
                        "dataset_mounts5b_01")
    p.add_argument("--sam2-gate-m", type=float, default=None,
                   help="SAM2 reading farther than this from the opponent's prediction is "
                        "refused; default --opponent-gate-m (or 3 sigma, the larger)")
    p.add_argument("--opponent-blackout", default=None, metavar="S:E[,S:E]",
                   help="seconds from the start of the replay: the opponent's frames (background "
                        "path and SAM2) are replaced by the empty arena and the lidar gives no "
                        "opponent; afterwards no operator helps")
    a = p.parse_args(argv)
    if a.sam2_latency_ms is None:
        a.sam2_latency_ms = SAM2_DEFAULT_LATENCY_MS
    if a.sam2_sigma_scale is None:
        a.sam2_sigma_scale = SAM2_DEFAULT_SIGMA_SCALE
    a.blackouts = []
    if a.opponent_blackout:
        for item in a.opponent_blackout.split(","):
            start, end = (float(v) for v in item.split(":"))
            if end <= start or start < 0:
                p.error(f"bad blackout interval {item}")
            a.blackouts.append((start, end))
    if a.sam2_mode != "off":
        if not a.opponent_stream:
            p.error("--sam2-mode needs --opponent-stream (SAM2 reads the small frames)")
        if a.camera_background is None or a.no_opponent:
            p.error("--sam2-mode needs the opponent track (--camera-background)")
        if a.sam2_max_cameras < 1 or a.sam2_hz <= 0:
            p.error("--sam2-max-cameras and --sam2-hz must be positive")
    if a.opponent_stream_codec != "y8":
        if not a.opponent_stream or a.link_emulation == "ideal":
            p.error("--opponent-stream-codec needs --opponent-stream and --link-emulation cm4/cm5")
        if a.opponent_stream_bitrate <= 0 or a.opponent_stream_gop_s <= 0:
            p.error("--opponent-stream-bitrate and --opponent-stream-gop-s must be positive")
    if a.full_stream_codec != "none":
        if a.link_emulation == "ideal":
            p.error("--full-stream-codec needs --link-emulation cm4/cm5 (the sensor path)")
        if a.opponent_stream_codec != "y8":
            p.error("--full-stream-codec: the small frame comes from the decoded full frame; "
                    "drop --opponent-stream-codec")
        if a.full_stream_fps <= 0 or a.full_stream_bitrate <= 0 or a.full_stream_gop_s <= 0 \
                or a.full_stream_latency_ms < 0 or a.full_stream_latency_jitter_ms < 0:
            p.error("--full-stream-* must be positive")
    if a.full_stream_color and a.full_stream_codec == "none":
        p.error("--full-stream-color needs --full-stream-codec h264")
    if a.seconds < 0:
        p.error("--seconds must be nonnegative")
    if a.transport_ms < 0 or a.processing_ms < 0:
        p.error("delays must be nonnegative")
    if a.cameras:
        a.cameras = [name for item in a.cameras for name in item.split(",") if name]
    if a.lidar_transport_ms is None:
        a.lidar_transport_ms = a.transport_ms
    if a.link_emulation != "ideal":
        if a.line_time_ns:
            p.error("--line-time-ns with link emulation would move a global-shutter "
                    "render in time; the row time is --link-line-time-ns, reported only")
        if a.roi_jpeg_max_px is None:
            a.roi_jpeg_max_px = 640
        if a.opponent_jpeg_max_px is None and not a.no_opponent_jpeg:
            a.opponent_jpeg_max_px = 640
    if a.no_opponent_jpeg:
        a.opponent_jpeg_max_px = None
    for name in ("roi_jpeg_max_px", "opponent_jpeg_max_px"):
        cap = getattr(a, name)
        low = a.roi_max_px if name == "roi_jpeg_max_px" else a.opponent_roi_max_px
        if cap is not None and cap <= low:
            setattr(a, name, None)          # nothing above the lossless cap to compress
        if cap is not None and not a.roi_exhausted_period_s:
            p.error("a JPEG cap needs --roi-exhausted-period-s for the full frames above it")
    return a


def heading_kwargs(a):
    return dict(heading_after_ms=a.heading_after_ms, heading_min_speed_mps=a.heading_min_speed,
                heading_sigma_deg=a.heading_sigma_deg, marker_yaw_reset=True)


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summary(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    array = np.asarray(values)
    return {"n": len(values), "mean": float(array.mean()),
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)), "max": float(array.max())}


class FrameSource:
    """One sequential decoder per camera; frames are never held in memory."""

    def __init__(self, path):
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open {path}")
        self.next_index = 0

    def read(self, index, decode=True):
        """Advance to ``index``; return RGB when ``decode``, else None."""
        if index < self.next_index:
            raise ValueError(f"frame {index} requested after {self.next_index - 1}")
        while self.next_index < index:
            if not self.capture.grab():
                raise RuntimeError(f"video ends before frame {index}")
            self.next_index += 1
        self.next_index += 1
        if not decode:
            if not self.capture.grab():
                raise RuntimeError(f"video ends at frame {index}")
            return None
        ok, bgr = self.capture.read()
        if not ok:
            raise RuntimeError(f"video ends at frame {index}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self):
        self.capture.release()


def operator_box_from_truth(dataset, cams, models, stamps_by_camera, object_id="opponent",
                            size_m=OPPONENT_SIZE_M):
    """The operator's rectangle around one rover, from the recorded cuboid.

    The only truth read in the replay.  Returns the first render stamp at
    which the whole cuboid projects inside at least one camera's frame, with
    the box for every camera that shows it whole at that stamp.
    """
    rows = [r for r in read_jsonl(dataset / "truth.jsonl") if r["object_id"] == object_id]
    if not rows:
        return None
    rows.sort(key=lambda r: r["stamp_ns"])
    ts = np.array([r["stamp_ns"] for r in rows], dtype=np.int64)
    length, width, height = size_m
    body = np.array([[sx * length / 2, sy * width / 2, z]
                     for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, height)])
    stamps = sorted({s for values in stamps_by_camera.values() for s in values})
    for stamp in stamps:
        i = int(np.searchsorted(ts, stamp))
        if i <= 0 or i >= len(ts):
            continue
        a = (stamp - ts[i - 1]) / max(ts[i] - ts[i - 1], 1)
        lo, hi = rows[i - 1], rows[i]
        x = lo["x"] * (1 - a) + hi["x"] * a
        y = lo["y"] * (1 - a) + hi["y"] * a
        yaw = lo["yaw"] + a * math.remainder(hi["yaw"] - lo["yaw"], 2 * math.pi)
        c, s = math.cos(yaw), math.sin(yaw)
        world = body @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) + [x, y, 0.0]
        boxes = {}
        for cid, cam in cams.items():
            if stamp not in stamps_by_camera[cid]:
                continue
            R = np.asarray(cam["R_world_optical"], dtype=float)
            optical = (world - np.asarray(cam["position_world"], dtype=float)) @ R
            if (optical[:, 2] <= 1e-6).any():
                continue
            model = models[cid]
            uv = model.project(optical)
            if not (np.isfinite(uv).all() and model.inside_valid_radius(uv).all()):
                continue
            x0, y0 = uv.min(axis=0)
            x1, y1 = uv.max(axis=0)
            if x0 < 2 or y0 < 2 or x1 > model.width - 3 or y1 > model.height - 3:
                continue
            boxes[cid] = [round(float(v), 1) for v in (x0, y0, x1, y1)]
        if boxes:
            return {"stamp_ns": int(stamp),
                    "frame_index": {cid: stamps_by_camera[cid][stamp] for cid in boxes},
                    "boxes_xyxy_px": boxes,
                    "cuboid_m": list(size_m),
                    "source": "truth_cuboid_projection_operator_stand_in"}
    return None


class Replay:
    def __init__(self, a):
        self.a = a
        self.dataset = a.dataset
        config_path = a.config or (a.dataset / "runtime_cameras.json")
        cfg = json.loads(Path(config_path).read_text())
        if cfg.get("role") != "image_calibrated":
            raise ValueError("run image calibration first; nominal/truth files are not runtime calibration")
        cams = {c["name"]: c for c in cfg["cameras"]}
        if a.cameras:
            missing = set(a.cameras) - set(cams)
            if missing:
                raise ValueError(f"unknown cameras: {sorted(missing)}")
            cams = {cid: cams[cid] for cid in a.cameras}
        for camera in cams.values():
            CameraModel.from_config(camera).validate().raise_for_status()
        self.cfg, self.cams = cfg, cams
        self.version = cfg["calibration_version"]
        tag_entries = cfg["tags"]
        families = {normalize_marker_family(t["family"]) for t in tag_entries}
        sizes = {float(t["size"]) for t in tag_entries}
        if len(families) != 1 or len(sizes) != 1:
            raise ValueError("one runtime track must use one marker family and one marker size")
        self.marker_family, self.tag_size_m = families.pop(), sizes.pop()
        self.tags = {int(t["id"]): {"rotation": t["R_base_tag"],
                                    "translation": t["T_base_tag_translation"]}
                     for t in tag_entries}
        pose_gates = dict(
            tag_placement={int(t["id"]): t.get("placement", "top") for t in tag_entries},
            inverted_base_z_m=a.inverted_base_z, base_z_nominal_m=a.base_z_nominal,
            base_z_tolerance_m=a.base_z_tolerance, max_incidence_deg=a.max_incidence_deg,
            pnp_ray_disagreement_m=a.pnp_ray_disagreement,
            min_edge_distance_px=a.min_edge_distance_px, xy_source=a.xy_source,
            line_time_ns=a.line_time_ns,
        )
        self.pose_gates = pose_gates
        self.pipeline_kwargs = dict(
            family=self.marker_family, tag_size_m=self.tag_size_m,
            detector_scale=a.detector_scale, detector_profile=a.detector_profile,
            marker_ids=tuple(self.tags), roi_detector_profile=a.roi_detector_profile,
            quality_min=a.tag_quality_min, max_reprojection_px=a.tag_max_reprojection_px,
            max_planar_tilt_deg=a.tag_max_planar_tilt_deg, min_side_px=a.tag_min_side_px,
            **pose_gates)
        self.models = {cid: CameraModel.from_config(c) for cid, c in cams.items()}
        planes = [marker_plane_z(t.get("placement", "top"), float(t["T_base_tag_translation"][2]),
                                 a.base_z_nominal, a.inverted_base_z) for t in tag_entries]
        # The planners live in the camera workers; these are their settings.
        self.planner_kwargs = dict(
            min_roi_px=a.roi_min_px, max_roi_px=a.roi_max_px,
            marker_size_m=self.tag_size_m, watchdog_period_s=a.watchdog_period_s,
            max_incidence_deg=None if a.no_roi_visibility_gates else a.max_incidence_deg,
            min_marker_px=None if a.no_roi_visibility_gates else a.tag_min_side_px,
            exhausted_full_frame_period_s=a.roi_exhausted_period_s or None,
            tag_plane_z=(min(planes), max(planes)), jpeg_max_roi_px=a.roi_jpeg_max_px,
            acquire_period_s=1.0 / a.acquire_full_frame_hz if a.acquire_full_frame_hz else None)
        aliases = {f"{self.marker_family}:{int(i)}" for i in self.tags}
        if not a.no_tag_operator_box:
            aliases.add(OPERATOR_TAG_IDENTITY)
        self.filters = {"tag_rover": ImmRoverFilter(
            coast_ms=a.coast_ms, identity_max_age_s=a.identity_max_age_s,
            lost_ms=a.lost_ms, max_speed_mps=a.max_speed_mps, identity_aliases=aliases,
            identity_hold_max_s=a.identity_hold_max_s, **heading_kwargs(a))}
        self.opponent_enabled = a.camera_background is not None and not a.no_opponent
        self.guard = TwoRoverIdentity(close_m=a.identity_close_m)
        self.opponent_cameras = {}
        self.operator_box = None
        self.operator_done = False
        self.tag_operator_box = None
        self.reacquisitions = []
        self.reacquired_at = None
        self.reacquire_candidate = None
        self.tag_seated = False
        if self.opponent_enabled:
            # Same filter class and the same identity age limit.  The operator's
            # rectangle confirms this identity once; afterwards tag_rover's
            # marker seen well away from this track re-confirms it (enqueue),
            # and the identity hold covers the gaps while the lidar confirms.
            self.filters["opponent"] = ImmRoverFilter(
                coast_ms=a.coast_ms, identity_max_age_s=a.identity_max_age_s, lost_ms=a.lost_ms,
                max_speed_mps=a.max_speed_mps, identity_aliases={OPERATOR_IDENTITY},
                identity_hold_max_s=a.identity_hold_max_s, **heading_kwargs(a))
            # Same planner class, aimed at the body instead of the marker.
            self.opponent_planner_kwargs = dict(
                min_roi_px=a.opponent_roi_min_px, max_roi_px=a.opponent_roi_max_px,
                marker_size_m=a.opponent_size[0], watchdog_period_s=a.watchdog_period_s,
                tag_plane_z=(0.0, a.opponent_size[2]),
                max_incidence_deg=a.opponent_max_incidence_deg,
                min_marker_px=a.opponent_min_size_px,
                exhausted_full_frame_period_s=a.roi_exhausted_period_s or None,
                jpeg_max_roi_px=a.opponent_jpeg_max_px)
        self.buffers = {name: AsyncObservationBuffer(int(a.group_window_ms * 1e6))
                        for name in self.filters}
        self.pending_observations = {name: [] for name in self.filters}
        self.metrics = {cid: {"processed": 0, "tag_hits": 0, "tag_accepted": 0, "yolo_hits": 0,
                              "idle_frames": 0, "last_capture_ns": 0, "latency_ms": 0}
                        for cid in cams}
        self.lidar = self._make_lidar()
        self.lidar_rows_written = 0
        self.frame_rows = {}           # (camera_id, capture_ns) -> camera_frames row
        self.frame_order = []
        self.accepted_times = {name: collections.deque(maxlen=300) for name in self.filters}
        self.published_times = {name: collections.deque(maxlen=300) for name in self.filters}
        self.timing = {cid: {"latency_ms": [], "decode_ms": [], "idle_decode_ms": [],
                             "by_mode": collections.defaultdict(list),
                             "reasons": collections.Counter(), "tag_ms": [],
                             "opponent_ms": [], "background_update_ms": [],
                             "opponent_reasons": collections.Counter(),
                             "small_stream_bytes": [], "full_stream_bytes": [],
                             "full_stream_arrival_ms": [],
                             "stages": collections.defaultdict(list)} for cid in cams}
        self.batch_ms = []
        self.main_ms = {"plan": [], "apply": [], "publish": []}
        self.lidar_ms = []
        self.last_clock_ns = None
        self.errors = []
        # The full-frame stream has no node-side windows and no port model:
        # its frames arrive by the measured latency (build_events).
        self.full_stream = a.full_stream_codec != "none"
        self.link = (LinkModel(a.link_emulation)
                     if a.link_emulation != "ideal" and not self.full_stream else None)
        self.link_latency_ms = collections.defaultdict(list)
        self.link_counts = collections.Counter()
        self.link_row_ms = []
        self.full_frame_px = {cid: int(c["image_size"][0]) * int(c["image_size"][1])
                              for cid, c in cams.items()}
        # SAM2 (built in run(), after the camera pool, so no worker inherits CUDA).
        self.sam2 = None
        self.sam2_next_ns = {cid: 0 for cid in cams}
        self.sam2_anchor = None             # last known opponent XY: which cameras run SAM2
        self.sam2_candidate = None
        self.sam2_ms, self.sam2_cameras_per_call = [], []
        self.sam2_counts = collections.Counter()
        self.sam2_rows = []
        self.last_silhouette_ns = None      # last accepted background-path camera reading
        self.last_main_ns = None            # ... or lidar: what may aim a SAM2 prompt
        self.blackout_ns = []

    def _make_lidar(self):
        a, lidar_config = self.a, self.cfg.get("lidar")
        if a.no_lidar or not lidar_config:
            return None
        path = Path(a.lidar_background or (ROOT / "config/lidar_background.json"))
        background = None
        if not a.no_lidar_background and path.exists():
            background = lidar_pipeline.StaticVoxelMap.from_dict(json.loads(path.read_text()))
        elif a.lidar_background and not a.no_lidar_background:
            raise SystemExit(f"no lidar background at {path}")
        self.lidar_background_path = str(path) if background is not None else None
        range_background = None
        if a.lidar_range_background is not None:
            range_background = lidar_pipeline.RangeBackground.load(a.lidar_range_background)
        return lidar_pipeline.ArenaLidar(
            lidar_config["position_world"], lidar_config["R_world_sensor"],
            z_band=tuple(a.lidar_z_band), sweep_duration_s=a.lidar_sweep_s,
            max_useful_radius_m=a.lidar_max_radius, max_extent_m=a.lidar_max_extent,
            min_points=a.lidar_min_points, background=background,
            range_background=range_background)

    # ------------------------------------------------------------ filter side

    @staticmethod
    def state_back_at(f, at_ns):
        """The track at ``at_ns``, which may be older than the filter's own time.

        ``state_at`` never predicts backwards; a frame that arrives late
        (full-frame stream) is planned for its own stamp, so the position is
        taken back along the velocity.
        """
        state, covariance, mu = f.state_at(at_ns)
        if f.stamp_ns is not None and f.stamp_ns > at_ns:
            state = np.array(state, dtype=float)
            dt = (int(at_ns) - int(f.stamp_ns)) / 1e9
            state[0] += state[VX] * dt
            state[1] += state[VY] * dt
            state[YAW] += state[OMEGA] * dt
        return state, covariance, mu

    def track_prediction(self, now_ns, at_ns=None):
        """(x, y, sigma) of the tag_rover track, or None when there is none.

        ``at_ns``: the instant the position is for (default ``now_ns``).
        """
        f = self.filters["tag_rover"]
        if not f.initialized or f.tracking_state(now_ns) == "LOST":
            return None
        state, covariance, _ = (f.state_at(now_ns) if at_ns is None
                                else self.state_back_at(f, at_ns))
        return (float(state[0]), float(state[1]),
                float(math.sqrt(max(covariance[0, 0], covariance[1, 1]))))

    def record_observation(self, obs, accepted, reason, now_ns):
        self.out_obs.write(json.dumps({
            "observation": dataclasses.asdict(obs), "accepted": bool(accepted),
            "selection_reason": reason, "wall_ns": int(now_ns),
            "replay_wall_ns": time.monotonic_ns()}) + "\n")

    def enqueue(self, obs, now_ns):
        if obs.calibration_version != self.version or obs.capture_time_ns > now_ns + 100_000_000:
            self.record_observation(obs, False, "calibration_or_future_stamp", now_ns)
            return
        buffer = self.buffers.get(obs.object_id)
        if buffer is None:
            return
        features = obs.pixel_features or {}
        level = float(features.get("planar_tilt_deg", 90.0)) < 10.0
        # The ray assumes the marker is on its nominal plane, PnP does not; a
        # marker lifted with its rover makes them disagree by lift * tan(incidence).
        seated = (abs(float(features.get("pnp_base_z_m", 99.0))
                      - (self.a.inverted_base_z if features.get("base_inverted")
                         else self.a.base_z_nominal)) < 0.08
                  and float(features.get("pnp_ray_disagreement_m", 99.0)) < 0.08)
        if obs.object_id == "tag_rover":
            self.tag_seated = level and seated
        if obs.object_id == "tag_rover" and "opponent" in self.filters and level and seated:
            # Only a marker on a rover standing level on the floor votes: one
            # leaning on or lifted onto the other puts its marker off the plane
            # the ray is intersected with, by up to 0.3 m on dataset 02.
            tag, opp = self.pose_of("tag_rover", now_ns), self.pose_of("opponent", now_ns)
            if tag and opp and self.guard.marker_says_swap(
                    observation_stamp_ns(obs), obs.position_m[:2], tag[:2], opp[:2]):
                self.guard.swap(observation_stamp_ns(obs), self.filters["tag_rover"],
                                self.filters["opponent"], obs.position_m[:2])
            elif opp and math.hypot(obs.position_m[0] - opp[0], obs.position_m[1] - opp[1]) \
                    >= self.a.identity_close_m:
                # The marker names tag_rover; seen well away from the opponent
                # track it also says that track is not on tag_rover -- the only
                # identity evidence a rover without a marker ever gets.
                self.filters["opponent"].confirm_identity(observation_stamp_ns(obs))
        for measurement in measurement_from_observation(obs):
            buffer.push(measurement)
        self.pending_observations[obs.object_id].append(obs)

    def drain(self, now_ns, force=False):
        for name, buffer in self.buffers.items():
            for group in buffer.drain(now_ns, force):
                applied = self.filters[name].apply_group(group)
                if applied:
                    self.accepted_times[name].append(now_ns)
                # An observation is accepted when its position was: a heading
                # from the same camera and stamp says nothing about where.
                taken = {(m.source, m.stamp_ns) for m in applied if m.kind == POSITION}
                window = {m.stamp_ns for m in group}
                remaining = []
                for obs in self.pending_observations[name]:
                    stamp = observation_stamp_ns(obs)
                    if stamp not in window:
                        remaining.append(obs)
                        continue
                    source = f"sam2:{obs.camera_id}" if obs.method == "sam2" else obs.camera_id
                    ok = (source, stamp) in taken
                    self.record_observation(obs, ok, "fusion_accepted" if ok else "fusion_gate", now_ns)
                    if ok and name == "opponent" and obs.method != "sam2":
                        self.last_main_ns = max(self.last_main_ns or 0, stamp)
                        if obs.method in SILHOUETTE_METHODS:
                            self.last_silhouette_ns = max(self.last_silhouette_ns or 0, stamp)
                    if ok and obs.camera_id in self.metrics and name == "tag_rover":
                        self.metrics[obs.camera_id]["tag_accepted"] += 1
                        row = self.frame_rows.get((obs.camera_id, obs.capture_time_ns))
                        if row is not None:
                            row["fusion_accepted"] += 1
                self.pending_observations[name] = remaining

    def publish(self, now_ns):
        begin = time.perf_counter_ns()
        self._publish(now_ns)
        self.main_ms["publish"].append((time.perf_counter_ns() - begin) / 1e6)

    def _publish(self, now_ns):
        self.drain(now_ns)
        time_uncertain = self.last_clock_ns is None or now_ns - self.last_clock_ns > 200_000_000
        published = {}
        for name, f in self.filters.items():
            item = f.publish(now_ns)
            state = item["state"]
            self.published_times[name].append(now_ns)
            row = {**item, "object_id": name, "capture_ns": f.last_measurement_ns,
                   "wall_ns": int(now_ns), "replay_wall_ns": time.monotonic_ns(),
                   # tag_rover's yaw is vouched for while a marker fixed it
                   # within --yaw-valid-s; lidar axis and velocity heading keep
                   # it near (P95 13 deg on datasets 01/03 with cameras 1,3,5)
                   # but not to the 5 deg a marker gives.
                   "yaw_valid": (name == "tag_rover" and state is not None
                                 and (item.get("marker_yaw_age_ms") or 1e12)
                                 <= self.a.yaw_valid_s * 1000.0),
                   "source_mask": list(item["sources"]),
                   "measurement_wall_hz": self.rate(self.accepted_times[name]),
                   "output_wall_hz": self.rate(self.published_times[name]),
                   "session": 0, "out_of_sequence": f.out_of_sequence,
                   "dropped_too_old": f.too_old, "id_rejections": f.id_rejections}
            # evaluate_recording.py and conductor_check.py time every row by
            # state["stamp_ns"], which the filter's own state does not carry,
            # and cannot read a row whose state is None.  Before the marker
            # creates the track the state holds the stamp and nothing else,
            # so no position is invented for an invalid row.
            row["state"] = {**(state or {}), "stamp_ns": int(now_ns)}
            if time_uncertain:
                row.update(valid=False, tracking_state="TIME_UNCERTAIN")
            self.out_odom.write(json.dumps(row) + "\n")
            published[name] = (state["x"], state["y"]) if row["valid"] and state else None
        if "opponent" in self.filters:
            self.guard.observe_tracks(now_ns, published.get("tag_rover"), published.get("opponent"))
            tag, opp = published.get("tag_rover"), published.get("opponent")
            if tag and opp and math.hypot(tag[0] - opp[0], tag[1] - opp[1]) < self.a.identity_close_m:
                # An encounter: continuation can no longer tell the rovers
                # apart, so neither identity outlives its age limit until
                # something that can confirm it does.
                for f in self.filters.values():
                    f.revoke_identity_hold()

    @staticmethod
    def rate(stamps):
        return ((len(stamps) - 1) * 1e9 / (stamps[-1] - stamps[0])
                if len(stamps) > 1 and stamps[-1] > stamps[0] else 0.0)

    # ------------------------------------------------------------ camera side

    def worker_spec(self, cid):
        """Everything a CameraWorker needs, as plain data (it may live in another process)."""
        a = self.a
        opponent = None
        if self.opponent_enabled:
            opponent = {"background_dir": str(a.camera_background),
                        "stride": a.background_stride, "threshold": a.background_threshold,
                        "alpha": a.background_alpha, "size": tuple(a.opponent_size),
                        "tag_size": TAG_BODY_M, "gate_m": a.opponent_gate_m,
                        "extent_plane_z": a.opponent_extent_plane,
                        "along_sigma_scale": a.silhouette_along_sigma}
        return {"camera_id": cid, "camera": self.cams[cid], "dataset": str(self.dataset),
                "tags": self.tags, "calibration_version": self.version,
                "pipeline": self.pipeline_kwargs, "transport_ns": self.transport_ns,
                "planner": self.planner_kwargs,
                "opponent_planner": getattr(self, "opponent_planner_kwargs", None),
                "no_roi_tracking": a.no_roi_tracking,
                "processing_ns": self.processing_ns, "gain": a.gain, "opponent": opponent,
                "link": {"mode": a.link_emulation},
                "opponent_stream": (None if not a.opponent_stream else
                                    [*map(int, a.opponent_stream.split("@")[0].split("x")),
                                     float(a.opponent_stream.split("@")[1]),
                                     len(self.cams) if a.opponent_stream_stagger else 0]),
                "full_stream": ({"codec": a.full_stream_codec, "fps": a.full_stream_fps,
                                 "bitrate_mbit": a.full_stream_bitrate,
                                 "gop_s": a.full_stream_gop_s,
                                 "color": a.full_stream_color} if self.full_stream else None),
                "small_stream_codec": {"codec": a.opponent_stream_codec,
                                       "bitrate_mbit": a.opponent_stream_bitrate,
                                       "gop_s": a.opponent_stream_gop_s,
                                       "jpeg_quality": a.opponent_stream_jpeg_quality,
                                       "link_as_y8": a.opponent_stream_link_as_y8}}

    def pose_of(self, name, now_ns, at_ns=None):
        """(x, y, yaw, sigma) of a live track, or None."""
        f = self.filters.get(name)
        if f is None or not f.initialized or f.tracking_state(now_ns) == "LOST":
            return None
        state, covariance, _ = (f.state_at(now_ns) if at_ns is None
                                else self.state_back_at(f, at_ns))
        return (float(state[0]), float(state[1]), float(state[YAW]),
                float(math.sqrt(max(covariance[0, 0], covariance[1, 1]))))

    def frame_context(self, now_ns, items):
        """Main thread: what every camera needs from the filters for this instant.

        Only track poses and flags; each camera plans its own windows from them.
        Full-frame stream: the frame arrives after its stamp and is planned
        on arrival, for its stamp.
        """
        stamp = int(items[0][1]["stamp_ns"])
        at = stamp if self.full_stream else None
        tag = self.pose_of("tag_rover", now_ns, at) if self.opponent_enabled else None
        opp = self.pose_of("opponent", now_ns, at) if self.opponent_enabled else None
        operator = (self.opponent_enabled and self.operator_box is not None
                    and not self.operator_done and stamp == self.operator_box["stamp_ns"])
        common = {"now_ns": now_ns,
                  "prediction": (None if self.a.no_roi_tracking
                                 else self.track_prediction(now_ns, at)),
                  "opponent_enabled": self.opponent_enabled, "tag": tag, "opp": opp,
                  "operator_done": self.operator_done}
        allowed = None
        if opp is not None and self.a.opponent_max_cameras:
            # The nearest cameras to the predicted opponent see it steepest.
            ranked = sorted((math.hypot(self.cams[cid]["position_world"][0] - opp[0],
                                        self.cams[cid]["position_world"][1] - opp[1]), cid)
                            for cid, _ in items)
            allowed = {cid for _, cid in ranked[:self.a.opponent_max_cameras]}
        tag_allowed = None
        prediction = common["prediction"]
        if prediction is not None and self.a.tag_max_cameras:
            ranked = sorted((math.hypot(self.cams[cid]["position_world"][0] - prediction[0],
                                        self.cams[cid]["position_world"][1] - prediction[1]), cid)
                            for cid, _ in items)
            tag_allowed = {cid for _, cid in ranked[:self.a.tag_max_cameras]}
        sam2 = self.sam2_cameras(stamp, items, opp, operator)
        blackout = self.in_blackout(stamp)
        return {cid: {**common, "tag_allowed": tag_allowed is None or cid in tag_allowed,
                      "operator_box": (self.operator_box["boxes_xyxy_px"].get(cid)
                                                 if operator else None),
                      "opponent_allowed": allowed is None or cid in allowed,
                      "sam2_frame": cid in sam2, "blackout": blackout}
                for cid, _ in items}

    def in_blackout(self, stamp_ns):
        return any(lo <= stamp_ns < hi for lo, hi in self.blackout_ns)

    def sam2_cameras(self, stamp, items, opp, operator):
        """Cameras whose small frame of this instant goes to SAM2."""
        if self.sam2 is None:
            return set()
        if operator:
            # The operator's rectangle prompts every camera it was drawn on.
            return set(self.operator_box["boxes_xyxy_px"]) & {cid for cid, _ in items}
        if opp is not None:
            self.sam2_anchor = opp[:2]
        if self.sam2_anchor is None:
            return set()
        ranked = sorted((math.hypot(self.cams[cid]["position_world"][0] - self.sam2_anchor[0],
                                    self.cams[cid]["position_world"][1] - self.sam2_anchor[1]), cid)
                        for cid, _ in items)
        chosen = {cid for _, cid in ranked[:self.a.sam2_max_cameras]}
        return {cid for cid in chosen if stamp >= self.sam2_next_ns[cid]}

    def camera_batch(self, now_ns, items):
        if self.full_stream and len({int(row["stamp_ns"]) for _, row in items}) > 1:
            for item in items:                 # one plan per frame stamp
                self.camera_batch(now_ns, [item])
            return
        begin = time.perf_counter_ns()
        contexts = self.frame_context(now_ns, items)
        self.main_ms["plan"].append((time.perf_counter_ns() - begin) / 1e6)
        begin = time.perf_counter_ns()
        outputs = self.pool.run([(cid, row, contexts[cid]) for cid, row in items])
        results = [(cid, row, out) for (cid, row), out in zip(items, outputs)]
        self.batch_ms.append((time.perf_counter_ns() - begin) / 1e6)
        begin = time.perf_counter_ns()
        if self.link is not None:
            self.send_over_link(results)
            self.tag_operator_start(now_ns, items)
            self.main_ms["apply"].append((time.perf_counter_ns() - begin) / 1e6)
            return
        plans = {cid: result["plan"] for cid, _, result in results}
        jobs = {cid: result["job"] for cid, _, result in results if result["job"] is not None}
        self.apply_results(now_ns, plans, jobs, results)
        self.tag_operator_start(now_ns, items)
        self.main_ms["apply"].append((time.perf_counter_ns() - begin) / 1e6)

    def send_over_link(self, results):
        """Link emulation: each camera's result reaches the laptop when the model says.

        The frame was planned and read at its stamp; the node then needs its
        floor and its work on the windows, and the shared port serialises
        what every camera sends.  The result is applied on arrival (RESULT),
        with the arrival as its receive time.
        """
        jobs = [(cid, result["stamp"], result.get("link_windows") or [])
                for cid, _, result in results if not result["idle"]]
        served = self.link.serve(jobs)
        for cid, row, result in results:
            if result["idle"]:
                self.apply_results(int(row["stamp_ns"]), {cid: result["plan"]},
                                   {}, [(cid, row, result)])
                continue
            windows = result.get("link_windows") or []
            if cid not in served:           # nothing to send: busy but empty window
                arrival = result["stamp"] + self.link.floor_ns
            else:
                arrival = served[cid][1]
                self.link_latency_ms[cid].append((arrival - result["stamp"]) / 1e6)
                full = [w for w in windows if w[0] * w[1] >= self.full_frame_px[cid]]
                self.link_counts["full_frames"] += len(full)
                self.link_counts["jpeg_windows"] += sum(1 for w in windows if w[2] == "jpeg")
                self.link_counts["raw_windows"] += sum(1 for w in windows
                                                      if w[2] == "raw" and w not in full)
            processed = arrival + self.processing_ns
            result["received"], result["processed"] = arrival, processed
            result["observations"] = [dataclasses.replace(obs, receive_time_ns=arrival,
                                                          processed_time_ns=processed)
                                      for obs in result["observations"]]
            for obs in result["observations"]:
                row_px = (obs.pixel_features or {}).get("exposure_row_px")
                if row_px is not None:
                    self.link_row_ms.append(float(row_px) * self.a.link_line_time_ns / 1e6)
            self.push(arrival, RESULT, (cid, row, result))

    def tag_operator_start(self, now_ns, items):
        """Main thread: the operator's rectangle around tag_rover starts its track.

        Once, at the box's render stamp, before any marker: each camera's box
        becomes a base-centre XY through ``localize_box`` and goes to the
        tag_rover filter as the one measurement that confirms its identity
        without a marker.  It carries no yaw, so yaw stays invalid until the
        first marker.
        """
        box = self.tag_operator_box
        if box is None or box.get("done") or int(items[0][1]["stamp_ns"]) != box["stamp_ns"]:
            return
        box["done"] = True
        f = self.filters["tag_rover"]
        if f.initialized:
            return                      # a marker in this very batch got there first
        stamp = box["stamp_ns"]
        for cid, row in items:
            xyxy = box["boxes_xyxy_px"].get(cid)
            if xyxy is None:
                continue
            fit = localize_box(xyxy, self.cams[cid], dimensions=TAG_BODY_M,
                               camera_model=self.models[cid])
            x, y = fit["position_m"][:2]
            cov = (0.1 ** 2, 0.0, 0.0, 0.1 ** 2)
            obs = Observation(
                SCHEMA_VERSION, cid, int(row["index"]), f"{cid}:{int(row['index'])}:tag_operator",
                "tag_rover", stamp, "sim", 0, 0, int(now_ns), int(now_ns), self.version, FRAME_ARENA,
                (float(x), float(y), self.a.base_z_nominal),
                (cov[0], cov[1], 0.0, cov[2], cov[3], 0.0, 0.0, 0.0, 0.04),
                0.5, "operator_box", None, None, pose_6d_valid=False, attitude_state="unknown",
                pixel_features={"box_fit_rms_px": fit["box_fit_rms_px"], "reading": "box_fit"}
            ).validate()
            self.buffers["tag_rover"].push(Measurement(stamp, POSITION, (float(x), float(y)),
                                                       cov, cid, OPERATOR_TAG_IDENTITY, True,
                                                       obs.quality))
            self.pending_observations["tag_rover"].append(obs)

    def apply_results(self, now_ns, plans, jobs, results):
        for cid, row, result in results:
            timing = self.timing[cid]
            timing["reasons"][f"{plans[cid].mode}:{plans[cid].reason}"] += 1
            job = jobs.get(cid)
            if job is not None:
                timing["opponent_reasons"][f"{job['plan'].mode}:{job['plan'].reason}"] += 1
            if result["idle"]:
                self.metrics[cid]["idle_frames"] += 1
                timing["idle_decode_ms"].append(result["decode_ms"])
                continue
            plan = plans[cid]
            if plan.mode != roi_tracker.IDLE:
                timing["tag_ms"].append(result["tag_ms"])
            timing["latency_ms"].append(result["latency_ms"])
            for stage, value in result.get("stages", {}).items():
                timing["stages"][stage].append(value)
            timing["decode_ms"].append(result["decode_ms"])
            if result.get("full_stream_bytes") is not None:
                timing["full_stream_bytes"].append(result["full_stream_bytes"])
                timing["full_stream_arrival_ms"].append(
                    (int(row["arrival_ns"]) - result["stamp"]) / 1e6)
            if result.get("small_stream_bytes") is not None:
                timing["small_stream_bytes"].append(result["small_stream_bytes"])
            timing["by_mode"][plan.mode].append(result["latency_ms"])
            if job is not None and job["plan"].mode != roi_tracker.IDLE:
                timing["opponent_ms"].append(result["opponent_ms"])
            if job is not None:
                timing["background_update_ms"].append(result["background_update_ms"])
            m = self.metrics[cid]
            m["processed"] += 1
            m["tag_hits"] += len(result["hits"])
            m["last_capture_ns"] = result["stamp"]
            m["latency_ms"] = result["latency_ms"]
            reading = result.get("opponent_reading")
            frame = {
                "camera_id": cid, "sequence": int(row["index"]), "capture_ns": result["stamp"],
                "received_wall_ns": result["received"], "processed_wall_ns": result["processed"],
                "detections": len(result["hits"]),
                "tag_ids": [int(hit.tag_id) for hit in result["hits"]],
                "pnp_valid": len(result["observations"]), "fusion_accepted": 0,
                "best_quality": max(result["qualities"]) if result["qualities"] else None,
                "best_reprojection_px": min(result["reprojection"]) if result["reprojection"] else None,
                "latency_ms": result["latency_ms"], "tag_ms": result["tag_ms"],
                "opponent_ms": result["opponent_ms"],
                "background_update_ms": result["background_update_ms"],
                "decode_ms": result["decode_ms"],
                "mode": plan.mode, "plan_reason": plan.reason,
                "roi": list(plan.roi) if plan.roi else None,
                "pnp_rejections": result["rejections"],
                "pnp_diagnostics": result["diagnostics"],
                "opponent_mode": job["plan"].mode if job else None,
                "opponent_reason": job["plan"].reason if job else None,
                "opponent_roi": list(job["plan"].roi) if job and job["plan"].roi else None,
                "opponent_method": reading.method if reading else None,
                "opponent_diag": result.get("opponent_diag"),
            }
            self.frame_rows[(cid, result["stamp"])] = frame
            self.frame_order.append(frame)
            for obs in result["observations"]:
                if self.processing_ns:
                    self.push(result["processed"], ENQUEUE, ("obs", obs))
                else:
                    self.enqueue(obs, now_ns)
            if job is not None and (reading is not None or result.get("opponent_box_fit")):
                self.opponent_observation(cid, row, result, job, now_ns)
        if self.sam2 is not None:
            frames = [(cid, row, result) for cid, row, result in results
                      if result.get("sam2_image") is not None]
            if frames:
                self.sam2_step(now_ns, frames)

    def opponent_observation(self, cid, row, result, job, now_ns):
        """Main thread: one opponent reading -> Observation + filter measurements."""
        reading = result.get("opponent_reading")
        operator = job["plan"].reason == "operator_box"
        if operator:
            self.operator_done = True
        reacquire = bool(job.get("reacquire"))
        if reacquire:
            tag = self.pose_of("tag_rover", now_ns)
            f = self.filters["opponent"]
            if (reading is None or tag is None or self.pose_of("opponent", now_ns) is not None
                    or not self.tag_seated
                    or math.hypot(reading.x - tag[0], reading.y - tag[1]) < REACQUIRE_CLEAR_M):
                # A tag_rover lifted onto or leaning on something has a marker
                # off its plane and a track that may be anywhere nearby, so
                # "far from tag_rover" means nothing until it stands again.
                return
            # Two readings a reacquire period apart must agree before a lost
            # identity is handed to a blob.
            previous = self.reacquire_candidate
            self.reacquire_candidate = (result["stamp"], reading.x, reading.y)
            if (self.reacquired_at != result["stamp"] and (
                    previous is None or result["stamp"] - previous[0] < REACQUIRE_PERIOD_NS
                    or result["stamp"] - previous[0] > 3 * REACQUIRE_PERIOD_NS
                    or math.hypot(reading.x - previous[1], reading.y - previous[2]) > 0.3)):
                return
            if self.reacquired_at != result["stamp"]:
                f.reset()
            self.reacquired_at = result["stamp"]
            self.reacquisitions.append({"stamp_ns": int(result["stamp"]), "camera_id": cid,
                                        "xy": [reading.x, reading.y],
                                        "tag_distance_m": math.hypot(reading.x - tag[0],
                                                                     reading.y - tag[1])})
            operator = True
        if reading is not None:
            x, y, cov = reading.x, reading.y, reading.covariance_xy
            method = ("silhouette_reacquire" if reacquire
                      else "operator_box" if operator else reading.method)
            features = {"incidence_deg": math.degrees(reading.incidence_rad),
                        "pixels": reading.pixels, "reading": reading.method, **reading.detail}
        else:
            fit = result["opponent_box_fit"]
            x, y = fit["position_m"][:2]
            cov = (0.1 ** 2, 0.0, 0.0, 0.1 ** 2)
            method = "operator_box"
            features = {"box_fit_rms_px": fit["box_fit_rms_px"], "reading": "box_fit"}
        stamp = result["stamp"]
        sigma = math.sqrt(max(cov[0], cov[3]))
        obs = Observation(
            SCHEMA_VERSION, cid, int(row["index"]), f"{cid}:{int(row['index'])}:opponent",
            "opponent", stamp, "sim", 0, 0, result["received"], result["processed"],
            self.version, FRAME_ARENA, (float(x), float(y), self.a.base_z_nominal),
            (cov[0], cov[1], 0.0, cov[2], cov[3], 0.0, 0.0, 0.0, 0.04),
            max(0.0, min(1.0, 0.05 / (0.05 + sigma))), method, None, None,
            pose_6d_valid=False, attitude_state="unknown", pixel_features=features).validate()
        measurements = [Measurement(stamp, POSITION, (float(x), float(y)), tuple(cov), cid,
                                    OPERATOR_IDENTITY if operator else None, operator,
                                    obs.quality)]
        f = self.filters["opponent"]
        if reading is not None and reading.yaw_axis is not None and f.initialized \
                and reading.detail.get("length_m", 0) > 1.2 * reading.detail.get("width_m", 1):
            # The long axis says nothing about which end is the front; take
            # the end nearer the track's own heading.
            axis = reading.yaw_axis
            if abs(math.remainder(axis - float(f.x[YAW]), 2 * math.pi)) > math.pi / 2:
                axis = math.remainder(axis + math.pi, 2 * math.pi)
            measurements.append(Measurement(stamp, YAW_ONLY, (axis,), (math.radians(10) ** 2,),
                                            cid, None, False, obs.quality))
        for measurement in measurements:
            self.buffers["opponent"].push(measurement)
        self.pending_observations["opponent"].append(obs)

    # -------------------------------------------------------------- SAM2 side

    def build_sam2(self):
        """Engine on the GPU and the policy for every camera's small frame."""
        from localization_contracts.sam2_opponent import Sam2Engine, Sam2Opponent
        a = self.a
        w, h = (int(v) for v in a.opponent_stream.split("@")[0].split("x"))
        factory = SAM2_ENGINE_FACTORY or (lambda: Sam2Engine(
            a.sam2_checkpoint, a.sam2_config, dtype=a.sam2_dtype, image_size=a.sam2_image_size))
        self.sam2_engine = factory()
        cameras, self.sam2_scale = {}, {}
        for cid, cam in self.cams.items():
            small, scale = scaled_camera(cam, w, h)
            cameras[cid] = (CameraModel.from_config(small), cam["R_world_optical"],
                            cam["position_world"])
            self.sam2_scale[cid] = scale
        top = next((t for t in self.cfg["tags"] if t.get("placement", "top") == "top"),
                   self.cfg["tags"][0])
        offset = [float(v) for v in top["T_base_tag_translation"]]
        self.sam2 = Sam2Opponent(
            self.sam2_engine, cameras, size_m=a.opponent_size,
            marker=(offset[0], offset[1], offset[2] + a.base_z_nominal, self.tag_size_m),
            score_threshold=a.sam2_score_threshold, gate_m=a.sam2_gate_m or a.opponent_gate_m,
            clear_of_tag_m=REACQUIRE_CLEAR_M, sigma_scale=a.sam2_sigma_scale,
            along_sigma_scale=a.silhouette_along_sigma, extent_plane_z=a.opponent_extent_plane)
        self.sam2_period_ns = int(1e9 / a.sam2_hz)
        self.sam2_engine.reset_peak_memory()
        self.sam2_device = self.sam2_engine.device_name()

    def sam2_step(self, now_ns, frames):
        """Main thread: the small frames of this arrival through SAM2.

        The readings reach the filter --sam2-latency-ms later (SAM2 event);
        the refusals are written now.
        """
        stamp = int(frames[0][2]["stamp"])
        for cid, _, result in frames:
            # Next SAM2 frame of this camera one period on (less half a small
            # frame, so 30 -> 15 Hz takes every other one).
            self.sam2_next_ns[cid] = int(result["stamp"]) + self.sam2_period_ns - 10_000_000
        opp = self.pose_of("opponent", now_ns)
        tag = self.pose_of("tag_rover", now_ns)
        f = self.filters["opponent"]
        # A prompt from the prediction only where the background path itself
        # has just seen the opponent (a silhouette within 50 ms of this frame):
        # a box aimed by the prediction alone (lidar, coasting) at a frame that
        # does not show the opponent makes SAM2 lock onto the floor in the box
        # and follow that patch with high confidence (blackout run, report 08).
        may_prompt = (opp is not None and f.tracking_state(now_ns) in ("TRACKING", "COASTING")
                      and self.last_silhouette_ns is not None
                      and abs(stamp - self.last_silhouette_ns) <= SAM2_PROMPT_SILHOUETTE_NS)
        operator = {}
        if self.operator_box is not None and stamp == self.operator_box["stamp_ns"]:
            for cid, _, _ in frames:
                box = self.operator_box["boxes_xyxy_px"].get(cid)
                if box is not None:
                    sx, sy = self.sam2_scale[cid]
                    operator[cid] = (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)
        gate = None if opp is None else max(self.a.sam2_gate_m or self.a.opponent_gate_m,
                                            3.0 * opp[3])
        readings, refusals, steps, gpu_ms = self.sam2.process(
            [(cid, result["sam2_image"]) for cid, _, result in frames], now_ns,
            opp_prediction=opp, opp_gate_m=gate, tag_prediction=tag,
            may_prompt=may_prompt, operator_boxes=operator)
        if gpu_ms is not None:
            self.sam2_ms.append(gpu_ms)
            self.sam2_cameras_per_call.append(len(steps))
        rows = {cid: (row, result) for cid, row, result in frames}
        for cid, step in steps.items():
            self.sam2_counts["frames"] += 1
            self.sam2_counts["prompted"] += int(step.prompted)
            self.sam2_rows.append({"camera_id": cid, "stamp_ns": int(rows[cid][1]["stamp"]),
                                   "wall_ns": int(now_ns), "score": round(step.score, 4),
                                   "mask_px": int(step.mask.sum()), "prompted": step.prompted,
                                   "gpu_ms": round(step.gpu_ms, 3)})
        for refusal in refusals:
            self.sam2_counts[f"refused:{refusal.reason}"] += 1
            self.sam2_rows[-len(steps) + list(steps).index(refusal.camera_id)]["refused"] = refusal.reason
            row, result = rows[refusal.camera_id]
            self.out_obs.write(json.dumps({
                "observation": {"object_id": "opponent", "camera_id": refusal.camera_id,
                                "frame_seq": int(row["index"]), "method": "sam2",
                                "capture_time_ns": int(result["stamp"]),
                                "position_m": (None if refusal.x is None
                                               else [refusal.x, refusal.y, self.a.base_z_nominal]),
                                "pixel_features": {"score": refusal.score,
                                                   "prompted": refusal.prompted}},
                "accepted": False, "selection_reason": f"sam2_reject:{refusal.reason}",
                "wall_ns": int(now_ns), "replay_wall_ns": time.monotonic_ns()}) + "\n")
        delay = int(round(self.a.sam2_latency_ms * 1e6))
        for reading in readings:
            row, result = rows[reading.camera_id]
            self.push(now_ns + delay, SAM2, (row, result["stamp"], result["received"], reading))

    def main_path_unreliable(self, now_ns):
        """backup: does the background path need SAM2 right now?  (reason or None)"""
        opp = self.pose_of("opponent", now_ns)
        if opp is None:
            return "opponent_lost"
        tag = self.pose_of("tag_rover", now_ns)
        if tag is not None and math.hypot(tag[0] - opp[0], tag[1] - opp[1]) < self.a.identity_close_m:
            return "rovers_close"
        if self.last_silhouette_ns is None or now_ns - self.last_silhouette_ns > SAM2_SILHOUETTE_GAP_NS:
            return "no_silhouette"
        return None

    def sam2_apply(self, now_ns, row, stamp, received, reading):
        """A SAM2 reading reaches the filter (or, in backup, waits in reserve)."""
        cid = reading.camera_id
        reason = None
        if self.a.sam2_mode == "backup":
            reason = self.main_path_unreliable(now_ns)
            if reason is None:
                self.sam2_counts["standby"] += 1
                self.record_sam2(row, stamp, received, now_ns, reading, False, "sam2_standby")
                return
            self.sam2_counts[f"fed:{reason}"] += 1
        else:
            self.sam2_counts["fed:always"] += 1
        f = self.filters["opponent"]
        confirms = False
        if self.pose_of("opponent", now_ns) is None:
            # Lost: a SAM2 reading may restart the track like the background
            # path's reacquisition -- standing tag_rover well away (already
            # checked by the policy) and two readings that agree.
            tag = self.pose_of("tag_rover", now_ns)
            previous = self.sam2_candidate
            self.sam2_candidate = (int(stamp), cid, reading.x, reading.y)
            if (tag is None or not self.tag_seated
                    or math.hypot(reading.x - tag[0], reading.y - tag[1]) < REACQUIRE_CLEAR_M):
                self.record_sam2(row, stamp, received, now_ns, reading, False, "sam2_lost_near_tag")
                return
            if (previous is None or (previous[0], previous[1]) == (int(stamp), cid)
                    or not 0 <= int(stamp) - previous[0] <= SAM2_REACQUIRE_NS
                    or math.hypot(reading.x - previous[2], reading.y - previous[3]) > SAM2_REACQUIRE_M):
                self.record_sam2(row, stamp, received, now_ns, reading, False, "sam2_reacquire_wait")
                return
            f.reset()
            self.reacquisitions.append({"stamp_ns": int(stamp), "camera_id": cid, "source": "sam2",
                                        "xy": [reading.x, reading.y],
                                        "tag_distance_m": math.hypot(reading.x - tag[0],
                                                                     reading.y - tag[1])})
            confirms = True
        cov = reading.covariance_xy
        sigma = math.sqrt(max(cov[0], cov[3]))
        obs = self.sam2_observation(row, stamp, received, now_ns, reading, sigma)
        self.buffers["opponent"].push(Measurement(
            int(stamp), POSITION, (reading.x, reading.y), tuple(cov), f"sam2:{cid}",
            OPERATOR_IDENTITY if confirms else None, confirms, obs.quality))
        self.pending_observations["opponent"].append(obs)

    def sam2_observation(self, row, stamp, received, now_ns, reading, sigma):
        cov = reading.covariance_xy
        return Observation(
            SCHEMA_VERSION, reading.camera_id, int(row["index"]),
            f"{reading.camera_id}:{int(row['index'])}:sam2", "opponent", int(stamp), "sim", 0, 0,
            int(received), int(now_ns), self.version, FRAME_ARENA,
            (float(reading.x), float(reading.y), self.a.base_z_nominal),
            (cov[0], cov[1], 0.0, cov[2], cov[3], 0.0, 0.0, 0.0, 0.04),
            max(0.0, min(1.0, 0.05 / (0.05 + sigma))), "sam2", None, None,
            pose_6d_valid=False, attitude_state="unknown",
            pixel_features={"score": reading.score, "prompted": reading.prompted,
                            "incidence_deg": math.degrees(reading.incidence_rad),
                            "pixels": reading.pixels, **reading.detail}).validate()

    def record_sam2(self, row, stamp, received, now_ns, reading, accepted, reason):
        cov = reading.covariance_xy
        obs = self.sam2_observation(row, stamp, received, now_ns, reading,
                                    math.sqrt(max(cov[0], cov[3])))
        self.record_observation(obs, accepted, reason, now_ns)

    def sam2_report(self):
        if self.sam2 is None:
            return None
        ms = self.sam2_ms
        return {
            "mode": self.a.sam2_mode, "device": self.sam2_device,
            "checkpoint": str(self.a.sam2_checkpoint), "config": self.sam2_engine.config,
            "dtype": self.a.sam2_dtype, "image_size": self.sam2_engine.image_size,
            "memory_window_frames": self.sam2_engine.window,
            "max_cameras": self.a.sam2_max_cameras, "hz": self.a.sam2_hz,
            "latency_ms_simulated": self.a.sam2_latency_ms,
            "sigma_scale": self.a.sam2_sigma_scale,
            "sam2_ms": summary(ms),
            "sam2_ms_by_cameras": {str(n): summary([m for m, k in zip(ms, self.sam2_cameras_per_call)
                                                    if k == n])
                                   for n in sorted(set(self.sam2_cameras_per_call))},
            "vram_peak_mb": self.sam2_engine.peak_memory_mb(),
            "prompts": dict(self.sam2.prompts),
            "memory_frames_end": self.sam2.memory_frames(),
            "counts": dict(sorted(self.sam2_counts.items())),
            "note": "sam2_ms: wall time of one SAM2 call (encoder for every camera of the "
                    "instant + one decoder step each), device synchronised.  The filter sees "
                    "the reading latency_ms_simulated after arrival, not this time",
        }

    # ------------------------------------------------------------- lidar side

    def lidar_scan(self, now_ns, row):
        begin = time.perf_counter_ns()
        with np.load(self.dataset / row["file"]) as z:
            parsed = {"ranges": z["ranges"].astype(float), "azimuth": z["azimuth"].astype(float),
                      "elevation": z["elevation"].astype(float),
                      "range_min": float(z["range_min"]), "range_max": float(z["range_max"]),
                      "stamp_ns": int(row["stamp_ns"])}
        scan = None
        shape = {"tag_rover": {}, "opponent": {"max_z_m": self.a.opponent_lidar_max_z,
                                               "top_slab_m": self.a.opponent_lidar_slab}}
        bodies = {name: self.lidar_body(name, parsed["stamp_ns"]) for name in self.filters}
        for name, track in self.filters.items():
            state, covariance, _ = track.state_at(now_ns)
            tracking = track.tracking_state(parsed["stamp_ns"])
            out = {"object_id": name, "stamp_ns": parsed["stamp_ns"], "wall_ns": int(now_ns),
                   "replay_wall_ns": time.monotonic_ns(), "tracking_state": tracking}
            # The lidar continues a track; it never creates one, for either rover.
            if name == "opponent" and self.in_blackout(parsed["stamp_ns"]):
                out["reason"] = "blackout"
            elif state is None or not lidar_pipeline.may_continue(tracking):
                out["reason"] = "no_confirmed_track"
            else:
                lidar = self.lidar
                if scan is None:
                    scan = lidar.scan_to_arena(parsed)
                velocity = (float(state[2]), float(state[3]), 0.0)
                points = lidar.deskew(scan, velocity)
                sigma = math.sqrt(max(float(covariance[0, 0]), float(covariance[1, 1])))
                speed = math.hypot(velocity[0], velocity[1])
                before = dict(lidar.rejections)
                others = [body for other, body in bodies.items()
                          if other != name and body is not None]
                cluster = lidar.detect(points, (float(state[0]), float(state[1])),
                                       sigma, speed, parsed["stamp_ns"], **shape[name],
                                       exclude_boxes=others)
                if others:
                    out["excluded_bodies"] = [[round(float(v), 3) for v in b] for b in others]
                out.update(returns=scan.returns, rays=scan.rays,
                           prediction=[float(state[0]), float(state[1])],
                           prediction_sigma_m=sigma)
                if cluster is None:
                    new = [k for k, v in lidar.rejections.items() if v != before.get(k)]
                    out["reason"] = new[0] if new else "unknown"
                else:
                    out.update(reason="accepted", x=cluster.x, y=cluster.y, z_max=cluster.z_max,
                               points=cluster.points, extent_x=cluster.extent_x,
                               extent_y=cluster.extent_y, residual_m=cluster.residual_m,
                               sigma_m=cluster.sigma_m)
                    measurement = lidar_pipeline.measurement_from_cluster(cluster)
                    self.buffers[name].push(measurement)
                    if (self.a.lidar_axis_sigma_deg > 0 and cluster.axis_rad is not None
                            and cluster.elongation >= self.a.lidar_axis_min_elongation):
                        # The axis says nothing about which end is the front:
                        # take the end nearer the track's own heading.
                        axis = cluster.axis_rad
                        if abs(math.remainder(axis - float(state[YAW]), 2 * math.pi)) > math.pi / 2:
                            axis = math.remainder(axis + math.pi, 2 * math.pi)
                        self.buffers[name].push(Measurement(
                            int(cluster.stamp_ns), YAW_ONLY, (axis,),
                            (math.radians(self.a.lidar_axis_sigma_deg) ** 2,), "lidar"))
                    variance = cluster.sigma_m ** 2
                    self.pending_observations[name].append(Observation(
                        SCHEMA_VERSION, "lidar", int(row["file"].split("_")[-1].split(".")[0]),
                        f"lidar:{parsed['stamp_ns']}:{name}", name, parsed["stamp_ns"], "sim",
                        0, 0, int(now_ns), int(now_ns), self.version, FRAME_ARENA,
                        (cluster.x, cluster.y, self.a.base_z_nominal),
                        (variance, 0.0, 0.0, 0.0, variance, 0.0, 0.0, 0.0, 0.04),
                        measurement.quality, "lidar_cluster",
                        pixel_features={"points": cluster.points,
                                        "residual_m": cluster.residual_m}).validate())
            out["processing_ms"] = (time.perf_counter_ns() - begin) / 1e6
            self.out_lidar.write(json.dumps(out) + "\n")
            self.lidar_rows_written += 1
        self.lidar_ms.append((time.perf_counter_ns() - begin) / 1e6)

    def lidar_body(self, name, stamp_ns):
        """The body of rover ``name`` at ``stamp_ns`` as a lidar exclusion box, or None.

        Only for a published-valid track known to ``--lidar-exclusion-max-sigma``:
        cutting out a body where a wrong track puts it would delete the rover
        the other track is looking for.  A heading known worse than 15 degrees
        gives a square box of the body length.
        """
        a = self.a
        f = self.filters.get(name)
        if (a.lidar_exclusion_max_sigma < 0 or f is None or not f.initialized
                or f.tracking_state(stamp_ns) not in ("TRACKING", "COASTING")):
            return None
        state, covariance, _ = f.state_at(stamp_ns)
        if math.sqrt(max(covariance[0, 0], covariance[1, 1])) > a.lidar_exclusion_max_sigma:
            return None
        # A scan older than the filter's last measurement: back along the velocity.
        dt = (int(stamp_ns) - max(int(stamp_ns), int(f.stamp_ns))) / 1e9
        x = float(state[0] + state[2] * dt)
        y = float(state[1] + state[3] * dt)
        length, width = (TAG_BODY_M if name == "tag_rover" else a.opponent_size)[:2]
        if math.sqrt(max(covariance[YAW, YAW], 0.0)) > math.radians(15):
            width = length
        margin = 2 * a.lidar_exclusion_margin
        return (x, y, float(state[YAW]), length + margin, width + margin)

    # --------------------------------------------------------------- schedule

    def push(self, when_ns, kind, payload):
        heapq.heappush(self.queue, (int(when_ns), kind, self.sequence, payload))
        self.sequence += 1

    def build_events(self):
        a = self.a
        self.transport_ns = int(round(a.transport_ms * 1e6))
        self.lidar_transport_ns = int(round(a.lidar_transport_ms * 1e6))
        self.processing_ns = int(round(a.processing_ms * 1e6))
        self.tick_ns = int(round(1e9 / a.publish_hz))
        self.queue, self.sequence = [], 0
        index = {cid: read_jsonl(self.dataset / f"{cid}.jsonl") for cid in self.cams}
        self.t0 = min(int(rows[0]["stamp_ns"]) for rows in index.values() if rows)
        limit = self.t0 + int(a.seconds * 1e9) if a.seconds else None
        self.stamps_by_camera = {cid: {} for cid in self.cams}
        last = self.t0
        self.frames_scheduled = 0
        period_ns = int(round(1e9 / a.full_stream_fps)) if self.full_stream else 0
        for number, cid in enumerate(sorted(index)):
            rng = np.random.default_rng([a.full_stream_seed, number])
            next_ns = arrived_ns = None
            for row in index[cid]:
                stamp = int(row["stamp_ns"])
                if limit is not None and stamp >= limit:
                    break
                if self.full_stream:
                    # Only the frames on the stream's grid exist for the tract;
                    # each arrives after the node's encode, the network and the
                    # decode, in order (one stream per camera).
                    if next_ns is not None and stamp < next_ns:
                        continue
                    next_ns = (next_ns or stamp) + period_ns
                    while next_ns <= stamp:
                        next_ns += period_ns
                    delay = a.full_stream_latency_ms + abs(rng.normal(0.0, a.full_stream_latency_jitter_ms)) \
                        if a.full_stream_latency_jitter_ms else a.full_stream_latency_ms
                    arrival = max(stamp + int(round(delay * 1e6)), arrived_ns or 0)
                    arrived_ns = arrival
                    self.stamps_by_camera[cid][stamp] = int(row["index"])
                    self.push(arrival, CAMERA, (cid, {**row, "arrival_ns": arrival}))
                    self.frames_scheduled += 1
                    last = max(last, arrival)
                    continue
                self.stamps_by_camera[cid][stamp] = int(row["index"])
                # Under link emulation a frame is planned and read at its
                # stamp (the node has the window request before exposure) and
                # its result arrives later, by the link model.
                self.push(stamp + (0 if self.link is not None else self.transport_ns),
                          CAMERA, (cid, row))
                self.frames_scheduled += 1
                last = max(last, stamp + (100_000_000 if self.link is not None
                                          else self.transport_ns))
        self.scans_scheduled = 0
        if self.lidar is not None and (self.dataset / "lidar.jsonl").exists():
            for row in read_jsonl(self.dataset / "lidar.jsonl"):
                stamp = int(row["stamp_ns"])
                if stamp < self.t0 or (limit is not None and stamp >= limit):
                    continue
                self.push(stamp + self.lidar_transport_ns, LIDAR, row)
                self.scans_scheduled += 1
                last = max(last, stamp + self.lidar_transport_ns)
        for row in read_jsonl(self.dataset / "clock.jsonl"):
            stamp = int(row["sim_ns"])
            if limit is None or stamp < limit:
                self.push(stamp, CLOCK, stamp)
        self.end_ns = last + self.processing_ns + int(a.group_window_ms * 1e6) + self.tick_ns
        # The clock is live from the first frame on; the dataset's first
        # /clock row can trail the first render by a millisecond.
        self.last_clock_ns = self.t0
        self.push(self.t0, TICK, None)

    def run(self):
        a = self.a
        out = a.output
        out.mkdir(parents=True)
        (out / "initial_calibration.json").write_text(json.dumps(self.cfg, indent=2) + "\n")
        self.build_events()
        operator_box = operator_box_from_truth(self.dataset, self.cams, self.models,
                                               self.stamps_by_camera)
        tag_operator_box = None if a.no_tag_operator_box else operator_box_from_truth(
            self.dataset, self.cams, self.models, self.stamps_by_camera,
            object_id="tag_rover", size_m=TAG_BODY_M)
        self.write_parameters(out, operator_box, tag_operator_box)
        self.operator_box = operator_box
        self.tag_operator_box = dict(tag_operator_box) if tag_operator_box else None
        if self.opponent_enabled:
            self.build_opponent_cameras()
        started = time.monotonic()
        specs = [self.worker_spec(cid) for cid in sorted(self.cams)]
        if a.parallel == "threads":
            self.pool = POOLS["threads"](specs, a.workers or None)
        elif a.parallel == "processes":
            self.pool = POOLS["processes"](specs, prefetch=a.prefetch, pin=a.pin_cpus)
        else:
            self.pool = POOLS[a.parallel](specs)
        self.background_build_s = self.pool.background_build_s()
        self.pool_start_s = time.monotonic() - started
        self.blackout_ns = [(self.t0 + int(lo * 1e9), self.t0 + int(hi * 1e9))
                            for lo, hi in a.blackouts]
        if a.sam2_mode != "off":
            self.build_sam2()
        wall_start = time.monotonic()
        next_report = self.t0
        with (out / "observations.jsonl").open("w") as self.out_obs, \
                (out / "odometry.jsonl").open("w") as self.out_odom, \
                (out / "lidar.jsonl").open("w") as self.out_lidar:
            while self.queue:
                when, kind, _, payload = heapq.heappop(self.queue)
                if kind == CLOCK:
                    self.last_clock_ns = max(self.last_clock_ns or payload, payload)
                elif kind == ENQUEUE:
                    self.enqueue(payload[1], when)
                elif kind == RESULT:
                    cid, row, result = payload
                    job = result["job"]
                    self.apply_results(when, {cid: result["plan"]},
                                       {cid: job} if job is not None else {}, [(cid, row, result)])
                elif kind == SAM2:
                    self.sam2_apply(when, *payload)
                elif kind == LIDAR:
                    self.lidar_scan(when, payload)
                elif kind == CAMERA:
                    items = [payload]
                    while self.queue and self.queue[0][0] == when and self.queue[0][1] == CAMERA:
                        items.append(heapq.heappop(self.queue)[3])
                    self.camera_batch(when, sorted(items, key=lambda item: item[0]))
                elif kind == TICK:
                    self.publish(when)
                    if when + self.tick_ns <= self.end_ns:
                        self.push(when + self.tick_ns, TICK, None)
                if when >= next_report:
                    print(f"replay sim {(when - self.t0) / 1e9:6.2f} s  wall "
                          f"{time.monotonic() - wall_start:7.1f} s", file=sys.stderr, flush=True)
                    next_report += 2_000_000_000
            self.drain(self.end_ns, force=True)
        self.pool.close()
        wall = time.monotonic() - wall_start
        if self.sam2 is not None:
            with (out / "sam2.jsonl").open("w") as handle:
                for row in self.sam2_rows:
                    handle.write(json.dumps(row) + "\n")
        with (out / "camera_frames.jsonl").open("w") as handle:
            for row in self.frame_order:
                handle.write(json.dumps(row) + "\n")
        (out / "timing.json").write_text(json.dumps(self.timing_report(wall), indent=2) + "\n")
        status = self.status(wall)
        (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        return status

    def build_opponent_cameras(self):
        """Geometry of each camera for the main thread's exclusion windows.

        The background models live in the camera workers.
        """
        for cid, cam in self.cams.items():
            self.opponent_cameras[cid] = OpponentCamera(
                cid, self.models[cid], cam["R_world_optical"], cam["position_world"], None,
                size_m=self.a.opponent_size, tag_size_m=TAG_BODY_M,
                gate_m=self.a.opponent_gate_m)

    # ---------------------------------------------------------------- reports

    def write_parameters(self, out, operator_box, tag_operator_box=None):
        a = self.a
        (out / "runtime_parameters.json").write_text(json.dumps({
            "marker_family": self.marker_family, "marker_ids": sorted(self.tags),
            "tag_size_m": self.tag_size_m, "detector_scale": a.detector_scale,
            "detector_profile": a.detector_profile,
            "roi_detector_profile": a.roi_detector_profile or a.detector_profile,
            "tag_quality_min": a.tag_quality_min,
            "tag_max_reprojection_px": a.tag_max_reprojection_px,
            "tag_max_planar_tilt_deg": a.tag_max_planar_tilt_deg,
            "tag_min_side_px": a.tag_min_side_px, "base_z_nominal_m": a.base_z_nominal,
            "inverted_base_z_m": a.inverted_base_z, "base_z_tolerance_m": a.base_z_tolerance,
            "max_incidence_deg": a.max_incidence_deg,
            "pnp_ray_disagreement_m": a.pnp_ray_disagreement,
            "min_edge_distance_px": a.min_edge_distance_px, "xy_source": a.xy_source,
            "line_time_ns": a.line_time_ns,
            "publish_hz": a.publish_hz, "group_window_ms": a.group_window_ms,
            "coast_ms": a.coast_ms, "identity_max_age_s": a.identity_max_age_s,
            "lost_ms": a.lost_ms, "roi_min_px": a.roi_min_px, "roi_max_px": a.roi_max_px,
            "roi_tracking": not a.no_roi_tracking, "watchdog_period_s": a.watchdog_period_s,
            "roi_exhausted_period_s": a.roi_exhausted_period_s,
            "roi_visibility_gates": not a.no_roi_visibility_gates,
            "lidar_enabled": self.lidar is not None,
            "lidar_topic": (self.cfg.get("lidar") or {}).get("topic") if self.lidar else None,
            "lidar_max_radius_m": a.lidar_max_radius, "lidar_sweep_s": a.lidar_sweep_s,
            "lidar_background": getattr(self, "lidar_background_path", None),
            "lidar_range_background": (str(a.lidar_range_background)
                                       if a.lidar_range_background else None),
            "camera_policy": "asynchronous_group_window",
            "covariance_model": "ray_plane_anisotropic_v2",
            "opponent_enabled": self.opponent_enabled,
            "opponent": ({"size_m": list(a.opponent_size), "camera_background": str(a.camera_background),
                          "background_threshold": a.background_threshold,
                          "background_alpha": a.background_alpha,
                          "background_stride": a.background_stride,
                          "roi_px": [a.opponent_roi_min_px, a.opponent_roi_max_px],
                          "max_incidence_deg": a.opponent_max_incidence_deg,
                          "min_size_px": a.opponent_min_size_px, "gate_m": a.opponent_gate_m,
                          "lidar_max_z_m": a.opponent_lidar_max_z,
                          "lidar_top_slab_m": a.opponent_lidar_slab,
                          "identity_close_m": a.identity_close_m, "silhouette": "extent",
                          "extent_plane_z_m": (a.opponent_extent_plane
                                               if a.opponent_extent_plane is not None
                                               else a.opponent_size[2] / 2),
                          "lidar_exclusion_margin_m": a.lidar_exclusion_margin,
                          "lidar_exclusion_max_sigma_m": a.lidar_exclusion_max_sigma}
                         if self.opponent_enabled else None),
            "gain": a.gain,
            "replay": {
                "dataset": str(self.dataset), "cameras": sorted(self.cams),
                "seconds": a.seconds, "transport_ms": a.transport_ms,
                "lidar_transport_ms": a.lidar_transport_ms, "processing_ms": a.processing_ms,
                "workers": a.workers or len(self.cams), "clock": "dataset_sim",
                "sim_start_ns": self.t0,
                "wall_ns_fields": "simulated delivery clock (real-time laptop model); "
                                  "real process time in replay_wall_ns",
                "frames_scheduled": self.frames_scheduled,
                "scans_scheduled": self.scans_scheduled,
            },
            "opponent_operator_box": operator_box,
            "sam2": ({"mode": a.sam2_mode, "max_cameras": a.sam2_max_cameras, "hz": a.sam2_hz,
                      "checkpoint": str(a.sam2_checkpoint), "config": a.sam2_config,
                      "image_size": a.sam2_image_size, "dtype": a.sam2_dtype,
                      "latency_ms": a.sam2_latency_ms, "score_threshold": a.sam2_score_threshold,
                      "sigma_scale": a.sam2_sigma_scale, "gate_m": a.sam2_gate_m or a.opponent_gate_m}
                     if a.sam2_mode != "off" else None),
            "opponent_blackout_s": a.blackouts or None,
            "tag_rover_operator_box": tag_operator_box,
            "link_emulation": {
                "mode": a.link_emulation, "roi_max_px": a.roi_max_px,
                "roi_jpeg_max_px": a.roi_jpeg_max_px,
                "opponent_roi_max_px": a.opponent_roi_max_px,
                "opponent_jpeg_max_px": a.opponent_jpeg_max_px,
                "opponent_max_cameras": a.opponent_max_cameras,
                "tag_max_cameras": a.tag_max_cameras,
                "opponent_stream": a.opponent_stream,
                "opponent_stream_codec": a.opponent_stream_codec,
                "opponent_stream_bitrate_mbit": a.opponent_stream_bitrate,
                "opponent_stream_gop_s": a.opponent_stream_gop_s,
                "opponent_stream_jpeg_quality": a.opponent_stream_jpeg_quality,
                "opponent_stream_link_as_y8": a.opponent_stream_link_as_y8,
                "full_stream_codec": a.full_stream_codec,
                "full_stream_fps": a.full_stream_fps,
                "full_stream_bitrate_mbit": a.full_stream_bitrate,
                "full_stream_gop_s": a.full_stream_gop_s,
                "full_stream_latency_ms": a.full_stream_latency_ms,
                "full_stream_latency_jitter_ms": a.full_stream_latency_jitter_ms,
                "full_stream_seed": a.full_stream_seed,
                "full_stream_color": a.full_stream_color,
                "acquire_full_frame_hz": a.acquire_full_frame_hz,
                "transport_ms": a.transport_ms if a.link_emulation == "ideal" else None,
                "link_line_time_ns_reported_only": a.link_line_time_ns,
            },
        }, indent=2) + "\n")

    def timing_report(self, wall):
        per_camera, all_latency = {}, []
        processed = idle = 0
        for cid in sorted(self.timing):
            t = self.timing[cid]
            all_latency.extend(t["latency_ms"])
            processed += len(t["latency_ms"])
            idle += len(t["idle_decode_ms"])
            frames = len(t["latency_ms"]) + len(t["idle_decode_ms"])
            modes = collections.Counter({"idle": len(t["idle_decode_ms"])})
            for mode, values in t["by_mode"].items():
                modes[mode] += len(values)
            per_camera[cid] = {
                "processed_frames": len(t["latency_ms"]), "idle_frames": len(t["idle_decode_ms"]),
                "mode_share": {m: round(n / max(frames, 1), 4) for m, n in sorted(modes.items())},
                "plan_reasons": dict(sorted(t["reasons"].items())),
                "opponent_plan_reasons": dict(sorted(t["opponent_reasons"].items())),
                "tag_ms": summary(t["tag_ms"]), "opponent_ms": summary(t["opponent_ms"]),
                "latency_ms": summary(t["latency_ms"]),
                "latency_ms_by_mode": {k: summary(v) for k, v in sorted(t["by_mode"].items())},
                "decode_ms": summary(t["decode_ms"] + t["idle_decode_ms"]),
                "stages_ms": {k: summary(v) for k, v in sorted(t["stages"].items())},
                "small_stream": self.small_stream_summary(t["small_stream_bytes"]),
            }
        sim_seconds = (self.end_ns - self.t0) / 1e9
        frames = processed + idle
        stages = collections.defaultdict(list)
        for t in self.timing.values():
            for stage, values in t["stages"].items():
                stages[stage].extend(values)
        stages["decode_ms"] = [v for t in self.timing.values() for v in t["decode_ms"]]
        stages["decode_idle_ms"] = [v for t in self.timing.values() for v in t["idle_decode_ms"]]
        stages["main_plan_ms"] = self.main_ms["plan"]
        stages["main_apply_ms"] = self.main_ms["apply"]
        stages["main_filter_publish_ms"] = self.main_ms["publish"]
        return {
            "note": "latency_ms is wall time of detector + PnP + opponent per frame; decoding "
                    "is excluded.  stages_ms: per busy frame (camera side) or per call "
                    "(main_*: plan = ROI and opponent plans for one instant, apply = results "
                    "into the filters for one instant, filter_publish = one 200 Hz tick)",
            "parallel": self.a.parallel,
            "workers": (self.a.workers or len(self.cams)) if self.a.parallel == "threads"
            else (len(self.cams) if self.a.parallel == "processes" else 1),
            "pool_start_s": getattr(self, "pool_start_s", None),
            "prefetch": self.a.parallel == "processes" and self.a.prefetch,
            "cpu_plan": getattr(self.pool, "cpu_plan", None),
            "stages_ms": {k: summary(v) for k, v in sorted(stages.items())},
            "wall_seconds": wall, "sim_seconds": sim_seconds,
            "frames_total": frames, "frames_processed": processed, "frames_idle": idle,
            "fps_total": frames / max(wall, 1e-9),
            "fps_processed": processed / max(wall, 1e-9),
            "realtime_ratio": sim_seconds / max(wall, 1e-9),
            "latency_ms_all_cameras": summary(all_latency),
            "tag_ms_all_cameras": summary(
                [v for t in self.timing.values() for v in t["tag_ms"]]),
            "opponent_ms_all_cameras": summary(
                [v for t in self.timing.values() for v in t["opponent_ms"]]),
            "background_update_ms_all_cameras": summary(
                [v for t in self.timing.values() for v in t["background_update_ms"]]),
            "background_build_s": getattr(self, "background_build_s", None),
            "batch_wall_ms": summary(self.batch_ms),
            "lidar_processing_ms": summary(self.lidar_ms),
            "jpeg_decode_ms": summary([v for t in self.timing.values()
                                       for v in t["stages"].get("jpeg_decode_ms", [])]),
            "link": self.link_report(),
            "full_stream": ({"codec": self.a.full_stream_codec, "fps": self.a.full_stream_fps,
                             "color": self.a.full_stream_color,
                             "bitrate_target_mbit": self.a.full_stream_bitrate,
                             "gop_s": self.a.full_stream_gop_s,
                             "latency_model_ms": [self.a.full_stream_latency_ms,
                                                  self.a.full_stream_latency_jitter_ms],
                             "arrival_ms": summary([v for t in self.timing.values()
                                                    for v in t["full_stream_arrival_ms"]]),
                             "all_cameras": small_size_summary(
                                 [v for t in self.timing.values() for v in t["full_stream_bytes"]],
                                 self.a.full_stream_fps),
                             "cameras": {cid: small_size_summary(t["full_stream_bytes"],
                                                                 self.a.full_stream_fps)
                                         for cid, t in sorted(self.timing.items())}}
                            if self.full_stream else None),
            "small_stream": ({"codec": self.a.opponent_stream_codec,
                              "bitrate_target_mbit": self.a.opponent_stream_bitrate
                              if self.a.opponent_stream_codec == "h264" else None,
                              "gop_s": self.a.opponent_stream_gop_s
                              if self.a.opponent_stream_codec == "h264" else None,
                              "jpeg_quality": self.a.opponent_stream_jpeg_quality
                              if self.a.opponent_stream_codec == "jpeg" else None,
                              "all_cameras": self.small_stream_summary(
                                  [v for t in self.timing.values()
                                   for v in t["small_stream_bytes"]])}
                             if self.a.opponent_stream else None),
            "sam2": self.sam2_report(),
            "cameras": per_camera,
        }

    def small_stream_summary(self, sizes):
        """Encoded size of the small frames sent (link emulation only)."""
        if not self.a.opponent_stream:
            return None
        return small_size_summary(sizes, float(self.a.opponent_stream.split("@")[1]))

    def link_report(self):
        """Port load and arrival times under link emulation; None for ideal."""
        if self.link is None:
            return None
        report = self.link.report(self.t0, self.end_ns)
        all_latency = [v for values in self.link_latency_ms.values() for v in values]
        report.update({
            "note": "series_mbps: port load per 100 ms bin, lidar's constant share included. "
                    "exposure_to_arrival_ms: first-row exposure to the last window of the "
                    "frame on the laptop (node floor + work + port queue + wire). "
                    "marker_row_ms: how much later than the frame stamp the marker's row "
                    "is exposed on a rolling-shutter IMX219 -- reported only, not applied "
                    "(Gazebo renders a global shutter)",
            "exposure_to_arrival_ms": summary(all_latency),
            "exposure_to_arrival_ms_by_camera": {cid: summary(v) for cid, v
                                                 in sorted(self.link_latency_ms.items())},
            "marker_row_ms": summary(self.link_row_ms),
            "full_frames": self.link_counts["full_frames"],
            "jpeg_windows": self.link_counts["jpeg_windows"],
            "raw_windows": self.link_counts["raw_windows"],
            "jpeg_window_share": self.link_counts["jpeg_windows"] / max(
                1, self.link_counts["jpeg_windows"] + self.link_counts["raw_windows"]),
        })
        return report

    def status(self, wall):
        f = self.filters["tag_rover"]
        return {
            "phase": "REPLAY", "hardware_verified": False,
            "opponent_enabled": self.opponent_enabled,
            "wall_seconds": wall, "calibration_version": self.version,
            "marker_family": self.marker_family, "marker_ids": sorted(self.tags),
            "cameras": {k: dict(v) for k, v in self.metrics.items()},
            "filters": {name: {"accepted": f.accepted, "rejected": f.rejected,
                               "out_of_sequence": f.out_of_sequence,
                               "dropped_too_old": f.too_old, "id_rejections": f.id_rejections,
                               "identity": f.identity,
                               "model_probabilities": [float(v) for v in f.mu]}
                        for name, f in self.filters.items()},
            "buffer_pending": {name: len(b.pending) for name, b in self.buffers.items()},
            "lidar": ({"scans": self.lidar.scans, "detections": self.lidar.detections,
                       "rejections": dict(self.lidar.rejections),
                       "background_voxels": (len(self.lidar.background.voxels)
                                             if self.lidar.background else 0),
                       "rows": self.lidar_rows_written}
                      if self.lidar is not None else None),
            "clock": {"sim_start_ns": self.t0, "sim_end_ns": self.end_ns},
            "identity": ({**self.guard.summary(),
                          "opponent_reacquisitions": list(self.reacquisitions)}
                         if self.opponent_enabled else None),
            "errors": list(self.errors),
        }


def main(argv=None):
    a = parse_args(argv)
    if a.output.exists():
        raise SystemExit(f"{a.output} exists; choose a new --output directory")
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass
    cv2.setNumThreads(1)
    status = Replay(a).run()
    print(json.dumps({"output": str(a.output), "filters": status["filters"],
                      "cameras": status["cameras"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
