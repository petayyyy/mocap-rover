# Simulation acceptance report

Дата: 2026-09-21. Результат: **NOT ACCEPTED**.

| Критерий | Результат | Доказательство/ограничение |
|---|---|---|
| S04 registry/fault isolation | PASS | 4 registry tests; bounded queues, reconnect and independent channels |
| Timing/replay contracts | PASS (boundary) | timing tests, provenance verification and 11-scenario fault matrix with measured offset/drift, replay reset and bounded delays; no hardware clock/trigger claim |
| Intrinsics/extrinsics reports | PARTIAL | JSON/CameraInfo YAML round-trip, image-derived per-view PnP plus robust bundle-adjustment path and held-out synthetic experiment; Gazebo image-derived calibration/optical gate remains unverified |
| AprilTag observation contract | PARTIAL | live camera_1 Gazebo smoke: 19/19 ID-0 detections and PnP accepts; evaluator-only pose run matched 17 with P50/P95 3D error≈0.1083 m, so metric accuracy gate fails |
| Friendly fusion degradation | PARTIAL | one-camera image→PnP→PlanarFusion/TrackStatus path and synthetic evaluator; no ROS odometry or trajectory accuracy |
| Opponent >=10 Hz and ID switches | PARTIAL | synthetic pipeline/contact projection/metric association tested; no trained detector or opponent wall-time camera run |
| Settings/debug UI | PARTIAL | localhost APIs, six-slot root HTML, stale-revision error ack and live six-preview/API/PPM smoke-check pass; no full interactive performance walkthrough or sustained UI comparison |
| 30-minute six-camera end-to-end run | PARTIAL | 30-minute simulation-time soak is bounded; real wall-time Gazebo end-to-end run still not passed |
| Gazebo baseline SDF | PASS | `gz sdf -k worlds/mocap_arena.sdf` => Valid |

Commands:

```text
python3 -m unittest discover -s tests -v  # PASS — 63/63 including image pipeline and dashboard smoke-check
python3 -m unittest tests.test_pipeline -v # PASS — synthetic nominal/drop acceptance regression
python3 -m unittest tests.test_acceptance_matrix -v # PASS — 11 reproducible S15 fault/replay scenarios
python3 - <<'PY'                            # evaluator-only hold-out metrics
from simulation.pipeline import run, evaluate_samples
print(evaluate_samples(run(2, return_samples=True)))
PY
  # matched=60, xy_p95_m≈0, age_p95_ms=0; truth is not runtime input
gz sdf -k worlds/mocap_arena.sdf          # PASS — Valid
/usr/bin/python3 scripts/check_image_pipeline.py --camera camera_1 --seconds 5
                                          # PASS (smoke only) — frames=19, detections=19, accepted=19, ID=0, CameraInfo runtime K/D, wall_fps=3.65, detector_p95≈313 ms
/usr/bin/python3 scripts/evaluate_live_pose.py --seconds 5
                                          # EVALUATOR — matched=17, P50/P95 3D error≈0.1083 m; truth is evaluator-only, accuracy gate NOT_ACCEPTED
/usr/bin/python3 scripts/check_sim.py --output /tmp/mocap-s15-preview-20260921 --measure-seconds 3
                                          # PASS — six RGB streams, two world poses, six PPM snapshots; wall_fps=4.33 each
ffmpeg -y -loglevel error -i /tmp/mocap-s15-preview-20260921/camera_1.ppm /tmp/mocap-s15-preview-20260921/camera_1.png
                                          # VISUAL CHECK — rendered Gazebo frame with visible tag rover
python3 - <<'PY'                            # S14 bounded synthetic benchmark
from simulation.benchmark import run
print(run(.05))
PY
python3 - <<'PY'                            # 30-minute simulation-time soak
from simulation.benchmark import soak
print(soak(30, drop_period=17))
PY
  # steps=27000, bounded=true, drops=1588, raw_frames_retained=false
```

Ground truth is evaluation-only. No physical cameras, IMX296/libcamera, udev,
trigger, hardware adapter or hardware-verified capability is included. This
report does not set `SIM_ACCEPTED`; H01–H04 remain unstarted.

An earlier OpenCV detector API probe crashed natively with exit 139 and is not
used. The committed legacy `cv2.aruco.detectMarkers` backend is isolated in the
simulation adapter; the live smoke path above completes, while Gazebo shutdown
still logs a pybind11/GIL abort after the report is emitted.

Additional required artifacts: `research_comparison_report.md` and
`operator_runbook_sim.md`.

Mandatory scenario traceability: `s15_gap_register.md` maps all nine required
scenarios to current evidence and gate status.

Launch manifests: `launch/simulation.launch.json` and
`launch/replay.launch.json`; both are declarative virtual/replay manifests and
do not instantiate physical devices.

Acceptance manifest command: `python3 scripts/run_acceptance.py --seed 9
--output /tmp/mocap-s15-acceptance-9.json`. The manifest records config digest,
pipeline/evaluator/fault matrix, runs seed sweep 42/7/123, emits a hashed 600-second episode manifest, hashes the world and `config/cameras.json`, records git revision, writes a timing replay recording and keeps
`sim_accepted: false`.

Re-run evidence on 2026-09-21: `python3 scripts/run_acceptance.py --seed 9
--output /tmp/mocap-s15-current.json` completed with seed sweep 42/7/123,
11/11 fault scenarios passing, 60 replay frames, 40 scenario episodes,
`truth_used_by_runtime=false`, and `sim_accepted=false`. The manifest records
git revision `19e8d27`, world/config/calibration digests, and
`hardware_verified=false`.

Final bundle smoke-check with seed 42 produced acceptance JSON/replay evidence;
the episode manifest covers 40 episodes (2 seeds × 5 styles × 4 trajectories),
fault matrix `all_pass=true`, gate `sim_accepted=false`.

Snapshot dashboard command: `python3 scripts/serve_dashboard.py
--snapshot-dir /tmp/mocap-s15-preview-20260921 --port 8080`.

Live dashboard smoke-check on port 18080: HTML length 1421 with 6 preview
slots; `/api/cameras`=6; `/api/previews`=6; `/preview/camera_1` returned PPM
magic `P6` and 5,760,017 bytes. The local process was stopped with Ctrl-C.

Live dashboard wiring smoke-check on port 18081 (Gazebo running):
`/api/status` length 202 reported frames for all six cameras;
`/api/cameras` length 2604 returned six entries;
`/api/previews` length 882 returned six available entries;
`/preview/camera_1` length 5,760,017 began with `P6`. This confirms the
live six-channel preview path, not the performance or metric-accuracy gates.
