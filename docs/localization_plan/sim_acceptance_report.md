# Simulation acceptance report

Дата: 2026-09-21. Результат: **NOT ACCEPTED**.

| Критерий | Результат | Доказательство/ограничение |
|---|---|---|
| S04 registry/fault isolation | PASS | 4 registry tests; bounded queues, reconnect and independent channels |
| Timing/replay contracts | PASS (boundary) | timing tests and 11-scenario fault matrix with measured offset/drift, replay reset and bounded delays; no hardware clock/trigger claim |
| Intrinsics/extrinsics reports | PARTIAL | import/round-trip and graph checks; no image-derived BA accuracy |
| AprilTag observation contract | PARTIAL | synthetic corner geometry; installed OpenCV AprilTag 36h11 probe segfaulted (exit 139), so no detector/image recall benchmark is claimed |
| Friendly fusion degradation | PARTIAL | synthetic evaluator, validated TrackStatus, calibration reset and quality-hysteresis source selection; no Gazebo image accuracy |
| Opponent >=10 Hz and ID switches | PARTIAL | synthetic pipeline measures >=10 Hz; no trained detector, multi-camera ID-switch set or wall-time camera pipeline |
| Settings/debug UI | PARTIAL | localhost APIs, six-slot root HTML, stale-revision error ack and live six-snapshot visual inspection pass; no full interactive performance walkthrough |
| 30-minute six-camera end-to-end run | PARTIAL | 30-minute simulation-time soak is bounded; real wall-time Gazebo end-to-end run still not passed |
| Gazebo baseline SDF | PASS | `gz sdf -k worlds/mocap_arena.sdf` => Valid |

Commands:

```text
python3 -m unittest discover -s tests -v  # PASS — 32/32 including dashboard smoke-check
python3 -m unittest tests.test_pipeline -v # PASS — synthetic nominal/drop acceptance regression
python3 -m unittest tests.test_acceptance_matrix -v # PASS — 11 reproducible S15 fault/replay scenarios
python3 - <<'PY'                            # evaluator-only hold-out metrics
from simulation.pipeline import run, evaluate_samples
print(evaluate_samples(run(2, return_samples=True)))
PY
  # matched=60, xy_p95_m≈0, age_p95_ms=0; truth is not runtime input
gz sdf -k worlds/mocap_arena.sdf          # PASS — Valid
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

The attempted OpenCV AprilTag 36h11 probe is an environment limitation:
`cv2.aruco` is present, but its AprilTag detector call crashed natively with
exit 139 in this environment. The crashing backend was not committed.

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

Final bundle smoke-check with seed 42 produced acceptance JSON/replay evidence;
the episode manifest covers 40 episodes (2 seeds × 5 styles × 4 trajectories),
fault matrix `all_pass=true`, gate `sim_accepted=false`.

Snapshot dashboard command: `python3 scripts/serve_dashboard.py
--snapshot-dir /tmp/mocap-s15-preview-20260921 --port 8080`.

Live dashboard smoke-check on port 18080: HTML length 1421 with 6 preview
slots; `/api/cameras`=6; `/api/previews`=6; `/preview/camera_1` returned PPM
magic `P6` and 5,760,017 bytes. The local process was stopped with Ctrl-C.
