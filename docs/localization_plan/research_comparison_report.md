# Research comparison report (R01–R06)

Дата: 2026-09-21. Профиль: `ceiling_grid_baseline`, simulation/replay only.

| R | Проверка | Результат | Evidence / limitation |
|---|---|---|---|
| R01 | layouts/FOV/coverage | PARTIAL | baseline SDF and geometric coverage exist; hybrid/perimeter and readability are not measured |
| R02 | marker/observer/attitude | PARTIAL | tag36h11 contract and synthetic geometry tests; no OpenCV backend recall comparison |
| R03 | timing/accuracy/2 m/s | NOT_ACCEPTED | synthetic evaluator exists; no image-derived calibration and no wall-time 2 m/s run |
| R04 | independent channels/faults | PASS (simulation boundary) | registry, bounded queues, camera_6 outage and fault tests |
| R05 | opponent/association | PARTIAL | confidence tracker and multi-camera gating; no trained detector or real/sim rendered bbox sweep |
| R06 | intrinsics/calibration | PARTIAL | distortion round-trip and graph guards; no ChArUco image solve or held-out optical comparison |

Alternatives are not silently accepted: ArUco/ChArUco comparison, fisheye vs
rational measured solve, and real-camera domain gap remain NOT_ACCEPTED. No
hardware claim follows from these simulation results.

## Required comparison matrix

1. **Mounting profiles.** `ceiling_grid_baseline` is present in the SDF and its
   geometric no-hole check is recorded in `sim_baseline_report.md`. The
   `hybrid_research` and `perimeter_only` layouts are not implemented in the
   current world, so GSD/angle/occlusion worst-case values are `NOT_ACCEPTED`.
2. **Marker backends.** The contract fixes AprilTag 36h11, IDs 0/1 and 0.40 m
   black border. ArUco/ChArUco, side/bottom bundle, partial damage and flip
   comparison have no valid detector evidence. The installed OpenCV AprilTag
   probe crashed natively (exit 139), recorded in S15.
3. **2 m/s/timing profile.** Fault injection covers deterministic drops,
   delay, reorder, offset and drift; the evaluator covers only the synthetic
   nominal trajectory. A rendered 900×700×250 mm, 2 m/s blur/shutter sweep and
   0/1/5/20 ms accuracy comparison are `NOT_ACCEPTED`.
4. **Calibration alternatives.** Intrinsic round-trip, stale-on-resize and
   connected graph guards are tested. No ≥20-point image-derived ChArUco/floor
   target solve, held-out residual heatmap, master duration or raw-vs-rectified
   comparison is available; all are `NOT_ACCEPTED`.
5. **Opponent alternatives.** Confidence gating, timeout, one-object
   multi-camera selection and ID-switch counting are tested. YOLO threshold
   sweep, height/3D-box/contact/triangulation comparison and rendered material
   sweep are unavailable; MOG2 is not used to create tracks.
6. **Separate channels/prediction.** Registry and pipeline exercise six
   independent image channels, camera_6 outage and 30/15 Hz synthetic counters.
   Optional 100/200 Hz prediction and noisy wheel/IMU fallback are intentionally
   not implemented.

The matrix accounts for every required comparison, but does not claim each
alternative passed. It blocks `SIM_ACCEPTED` until the baseline mandatory set
and applicable comparison evidence are complete.
