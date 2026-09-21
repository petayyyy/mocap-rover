# Simulation acceptance report

Дата: 2026-09-21. Результат: **NOT ACCEPTED**.

| Критерий | Результат | Доказательство/ограничение |
|---|---|---|
| S04 registry/fault isolation | PASS | 4 registry tests; bounded queues, reconnect and independent channels |
| Timing/replay contracts | PASS (boundary) | 2 timing tests; no hardware clock/trigger claim |
| Intrinsics/extrinsics reports | PARTIAL | import/round-trip and graph checks; no image-derived BA accuracy |
| AprilTag observation contract | PARTIAL | synthetic corner geometry; no detector/image recall benchmark |
| Friendly fusion degradation | PARTIAL | synthetic evaluator: 60 matched, XY P95 ~0 m, age P95 0 ms; all six registry channels exercised, camera_6 outage isolated; no Gazebo image accuracy |
| Opponent >=10 Hz and ID switches | PARTIAL | synthetic pipeline measures >=10 Hz; no trained detector, multi-camera ID-switch set or wall-time camera pipeline |
| Settings/debug UI | PARTIAL | localhost `/` and `/api/status` smoke-check pass; no live six-preview wiring or visual walkthrough |
| 30-minute six-camera end-to-end run | PARTIAL | 30-minute simulation-time soak is bounded; real wall-time Gazebo end-to-end run still not passed |
| Gazebo baseline SDF | PASS | `gz sdf -k worlds/mocap_arena.sdf` => Valid |

Commands:

```text
python3 -m unittest discover -s tests -v  # PASS — 32/32 including dashboard smoke-check
python3 -m unittest tests.test_pipeline -v # PASS — synthetic nominal/drop acceptance regression
python3 - <<'PY'                            # evaluator-only hold-out metrics
from simulation.pipeline import run, evaluate_samples
print(evaluate_samples(run(2, return_samples=True)))
PY
  # matched=60, xy_p95_m≈0, age_p95_ms=0; truth is not runtime input
gz sdf -k worlds/mocap_arena.sdf          # PASS — Valid
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

Additional required artifacts: `research_comparison_report.md` and
`operator_runbook_sim.md`.
