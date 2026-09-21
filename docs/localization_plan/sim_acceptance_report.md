# Simulation acceptance report

Дата: 2026-09-21. Результат: **NOT ACCEPTED**.

| Критерий | Результат | Доказательство/ограничение |
|---|---|---|
| S04 registry/fault isolation | PASS | 4 registry tests; bounded queues, reconnect and independent channels |
| Timing/replay contracts | PASS (boundary) | 2 timing tests; no hardware clock/trigger claim |
| Intrinsics/extrinsics reports | PARTIAL | import/round-trip and graph checks; no image-derived BA accuracy |
| AprilTag observation contract | PARTIAL | synthetic corner geometry; no detector/image recall benchmark |
| Friendly fusion degradation | PARTIAL | synthetic end-to-end pipeline passes nominal/drop tests; no Gazebo image accuracy |
| Opponent >=10 Hz and ID switches | PARTIAL | synthetic pipeline measures >=10 Hz; no trained detector, multi-camera ID-switch set or wall-time camera pipeline |
| Settings/debug UI | PARTIAL | localhost `/` and `/api/status` smoke-check pass; no live six-preview wiring or visual walkthrough |
| 30-minute six-camera end-to-end run | FAIL | current baseline wall FPS ~3.316–3.648 and no full pipeline |
| Gazebo baseline SDF | PASS | `gz sdf -k worlds/mocap_arena.sdf` => Valid |

Commands:

```text
python3 -m unittest discover -s tests -v  # PASS — 32/32 including dashboard smoke-check
python3 -m unittest tests.test_pipeline -v # PASS — synthetic nominal/drop acceptance regression
gz sdf -k worlds/mocap_arena.sdf          # PASS — Valid
python3 - <<'PY'                            # S14 bounded synthetic benchmark
from simulation.benchmark import run
print(run(.05))
PY
```

Ground truth is evaluation-only. No physical cameras, IMX296/libcamera, udev,
trigger, hardware adapter or hardware-verified capability is included. This
report does not set `SIM_ACCEPTED`; H01–H04 remain unstarted.
