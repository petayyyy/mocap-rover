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
