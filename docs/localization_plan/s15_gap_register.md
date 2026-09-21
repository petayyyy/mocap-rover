# S15 mandatory scenario gap register

This register is intentionally conservative. `PASS` means the stated evidence
exists in the current simulation boundary; it does not promote synthetic data
to hardware verification.

| # | Required scenario | Evidence | Gate |
|---|---|---|---|
| 1 | 10-minute rest/motion trajectories | short synthetic pipeline only; no 10-minute trajectory manifest | NOT_ACCEPTED |
| 2 | seams, corners, field boundaries, partial overlap | geometric baseline coverage and camera snapshots; no detector seam accuracy | PARTIAL |
| 3 | five opponent styles/lights/brightness | config has one active scenario style; no detector sweep | NOT_ACCEPTED |
| 4 | independent mounting seeds/ideal control | S03 seed/fault infrastructure and baseline geometry | PARTIAL |
| 5 | close/crossing rovers, hidden tag, similar materials | tracker unit tests only; no rendered crossing/occlusion dataset | NOT_ACCEPTED |
| 6 | drops 5/10/30%, reorder, 20/50/100/200 ms delay | 11-scenario `acceptance_matrix.py`, reproducible | PASS (boundary) |
| 7 | camera disable/restore/runtime remap/wrong calibration | S04 registry tests and camera_6 outage; no Gazebo wrong-calibration accuracy | PARTIAL |
| 8 | clock drift/jump, pause/reset, seek, exposure variation | clock offset/drift and replay reset rows; no Gazebo pause/seek exposure episode | PARTIAL |
| 9 | 30-minute UI+6 previews+recording and no-UI run | simulation-time bounded soak and live snapshot dashboard; no 30-minute wall-time run | NOT_ACCEPTED |

The register is an acceptance gap, not a waiver. `SIM_ACCEPTED` remains unset
until mandatory baseline criteria are actually measured.
