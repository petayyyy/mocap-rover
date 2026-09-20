# Localization plan status

Текущая фаза: **SIMULATION**  
Текущий этап: **S01 — аудит симулятора и бюджет производительности**  
Статус: **частично выполнен / baseline зафиксирован; SIM_ACCEPTED не объявлен**

Сделано:

- Проведён аудит Gazebo, конфигураций камер, генератора и smoke-check.
- Создан [sim_baseline_report.md](sim_baseline_report.md).
- Зафиксированы профили `ceiling_grid_baseline`, `hybrid_research`,
  `perimeter_only`, пересчёт footprints и R01/R03-ограничения.
- Создан [hardware_backlog.md](hardware_backlog.md); аппаратная фаза не начиналась.

Проверено:

- `gz sdf -k worlds/mocap_arena.sdf` — PASS.
- `/usr/bin/python3 -m unittest discover -s tests -v` — 4/4 PASS.
- `generate_world.py --seed 42` — воспроизводимый baseline, 0 uncovered samples
  на Z=0/0.3654/0.5 м без occlusion.
- `check_sim.py` — 6 RGB streams, 2 world poses, 1600×1200, exit 0.

Ограничения:

- Нет detector/tracker/fusion/publisher, recorder/replay и RTF benchmark.
- Целевые 30 Гц своего и ≥10 Гц второго пока не подтверждены; observed wall
  intervals нестабильны и не заменяют независимый FPS benchmark.
- CPU sample загрязнён другим Gazebo process; GPU telemetry заблокирована
  NVML/driver mismatch. Accuracy, latency и аппаратная реализуемость не оценены.

Исследовательский профиль: baseline `ceiling_grid_baseline`; R01–R06 учтены в
отчёте как: R01/R03 — профили и footprints; R02 — AprilTag baseline и transforms
сохраняются; R04 — разделить timestamps/counters и ground truth; R05 — YOLO
пока отсутствует; R06 — калибровка и distortion отложены. Ни один пункт не
объявлен проверенным измерением, если для него нет runtime данных.

Следующий этап: **S02**  
Следующий промпт: `docs/localization_plan/prompts/simulation/02_contracts.md`

Аппаратную фазу H01–H04 не начинать.
