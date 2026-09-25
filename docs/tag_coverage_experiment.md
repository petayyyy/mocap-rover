# Полный прогон покрытия AprilTag

Эксперимент предназначен для оценки стабильности локализации `tag_rover` по
всей арене и последующего подбора положения/наклона центральных камер.

Во время запуска скрипт автоматически:

1. создаёт копию `mocap_arena_l2.sdf` без `yolo_rover`;
2. извлекает из SDF положения и параметры всех шести камер;
3. запускает Gazebo и текущий AprilTag runtime;
4. проводит ровер змейкой по рабочей зоне;
5. публикует эталонную позу Gazebo в
   `/mocap/tag_rover/ground_truth` (`nav_msgs/msg/Odometry`, frame `arena`);
6. останавливает только запущенные им процессы;
7. формирует CSV для анализа.

## Запуск

Из нового терминала:

```bash
cd /home/popka/Drone/mocap-rover
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 scripts/run_tag_coverage_experiment.py
```

По умолчанию GUI Gazebo остаётся видимым. Прогон занимает до восьми минут.
Каталог результата создаётся в `artifacts/tag_coverage_<date>_<time>`.

Для более быстрого первого прогона:

```bash
/usr/bin/python3 scripts/run_tag_coverage_experiment.py \
  --lane-spacing 1.5 --speed 0.65 --max-seconds 360
```

Headless-вариант:

```bash
/usr/bin/python3 scripts/run_tag_coverage_experiment.py --headless
```

`Ctrl+C` безопасно останавливает ровер, локализатор и Gazebo. Исходный мир не
перезаписывается; противник удалён только из `experiment_world.sdf` конкретного
прогона.

## Топики во время прогона

```text
/mocap/tag_rover/ground_truth  nav_msgs/msg/Odometry  # только evaluator
/mocap/tag_rover/odom          nav_msgs/msg/Odometry  # оценка по AprilTag
/mocap/tag_rover/status        std_msgs/msg/String
```

Ground truth никогда не передаётся в детектор или fusion и используется только
для записи и оценки ошибок.

## Результаты

- `truth.csv` — фактическая поза Gazebo, команды и номер waypoint;
- `camera_frames.csv` — каждый обработанный кадр каждой камеры, включая пропуски;
- `observations.csv` — все валидные PnP-наблюдения и их ошибки;
- `estimates.csv` — fused pose и ошибка относительно Gazebo;
- `camera_summary.csv` — recall/PnP/latency/error отдельно по камерам;
- `spatial_bins.csv` — покрытие каждой камеры по метровым клеткам арены;
- `manifest.json` — схема камер, параметры маршрута и итог выполнения;
- `summary.json` — общие p50/p95;
- `runtime/*.jsonl` — первичные записи без потери подробностей;
- `gazebo.log`, `localization.log` — диагностика запуска.

Для повторного построения CSV из уже завершённого каталога:

```bash
/usr/bin/python3 scripts/analyze_tag_coverage.py \
  artifacts/tag_coverage_<date>_<time>
```

Для первичного разбора достаточно передать все CSV, `manifest.json` и
`summary.json`. Если потребуется менять параметры самого детектора, дополнительно
понадобятся `runtime/camera_frames.jsonl`, `runtime/observations.jsonl` и логи.
