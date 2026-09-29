# Журнал дирижёра (ноутбук Intel iGPU)

Ведётся чатом-дирижёром на машине без NVIDIA. Первое, что читает исходный
дирижёр, вернувшись к отладке GPU. Формат записи: дата, промпт, итерация,
метрики своего прогона, решение.

## Машина и окружение

- Intel Core Ultra 7 155H, 22 потока, 30 ГБ ОЗУ, Mesa Intel Arc (MTL),
  Ubuntu 24.04, Python 3.12.3, системный OpenCV 4.6 (без
  `generateImageMarker`), ROS 2 Jazzy + `ros-jazzy-ros-gz`, Gazebo Sim 8.15.0.
- В Bash-инструменте `source` работает только внутри `bash -c '...'`; иначе
  `gz` и модуль `gz` не видны. Команды дирижёра пишутся в этой форме.
- Инструменты дирижёра: `scripts/conductor_check.py truth <датасет>` и
  `scripts/conductor_check.py replay <реплей> --truth <датасет>/truth.jsonl`
  (P50/P95/valid по сим-времени/подмены/дальность/скорость/отказы PnP/режимы
  ROI). Тест `tests/test_conductor_check.py`. Тракт дирижёр не пишет.

## Хронология

### 2026-09-29 — старт

- Прочитаны README, методичка, промпты 01–04, `hybrid_rover_tracking.md`,
  `readme.md`, `localization_contracts/*`, `run_localization.py`,
  `evaluate_recording.py`, `check_dataset.py`, `record_dataset.py`.
- На этой машине ещё нет `.venv` и нет `artifacts/` (датасеты пишутся здесь
  заново рабочим чатом по `handoff_worker.md`, шаги 0–1).
- Замечания к существующему тракту, которые дирижёр держит в уме при приёмке
  (рабочему чату не подсказываются, пока он сам не упрётся):
  - `evaluate_recording.py` печатает только P95, `availability` по выходам и
    порог yaw 3°; критерии README (P50, valid по сим-времени, подмены, yaw 5°)
    считаются `conductor_check.py`.
  - `PnpAprilTagObserver`: путь fisheye уже есть (undistort → PnP с нулевой
    дисторсией, `max_valid_radius = inf`), но гейт `base_height` при
    fisheye режет большинство детекций (известно из README).
  - `CameraRoiPlanner.plan` берёт `now_ns` из wall-времени
    (`run_localization.py` передаёт `time.monotonic_ns()`), watchdog поэтому
    привязан к wall, а не к сим-времени; в реплее это надо учитывать.
  - `AsyncObservationBuffer.drain(now_ns)` вызывается из тика публикации с
    `clock["sim"]`; окно группы 12 мс добавляется к задержке измерения.
  - `run_localization.py` требует `torch`/`ultralytics` без `--tag-only`;
    в реплее YOLO-путь должен отсутствовать целиком.
  - `cuboid.localize_box` использует `cv2.undistortPoints` с pinhole D, не
    fisheye — для соперника (промпт 03) это резервный путь и он потребует
    `CameraModel`.
  - `lidar_pipeline.ArenaLidar` захардкожены шаги L2 (1.25°/1.43°) в
    `footprint_m`; Airy 0.4°/0.947°. Влияет на ячейку кластеризации и сигму.

## Ждёт RTX-машины (здесь не измерить)

- Абсолютные времена кадра (P95 ≤ 5 мс на 6 потоках), пропускная способность
  реплея ≥ 1.0× сим-времени, нагрузка CPU ≤ 60 % — все пороги промпта 04.
- CUDA-варианты модели фона (`cv2.cuda` MOG2) против CPU-в-окнах: сравнение
  по P95 времени кадра.
- Любые TensorRT/torch-на-GPU шаги (не планируются, но если появятся).
- Профиль по стадиям в `timing.json` на реальной машине.

## Решение 2026-09-29: связь с проектом br_lidar

Пользователь решил: отладка алгоритма и матмодели идёт в `mocap-rover` на
Gazebo (арена 12×12, 6 камер IMX219, Airy). Затем наработки заливаются в
отдельную ветку `mocap-rover` репозитория `br_lidar`
(`/home/petayyyy/Drones/br_lidar`) для проверки на их стенде (реальный Airy,
Orange Pi + stitchd, одна панорамная камера, ринг 10×10). Полная арена с 6
камерами вживую — только после этого.

Из `br_lidar` берём то, что отработано на железе (лидар у них финальный):
- фон лидара по лучам с σ на луч (`airy_py/background.py`) вместо
  воксельной карты;
- калибровка плоскости пола RANSAC (`airy_py/floor_calibration.py`);
- медианный центр с отбраковкой, PCA-габарит, зоны масок арены, флаг
  тряски (`airy_py/detector.py`);
- приём MSOP/DIFOP, режимы часов и сверка часов платы
  (`airy_lidar.py`, `runtime/sensor_sync.py`) — для ветки интеграции, в
  Gazebo не нужны;
- сборщик кадров и детектор на C++ (`navigation/src/airy.cpp`) — основа для
  посекторной обработки, позже.

Не берём: аффинную привязку камеры, медианный фильтр позы, выбор цели
«единственный второй контур», отсутствие детекции соперника по камере.

Граница интеграции в `br_lidar`: `CameraLidarPoseSource.process_frame` и
`TargetIdentity.select`; потребитель `autonomy.py` ждёт `OwnPose` и
`TrackedTarget`. Наш `ImmRoverFilter.publish` покрывает оба контракта.
Их контроллер не экстраполирует цель и берёт позу с возрастом до 0,45 с —
это придётся менять на их стороне, иначе выигрыш по задержке не дойдёт.
