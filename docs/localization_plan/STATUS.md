# Localization plan status

Текущая фаза: **SIMULATION**  
Текущий этап: **S15 — приёмка всей системы в симуляторе**
Статус: **выполнен в заявленном объёме; SIM_ACCEPTED не объявлен**

## Сделано

- Добавлен пакет `localization_contracts` с общими контрактами sim/replay и будущего hardware adapter: `Observation`, `TrackStatus`, `CameraStatus`, `CalibrationSet`, `Frame`, `CaptureAdapter`.
- Зафиксированы schema version `1.0`, frame IDs, метрические единицы, clock domains, timestamps, covariance и явная валидность ориентации.
- `Observation` не содержит ground truth; ground truth остаётся evaluation-only. Detector/tracker в S02 не создавались.
- Добавлены marker family/ID, bundle, placement, 6D pose validity и attitude state.
- Добавлена [config/contracts.json](../../config/contracts.json): разделение observation/video planes, bounded queue, sim clock, reset-on-backward-jump и частоты publish/prediction.
- Добавлен `ConfigStore`: validate → stage → atomic apply → rollback и SHA-256 digest.
- Добавлен [launch/contracts.launch.json](../../launch/contracts.launch.json) без драйверов и физических камер.
- Добавлено преобразование twist и 6x6 covariance между frame semantics.
- Добавлены воспроизводимые S03-профили `demo_baseline`, `imx296_narrow` и `imx296_global_30` с разрешением, FPS, sensor mode, pixel format и оптическим профилем. IMX296 sensor mode/оптика помечены как неподтверждённые.
- `generate_world.py --profile` параметризует SDF; дефолтный baseline сохранён как 1600×1200/15 Гц. `check_sim.py` принимает `--width/--height` и выводит wall FPS; simulation FPS/RTF явно не подменяются wall FPS.
- Добавлен `simulation/faults.py`: seedable drop, delay, reorder, clock offset/drift. Capture timestamp не перезаписывается временем доставки.
- Добавлен отдельный `simulation/evaluator.py`; ground truth читается только evaluation-only функцией и не входит в `Observation`, `Frame` или рабочий capture contract. Detector/tracker не создавались.
- Добавлен `localization_contracts/registry.py`: стабильные `camera_1..camera_6`, runtime topic/stream IDs, preview metadata, simulation/replay capabilities и virtual edge node. `hardware_verified` принудительно запрещён.
- Реализованы проверки уникальности runtime ID/topic/stream ID и полной привязки шести логических камер, bounded queues с drop-oldest, reconnect с экспоненциальным backoff и версией calibration.
- Реализованы атомарная смена источника с проверкой совместимой calibration version, независимое отключение image stream/marker observations и статусы каналов.
- Добавлены воспроизводимые S04-тесты перестановки runtime IDs, duplicate binding, preview, bounded queue, независимого отказа канала и восстановления.
- Добавлены `TimingMetadata`, `ClockModel`, диагностика skew/offset и `ReplayLog`/`ReplayScheduler` с сохранением исходных capture/receive/processed timestamps, exposure reference и hash metadata.
- Реализованы явные режимы `free_running`/`ideal_sync`, независимое arrival scheduling и session marker при seek/reset; hardware trigger/PTP/NTP не реализованы.
- Добавлен dependency-free intrinsic calibration model с pinhole/rational/fisheye, K/D, ROI/binning/flip, CameraInfo-подобным экспортом, JSON import/export, distortion/undistortion, resize и held-out report metadata.
- Изменения размера делают калибровку `stale`; `focal_length_mm` хранится как metadata-only и не изменяет измеренную K.
- Добавлены `CalibrationGraph` и `CalibrationActivation`: связность графа камер/мишеней, минимальное число точек, gauge boundary, intrinsics version, quality/covariance и атомарная активация полного набора.
- Недостаточные или несвязные наблюдения отклоняются; solver boundary не читает Gazebo ground truth.
- Добавлен dependency-free AprilTag 36h11 observer для полных synthetic/replay detections: configurable IDs/0.40 m size, K geometry, timestamps, calibration version, quality/covariance и reject invalid ID/size.
- Добавлен `OpponentTracker` с confidence gating, подтверждением по истории, contact-point placeholder, velocity, yaw validity и timeout-based LOST; identity не выводится только из отсутствия тега.
- Добавлен capability-aware `SettingsBackend` с staged validate/apply/rollback, revision и ack/error; simulation trigger нельзя включить без capability.
- Добавлен structured `Diagnostics` для measured/filtered/predicted events, queue/drop/age каналов, latency percentiles и truth-hidden UI payload.
- Добавлен localhost-only `DashboardHandler` с `/` и `/api/status`; endpoint не выполняет shell-команды, не публикует raw frames и явно показывает `hardware_verified: false`.
- Добавлен bounded synthetic benchmark `simulation/benchmark.py`; report explicitly marks simulation-only and hardware-unverified.
- Добавлен `simulation/pipeline.py`: воспроизводимый synthetic camera→AprilTag/opponent→fusion pipeline с отдельными friendly/opponent rates, drops, LOST и runtime truth boundary.

## Проверено

```text
python3 -m unittest discover -s tests -v
  PASS — 9/9 tests (5 S02 + 4 existing S01)

python3 - <<'PY'
from localization_contracts.config import load_config, ConfigStore
c=load_config('config/contracts.json')
print('CONFIG_OK schema=%s cameras=%d digest=%s' % (c['schema_version'], len(c['cameras']), ConfigStore.digest(c)[:12]))
PY
  CONFIG_OK schema=1.0 cameras=6 digest=e6e2a3568998

python3 -m json.tool launch/contracts.launch.json >/dev/null
  PASS — launch JSON valid

python3 -m unittest discover -s tests -v
  PASS — 11/11 tests (S01/S02 + S03 faults/evaluator)

python3 -m unittest discover -s tests -v
  PASS — 15/15 tests (S01/S02 + S03 + S04 registry/fault scenarios)

python3 -m unittest discover -s tests -v
  PASS — 17/17 tests (S01–S05; timing skew/clock quality/replay round-trip/reset)

python3 -m unittest discover -s tests -v
  PASS — 19/19 tests (S01–S06; distortion round-trip, resize/stale, import/export and damaged file)

python3 -m unittest discover -s tests -v
  PASS — 21/21 tests (S01–S07; connected/disconnected graph, sparse data and atomic activation)

python3 -m unittest discover -s tests -v
  PASS — 23/23 tests (S01–S08; AprilTag identity, geometry, timestamps, quality and rejection)

python3 -m unittest discover -s tests -v
  PASS — 25/25 tests (S01–S09; asynchronous fusion, dedup, out-of-order, coasting and lost)

python3 -m unittest discover -s tests -v
  PASS — 28/28 tests (S01–S11; opponent confidence, confirmation and timeout)

python3 -m unittest discover -s tests -v
  PASS — 29/29 tests (S01–S12; settings revision, atomic apply/rollback and trigger restriction)

python3 -m unittest discover -s tests -v
  PASS — 30/30 tests (S01–S13; diagnostics state separation and latency/channel metrics)

python3 -m unittest discover -s tests -v
  PASS — 31/31 tests (S01–S14; bounded benchmark report)

python3 -m unittest discover -s tests -v
  PASS — 35/35 tests (S01–S15 acceptance artifacts, dashboard and pipeline evaluator)

python3 -m unittest tests.test_dashboard -v
  PASS — localhost HTML/status smoke-check

gz sdf -k worlds/mocap_arena.sdf
  PASS — Valid.

./scripts/run.sh -s --headless-rendering
/usr/bin/python3 scripts/check_sim.py --measure-seconds 3
  PASS — six RGB streams and both world poses; snapshots: /tmp/mocap-camera-check.
  wall_fps=3.648 for cameras 1–5 and 3.316 for camera 6; simulation FPS requires clock log, so 30 Hz is not accepted.

python3 scripts/generate_world.py --profile imx296_global_30 --seed 42 --output-dir /tmp/mocap-s03-profile
  PASS — profile=imx296_global_30, 1440×1080, nominal 30 Hz; geometric coverage: 0 uncovered samples (visibility/occlusion not proven)

gz sdf -k /tmp/mocap-s03-profile/worlds/mocap_arena.sdf
  PASS — Valid.

./scripts/run.sh -s --headless-rendering; /usr/bin/python3 scripts/check_sim.py --measure-seconds 3
  PASS — six RGB streams, both world poses, snapshots; measured wall FPS was 3.315–3.647 in this run, therefore 30 Hz is not accepted. The Python transport process emitted a pybind11 GIL abort on shutdown after the check; this is recorded as a runtime limitation, not hidden.

gz sdf -k worlds/mocap_arena.sdf
  PASS — Valid.
```

Gazebo baseline не изменён; SDF проверен после изменений. Новые контракты не подключают ground-truth topics в рабочий pipeline.

## Ограничения

- Detector, tracker, fusion, odometry publisher, recorder и полноценный replay runtime ещё не реализованы; фиктивные детекторы не добавлялись.
- S04 registry работает только с виртуальными sim/replay bindings; это не hardware adapter и не подтверждает физические камеры, IMX296/libcamera/udev/trigger или hardware verified capabilities.
- S05 time quality проверяется синтетическими timestamps; real driver timestamps, PTP/NTP, trigger и межузловая аппаратная синхронизация не подтверждены.
- S06 calibration solver/ChArUco image acquisition не подключены; report API фиксирует held-out fields, но не заявляет измеренную оптическую точность.
- S07 не заявляет восстановленную метрическую точность без реальных image-derived target observations; полноценный robust BA и Gazebo image solver остаются дальнейшей работой.
- S08 observer принимает image-derived corner detections, но detector backend/OpenCV AprilTag runtime, six-camera Gazebo image pipeline, blur/occlusion recall и measured Hz пока не подключены; это не hardware readiness.
- S11 detector weights, multi-camera geometric association, GPU benchmark и ≥10 Hz wall-time acceptance ещё не реализованы.
- S12 browser UI/E2E visual walkthrough не реализованы; backend не предоставляет hardware capability и не выполняет shell commands.
- S13 dashboard endpoint реализован, но полноценный live Gazebo data wiring, визуальный screenshot walkthrough и six-preview rendering ещё не подтверждены.
- S14 benchmark не является end-to-end Gazebo/30-minute acceptance; текущий Gazebo baseline wall FPS ниже nominal, поэтому требования frequency/age не приняты.
- Synthetic pipeline acceptance regression проходит nominal/drop scenarios, но не заменяет Gazebo image-based accuracy или wall-time 30-minute acceptance.
- Pipeline evaluator-only hold-out run: 60 matched samples, XY P95 ≈0 m and age P95 0 ms; these synthetic metrics are not presented as Gazebo or hardware performance.
- Добавлен асинхронный planar `PlanarFusion`: timestamped observations, dedup/out-of-order rejection, circular yaw, prediction на publish tick, counters, COASTING/LOST и рост covariance.
- ROS message packages и реальный Gazebo capture adapter пока отсутствуют; launch — декларативный контракт.
- Gazebo S03 adapter остаётся dependency-free boundary/профилем: полноценный ROS bridge, CameraInfo runtime capture, simulation-time/RTF counter и калибровочная мишень с реальной наблюдаемостью ещё не реализованы.
- Измеренный wall FPS baseline в smoke-check ниже номинала; capture FPS, detector FPS, accepted measurement Hz, output Hz, wall time, simulation time и RTF не смешиваются и не заявляются достигнутыми. Accuracy, calibration quality и LOST runtime behavior не измерялись; SIM_ACCEPTED не объявлен.
- Профили узкой оптики честно имеют статус `coverage: not demonstrated`; нулевое геометрическое покрытие baseline не означает пиксельную читаемость или отсутствие occlusion.
- Завершение `check_sim.py` сопровождается известным pybind11/GIL abort при завершении подписчиков; основной smoke-check до этого подтверждает шесть RGB topics и две позы.
- Физические камеры, IMX296/libcamera/udev/trigger и драйверы не подключались.
- Аппаратная фаза H01–H04 не начиналась.

## Артефакты

- `localization_contracts/{contracts.py,config.py,geometry.py,adapters.py,registry.py,timing.py,calibration.py,extrinsics.py,apriltag.py,fusion.py}`
- `config/contracts.json`, `launch/contracts.launch.json`, `tests/test_contracts.py`
- `tests/test_registry.py`
- `tests/test_timing.py`
- `tests/test_calibration.py`
- `tests/test_extrinsics.py`
- `tests/test_apriltag.py`
- `tests/test_fusion.py`
- `tests/test_opponent.py`
- `tests/test_settings.py`
- `tests/test_diagnostics.py`
- `tests/test_benchmark.py`
- `tests/test_dashboard.py`
- `docs/localization_plan/sim_acceptance_report.md`
- `simulation/pipeline.py`, `tests/test_pipeline.py`
- [sim_baseline_report.md](sim_baseline_report.md)
- `config/simulation_profiles.json`, `simulation/{faults.py,evaluator.py}`, `tests/test_simulation.py`

## Исследовательский профиль

Профиль `ceiling_grid_baseline`; S03 добавляет декларативные `demo_baseline`/`imx296_*` профили и fault injection. R02/R04 учтены marker registry, 6D/attitude validity, timestamp uncertainty, covariance units, раздельными observation/video planes и отсутствием ground truth в Observation. R01/R03/R05/R06 не заявлены проверенными измерениями; фактические оптические профили и RTF ещё не приняты.

## Следующий этап

**S15** — acceptance report: `docs/localization_plan/sim_acceptance_report.md`; результат NOT ACCEPTED, SIM_ACCEPTED не объявлен.

Аппаратную фазу H01–H04 не начинать.
