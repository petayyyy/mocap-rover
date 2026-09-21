# Localization plan status

Текущая фаза: **SIMULATION**  
Текущий этап: **S15 — заблокирован до закрытия сквозных image-based критериев**
Статус: **контракты и synthetic-регрессии выполнены; live Gazebo image→pose приёмка не выполнена; SIM_ACCEPTED не объявлен**

## Correction checklist по ревью

- [x] Исправлено: synthetic AprilTag geometry больше не объявляет PnP/6-D pose и valid attitude (`localization_contracts/apriltag.py`).
- [x] Исправлено: одновременные observations не создают искусственный огромный velocity; rates считаются от начала текущей session (`localization_contracts/fusion.py`).
- [x] Добавлены регрессии на оба случая (`tests/test_apriltag.py`, `tests/test_fusion.py`).
- [x] Добавлен OpenCV IPPE-SQUARE PnP boundary с явной цепочкой `T_arena_camera · T_camera_tag · inv(T_base_tag)` и метрическим synthetic regression (`localization_contracts/apriltag.py`).
- [x] Добавлен OpenCV legacy AprilTag image detector adapter (`localization_contracts/detector.py`) и detector→PnP regression с capture timestamp; он не получает ground truth.
- [x] Добавлен runnable `scripts/check_image_pipeline.py`: Gazebo image topic → RGB frame → OpenCV AprilTag detector → PnP observer, с capture timestamp, latency и hardware/simulation provenance flags.
- [x] Добавлен covariance-based innovation gate и reject counter для implausible jumps; prediction horizon и publication tick остаются отдельными от measurement acceptance (`localization_contracts/fusion.py`).
- [x] Добавлен image-derived per-view PnP calibration path с metric board points, K/D, board pose, reprojection reject и robust median translation (`CalibrationGraph.add_image_observation/solve_image_observations`).
- [x] S11 correction: opponent tracker accepts metric contact points, rejects non-monotonic timestamps, and multi-camera association gates metric positions; camera handoff no longer counts as an ID switch (`localization_contracts/opponent.py`).
- [x] S10 добавлен reproducible image/YOLO-label builder (`simulation.dataset.build_image_dataset`): PNGs, labels, session-level train/val/test split and leakage-safe manifest; weights/training status remain explicitly unset.
- [x] Dashboard теперь имеет dynamic status polling (500 ms), XY map marker и явный LOST color path; это UI capability smoke coverage, не подтверждение live Gazebo data wiring.
- [x] Image pipeline теперь требует и принимает live `/cameras/camera_N/camera_info` K/D перед PnP; отсутствие CameraInfo приводит к диагностической ошибке.
- [x] Detector добавил bounded raw+Otsu contrast path; на сохранённом Gazebo snapshot crop этот путь воспроизводит ID 0, но полный live run после текущего SDF rendering пока остаётся `detections=0`.
- [ ] Не закрыто: настоящий detector, PnP/K/D, T_arena_camera/T_base_marker и ROS 2 Odometry.
- [ ] Не закрыто: image-derived calibration/BA, шесть live image channels, настоящий YOLO и 30-minute wall-time profile.
- [ ] S07 remains partial: current solver is robust per-view PnP aggregation, not joint bundle adjustment; held-out calibration accuracy and hidden-mount-error recovery remain unverified.

Эти исправления не превращают synthetic boundary в доказательство сквозной точности или аппаратной готовности.

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
- `ReplayLog` теперь проверяет config/calibration/model provenance hashes и отклоняет несовместимый replay до обработки кадров.
- Добавлен dependency-free intrinsic calibration model с pinhole/rational/fisheye, K/D, ROI/binning/flip, CameraInfo-подобным экспортом, JSON import/export, distortion/undistortion, resize и held-out report metadata.
- Добавлен ROS CameraInfo/OpenCV-style YAML import/export round-trip using the available PyYAML runtime; malformed YAML remains rejected.
- Изменения размера делают калибровку `stale`; `focal_length_mm` хранится как metadata-only и не изменяет измеренную K.
- Добавлены `CalibrationGraph` и `CalibrationActivation`: связность графа камер/мишеней, минимальное число точек, gauge boundary, intrinsics version, quality/covariance и атомарная активация полного набора.
- Недостаточные или несвязные наблюдения отклоняются; solver boundary не читает Gazebo ground truth.
- Добавлен dependency-free AprilTag 36h11 observer для полных synthetic/replay detections: configurable IDs/0.40 m size, K geometry, timestamps, calibration version, quality/covariance и reject invalid ID/size.
- Observer намеренно публикует только planar geometry: `pose_6d_valid=False`, `attitude_state=unknown`; PnP и внешние transforms остаются незавершённым SIM-критерием.
- `PnpAprilTagObserver` теперь отдельно публикует 6D-valid pose только после успешного PnP/reprojection check; detector, Gazebo image capture и ROS odometry к нему пока не подключены.
- Добавлен `OpponentTracker` с confidence gating, подтверждением по истории, contact-point placeholder, velocity, yaw validity и timeout-based LOST; identity не выводится только из отсутствия тега.
- Добавлен `MultiCameraAssociator`: один выбранный candidate на timestamp, temporal/pixel gating, отсутствие duplicate tracks и измеряемый `id_switches` counter.
- Added height-aware bbox contact-point projection to arena ground plane with explicit height uncertainty; bbox center is not treated as body center.
- Добавлен capability-aware `SettingsBackend` с staged validate/apply/rollback, revision и ack/error; simulation trigger нельзя включить без capability.
- Settings backend rejects stale `expected_revision` with an explicit error ack, covering concurrent client apply protection.
- Добавлен structured `Diagnostics` для measured/filtered/predicted events, queue/drop/age каналов, latency percentiles и truth-hidden UI payload.
- Добавлен localhost-only `DashboardHandler` с `/` и `/api/status`; endpoint не выполняет shell-команды, не публикует raw frames и явно показывает `hardware_verified: false`.
- Dashboard дополнен `/api/cameras`: единый registry отдаёт шесть virtual camera statuses, channel health и simulation-only capabilities для UI.
- Dashboard дополнен `/api/previews`: шесть preview metadata entries (size/pixel format/availability) без raw payload.
- Dashboard дополнен explicit localhost `/preview/camera_N` PPM serving from checked snapshot paths; raw frames are served only as preview HTTP responses, never embedded in JSON.
- Dashboard root now renders six preview slots with camera labels and asynchronously loaded channel metadata; browser smoke-test verifies all six preview URLs.
- Added `scripts/serve_dashboard.py` to wire the shared registry and checked Gazebo snapshot directory into the localhost dashboard without physical devices.
- Добавлен bounded synthetic benchmark `simulation/benchmark.py`; report explicitly marks simulation-only and hardware-unverified.
- Добавлен simulation-time soak на 30 минут: 27,000 steps, 6 каналов, bounded queues, 1,588 искусственных drops, raw frames не удерживаются.
- Добавлен `simulation/pipeline.py`: воспроизводимый synthetic camera→AprilTag/opponent→fusion pipeline с отдельными friendly/opponent rates, drops, LOST и runtime truth boundary.
- Review correction: fusion rejects late/deduplicated observations, handles equal capture timestamps without artificial velocity, and reports per-session rates; this remains a filter contract, not a validated ROS/Gazebo odometry node.

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
  PASS — 62/62 tests (S01–S15 acceptance artifacts, Gazebo snapshot contrast detector regression, reproducible image/label split regression, metric opponent handoff/association regressions, image-derived calibration PnP regression, real image detector→PnP path, innovation-gate, PnP transform/reprojection regression, review regressions, runnable snapshot dashboard, pipeline evaluator, multi-camera association/selection, contact projection, TrackStatus contract, YAML calibration round-trip, replay provenance, soak, fault matrix, launch manifests, scenario manifest, stale revision and calibration reset)

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
- OpenCV AprilTag 36h11 probe was attempted but crashed natively (exit 139) in the installed build; no crashing detector backend was retained or presented as verified.
- S11 detector weights, multi-camera geometric association, GPU benchmark и ≥10 Hz wall-time acceptance ещё не реализованы.
- S11 multi-camera association boundary и ID-switch counter покрыты тестом; detector weights, full projection/triangulation and wall-time acceptance remain unverified.
- S12 browser UI/E2E visual walkthrough не реализованы; backend не предоставляет hardware capability и не выполняет shell commands.
- S13 dashboard endpoint реализован, но полноценный live Gazebo data wiring, визуальный screenshot walkthrough и six-preview rendering ещё не подтверждены.
- `/api/cameras` smoke-test подтверждает шесть virtual entries; live Gazebo previews и визуальный walkthrough остаются непроверенными.
- `/api/previews` smoke-test подтверждает шесть metadata entries; live rendered frames и visual screenshot walkthrough остаются непроверенными.
- Preview file endpoint smoke-test подтверждает local PPM response; Gazebo snapshot directory still requires a live smoke-check before visual acceptance.
- Root HTML smoke-test confirms six preview slots; it is a localhost rendering check, not a full interactive Gazebo performance acceptance.
- Live dashboard smoke-check on port 18080: HTML 1421 bytes/6 slots, cameras=6, previews=6, camera_1 PPM magic `P6`/5,760,017 bytes; process stopped with Ctrl-C.
- Live preview smoke-check: `/usr/bin/python3 scripts/check_sim.py --output /tmp/mocap-s15-preview-20260921 --measure-seconds 3` подтвердил 6 RGB streams, 2 world poses и 6 PPM (1600×1200); wall_fps=4.33 each, target 30 Hz не принят.
- Image pipeline smoke-check: `/usr/bin/python3 scripts/check_image_pipeline.py --camera camera_1 --seconds 5` получил 19 Gazebo frames, `detections=19`, `accepted=19`, ID 0, runtime CameraInfo K/D, `wall_fps=3.65`, detector p95≈313.2 ms; ground truth runtime не использован. Accuracy/frequency gate остаётся NOT_ACCEPTED.
- Gazebo shutdown logged unresolved `pybind11::handle::dec_ref()`/GIL abort after the smoke-check; process was stopped and this runtime issue remains an explicit baseline limitation.
- Visual inspection of converted `camera_1.png` and six-view montage confirmed rendered Gazebo imagery and visible tag rover/AprilTag; this does not establish detector recall or metric accuracy.
- Research comparison report expanded with explicit matrix for all required items 1–6 from `04_implementation_plan.md`; unavailable alternatives remain NOT_ACCEPTED rather than inferred.
- Added `simulation/acceptance_matrix.py`: reproducible 10-scenario matrix for 5/10/30% drops, 20/50/100/200 ms delay, reorder, clock/replay boundary and camera_6 outage.
- Added `simulation/scenarios.py`: reproducible rest/straight/circle/eight episodes across independent seeds, five styles, lighting/brightness and occlusion/crossing labels; truth remains evaluation-only.
- Clock/replay matrix row now executes `ReplayScheduler.schedule(reset=True)` and asserts an explicit `new_session` marker before replay frames.
- Delay rows in the matrix now record observed transport delay and assert it stays within each configured 20/50/100/200 ms bound.
- Matrix now also measures a 5 ms offset + 20 ppm drift case (`clock_delta_ns=5,020,000`) and checks replay reset separately.
- Added declarative `launch/simulation.launch.json` and `launch/replay.launch.json`; both use the shared registry, do not connect physical devices and explicitly set `hardware_verified: false`.
- Added `scripts/run_acceptance.py`: deterministic S15 manifest with config digest, seed, pipeline/evaluator/fault-matrix results and explicit `sim_accepted: false` gate.
- Acceptance runner now writes a bounded `ReplayLog` recording with config/calibration/model hashes and timing metadata, and records its path/frame count in the manifest.
- Acceptance manifest now hashes and records the concrete `config/cameras.json` calibration snapshot, and uses that digest in the replay header.
- Acceptance manifest now also records `worlds/mocap_arena.sdf` SHA-256 and current git revision for reproducible world/code provenance.
- Acceptance manifest now includes independent synthetic seed sweep 42/7/123 with friendly/opponent rates and camera_6 outage counts.
- Acceptance runner now emits a hashed 600-second episode manifest (rest/straight/circle/eight, seeds 42/7, styles/lighting/occlusion labels) alongside replay evidence.
- Final runner smoke-check (seed 42): acceptance JSON 3,616 B, replay 14,999 B/60 frames; episode manifest now covers 40 episodes (2 seeds × 5 styles × 4 trajectories), fault matrix all_pass=true, sim_accepted=false.
- S14 benchmark не является end-to-end Gazebo/30-minute acceptance; текущий Gazebo baseline wall FPS ниже nominal, поэтому требования frequency/age не приняты.
- S14 soak подтверждает только bounded simulation-time behavior; реальный 30-minute wall-time Gazebo end-to-end run по-прежнему не принят.
- Synthetic pipeline acceptance regression проходит nominal/drop scenarios, но не заменяет Gazebo image-based accuracy или wall-time 30-minute acceptance.
- Pipeline evaluator-only hold-out run: 60 matched samples, XY P95 ≈0 m and age P95 0 ms; these synthetic metrics are not presented as Gazebo or hardware performance.
- Synthetic pipeline now exercises all six registry cameras; deterministic camera_6 outage (10 frames over 2 s) leaves camera_1/friendly output active.
- Добавлен асинхронный planar `PlanarFusion`: timestamped observations, dedup/out-of-order rejection, circular yaw, prediction на publish tick, counters, COASTING/LOST и рост covariance.
- Added `ObservationSelector` quality hysteresis for neighboring cameras; active source changes only after the configured quality margin.
- Fusion now exposes validated common `TrackStatus` with state, age, rates, source camera, calibration version and session/reset counter.
- Fusion now resets state/session and per-session counters on calibration-version change, preventing geometry from mixing calibration sets.
- ROS message packages и реальный Gazebo capture adapter пока отсутствуют; launch — декларативный контракт.
- S08 image adapter теперь runnable against Gazebo topics и даёт detector→PnP observations на baseline camera_1; no measured trajectory accuracy or ROS odometry claim is made.
- Gazebo S03 adapter остаётся dependency-free boundary/профилем: полноценный ROS bridge, CameraInfo runtime capture, simulation-time/RTF counter и калибровочная мишень с реальной наблюдаемостью ещё не реализованы.
- Измеренный wall FPS baseline в smoke-check ниже номинала; capture FPS, detector FPS, accepted measurement Hz, output Hz, wall time, simulation time и RTF не смешиваются и не заявляются достигнутыми. Accuracy, calibration quality и LOST runtime behavior не измерялись; SIM_ACCEPTED не объявлен.
- Профили узкой оптики честно имеют статус `coverage: not demonstrated`; нулевое геометрическое покрытие baseline не означает пиксельную читаемость или отсутствие occlusion.
- Завершение `check_sim.py` сопровождается известным pybind11/GIL abort при завершении подписчиков; основной smoke-check до этого подтверждает шесть RGB topics и две позы.
- Физические камеры, IMX296/libcamera/udev/trigger и драйверы не подключались.
- Аппаратная фаза H01–H04 не начиналась.
- Hardware-only unresolved items are tracked in `hardware_backlog.md`; no SIM result is promoted to hardware verification.

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
- `docs/localization_plan/research_comparison_report.md`
- `docs/localization_plan/operator_runbook_sim.md`
- `simulation/pipeline.py`, `tests/test_pipeline.py`
- `simulation/acceptance_matrix.py`, `tests/test_acceptance_matrix.py`
- `simulation/scenarios.py`, `tests/test_scenarios_manifest.py`
- `launch/simulation.launch.json`, `launch/replay.launch.json`, `tests/test_launch_manifests.py`
- `scripts/run_acceptance.py`, `tests/test_acceptance_runner.py`
- `docs/localization_plan/s15_gap_register.md`
- `scripts/serve_dashboard.py`, `tests/test_serve_dashboard.py`
- [sim_baseline_report.md](sim_baseline_report.md)
- `config/simulation_profiles.json`, `simulation/{faults.py,evaluator.py}`, `tests/test_simulation.py`

## Исследовательский профиль

Профиль `ceiling_grid_baseline`; S03 добавляет декларативные `demo_baseline`/`imx296_*` профили и fault injection. R02/R04 учтены marker registry, 6D/attitude validity, timestamp uncertainty, covariance units, раздельными observation/video planes и отсутствием ground truth в Observation. R01/R03/R05/R06 не заявлены проверенными измерениями; фактические оптические профили и RTF ещё не приняты.

## Следующий этап

**S15** — acceptance report: `docs/localization_plan/sim_acceptance_report.md`; результат NOT ACCEPTED, SIM_ACCEPTED не объявлен.

Аппаратную фазу H01–H04 не начинать.
