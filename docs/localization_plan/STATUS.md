# Localization plan status

Текущая фаза: **SIMULATION**  
Текущий этап: **S03 — адаптер симуляции и профиль камеры**
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
- ROS message packages и реальный Gazebo capture adapter пока отсутствуют; launch — декларативный контракт.
- Gazebo S03 adapter остаётся dependency-free boundary/профилем: полноценный ROS bridge, CameraInfo runtime capture, simulation-time/RTF counter и калибровочная мишень с реальной наблюдаемостью ещё не реализованы.
- Измеренный wall FPS baseline в smoke-check ниже номинала; capture FPS, detector FPS, accepted measurement Hz, output Hz, wall time, simulation time и RTF не смешиваются и не заявляются достигнутыми. Accuracy, calibration quality и LOST runtime behavior не измерялись; SIM_ACCEPTED не объявлен.
- Профили узкой оптики честно имеют статус `coverage: not demonstrated`; нулевое геометрическое покрытие baseline не означает пиксельную читаемость или отсутствие occlusion.
- Завершение `check_sim.py` сопровождается известным pybind11/GIL abort при завершении подписчиков; основной smoke-check до этого подтверждает шесть RGB topics и две позы.
- Физические камеры, IMX296/libcamera/udev/trigger и драйверы не подключались.
- Аппаратная фаза H01–H04 не начиналась.

## Артефакты

- `localization_contracts/{contracts.py,config.py,geometry.py,adapters.py}`
- `config/contracts.json`, `launch/contracts.launch.json`, `tests/test_contracts.py`
- [sim_baseline_report.md](sim_baseline_report.md)
- `config/simulation_profiles.json`, `simulation/{faults.py,evaluator.py}`, `tests/test_simulation.py`

## Исследовательский профиль

Профиль `ceiling_grid_baseline`; S03 добавляет декларативные `demo_baseline`/`imx296_*` профили и fault injection. R02/R04 учтены marker registry, 6D/attitude validity, timestamp uncertainty, covariance units, раздельными observation/video planes и отсутствием ground truth в Observation. R01/R03/R05/R06 не заявлены проверенными измерениями; фактические оптические профили и RTF ещё не приняты.

## Следующий этап

**S04** — следующий промпт: `docs/localization_plan/prompts/simulation/04_registry.md` (переход не выполнялся).

Аппаратную фазу H01–H04 не начинать.
