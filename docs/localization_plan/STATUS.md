# Localization plan status

Текущая фаза: **SIMULATION**  
Текущий этап: **S02 — общие контракты, конфигурация и launch**
Статус: **выполнен; SIM_ACCEPTED не объявлен**

## Сделано

- Добавлен пакет `localization_contracts` с общими контрактами sim/replay и будущего hardware adapter: `Observation`, `TrackStatus`, `CameraStatus`, `CalibrationSet`, `Frame`, `CaptureAdapter`.
- Зафиксированы schema version `1.0`, frame IDs, метрические единицы, clock domains, timestamps, covariance и явная валидность ориентации.
- `Observation` не содержит ground truth; ground truth остаётся evaluation-only. Detector/tracker в S02 не создавались.
- Добавлены marker family/ID, bundle, placement, 6D pose validity и attitude state.
- Добавлена [config/contracts.json](../../config/contracts.json): разделение observation/video planes, bounded queue, sim clock, reset-on-backward-jump и частоты publish/prediction.
- Добавлен `ConfigStore`: validate → stage → atomic apply → rollback и SHA-256 digest.
- Добавлен [launch/contracts.launch.json](../../launch/contracts.launch.json) без драйверов и физических камер.
- Добавлено преобразование twist и 6x6 covariance между frame semantics.

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

gz sdf -k worlds/mocap_arena.sdf
  PASS — Valid.
```

Gazebo baseline не изменён; SDF проверен после изменений. Новые контракты не подключают ground-truth topics в рабочий pipeline.

## Ограничения

- Это S02-каркас: detector, tracker, fusion, odometry publisher, recorder и полноценный replay runtime ещё не реализованы; фиктивные детекторы не добавлялись.
- ROS message packages и реальный Gazebo capture adapter пока отсутствуют; launch — декларативный контракт.
- Частоты, latency, RTF, accuracy, calibration quality и LOST runtime behavior не измерялись; SIM_ACCEPTED не объявлен.
- Физические камеры, IMX296/libcamera/udev/trigger и драйверы не подключались.
- Аппаратная фаза H01–H04 не начиналась.

## Артефакты

- `localization_contracts/{contracts.py,config.py,geometry.py,adapters.py}`
- `config/contracts.json`, `launch/contracts.launch.json`, `tests/test_contracts.py`
- [sim_baseline_report.md](sim_baseline_report.md)

## Исследовательский профиль

Профиль `ceiling_grid_baseline`. R02/R04 учтены marker registry, 6D/attitude validity, timestamp uncertainty, covariance units, раздельными observation/video planes и отсутствием ground truth в Observation. R01/R03/R05/R06 не заявлены проверенными измерениями.

## Следующий этап

**S03** — следующий промпт из `docs/localization_plan/prompts/simulation/` (уточнить точное имя перед началом). S03 не начинался.

Аппаратную фазу H01–H04 не начинать.
