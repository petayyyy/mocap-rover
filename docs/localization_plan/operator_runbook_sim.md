# Запуск и продолжение SIM-фазы

**Сейчас пользователь запретил новые прогоны Gazebo: он занят другой задачей.**
Не запускать команды раздела Gazebo до явного разрешения. Не останавливать
чужой экземпляр симулятора. Текущее доказательство и ограничения:
[post_reboot_progress.md](reports/post_reboot_progress.md).

## Автономные проверки — без Gazebo

Из корня проекта:

```bash
/usr/bin/python3 -m unittest discover -s tests -q
.venv/bin/python -m unittest discover -s tests -q
/usr/bin/python3 scripts/check_dashboard_offline.py
/usr/bin/python3 scripts/evaluate_recording.py \
  --runtime artifacts/post-reboot/live_static \
  --truth artifacts/post-reboot/static_truth3.jsonl \
  --output artifacts/post-reboot/static_evaluation.json
```

Последняя команда должна оставить точность непринятой: в старой записи truth
неверные timestamps. Offline browser использует только сохранённые файлы;
это не live-приёмка панели или производительности.

## Зависимости

Среда `.venv` уже установлена. Для воспроизведения на этой Ubuntu:

```bash
/usr/bin/python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-simulation.txt
```

Gazebo Python bindings используются из системных пакетов. Системные драйверы
NVIDIA, IMX296, udev и libcamera этим не меняются.

## Gazebo — только после разрешения

Терминал 1 — GUI арены:

```bash
./scripts/run.sh
```

Терминал 2 — локализация и панель, новый каталог для каждой записи:

```bash
./scripts/run_localization.sh \
  --config artifacts/post-reboot/calibrated_cameras.json \
  --weights artifacts/post-reboot/training/rover/weights/best.pt \
  --output artifacts/resumed_static_01 --seconds 75
```

Панель: `http://127.0.0.1:8081/`. Порт 8080 ранее был занят другой службой.
Топики ROS2: `/mocap/tag_rover/odom`, `/mocap/tag_rover/status`,
`/mocap/opponent/odom`, `/mocap/opponent/status`.

Терминал 3 — независимая запись эталона, начать до локализации:

```bash
/usr/bin/python3 scripts/record_gazebo_truth.py \
  --output artifacts/resumed_static_01_truth.jsonl --seconds 90
```

После обоих прогонов:

```bash
/usr/bin/python3 scripts/evaluate_recording.py \
  --runtime artifacts/resumed_static_01 \
  --truth artifacts/resumed_static_01_truth.jsonl \
  --output artifacts/resumed_static_01_evaluation.json
```

Для движущейся проверки рекордеру задаётся `--drive`. Это управление двумя
роверами; не включать его в занятом чужой задачей мире. Тестовый driver вправе
использовать truth для задания траекторий; runtime локализации его не читает.

## Пересборка данных и калибровки

Эти команды меняют объекты/позы в текущей арене, тоже только после разрешения.
Калибровка соответствует конкретной установке камер; после изменения seed,
разрешения или монтажа переснять её.

```bash
/usr/bin/python3 scripts/calibrate_gazebo.py \
  --config config/cameras.json --output artifacts/new_calibration/cameras.json
/usr/bin/python3 scripts/capture_yolo_dataset.py \
  --scenario . --output artifacts/new_dataset --samples 120 --seed 91
```

Обучение не требует запущенного Gazebo:

```bash
.venv/bin/python scripts/train_yolo.py \
  --dataset artifacts/new_dataset \
  --model artifacts/post-reboot/pretrained/yolo11n.pt \
  --output artifacts/new_training --epochs 40
```

## Остановка

Просмотр процессов проекта без остановки:

```bash
./scripts/stop.sh --dry-run
```

Остановка процессов этого проекта и их дочерних процессов:

```bash
./scripts/stop.sh
```

Сначала TERM, через пять секунд KILL для оставшихся. Пока Gazebo занят другой
задачей, эту команду не запускать: если она работает в том же репозитории,
её Gazebo тоже будет выбран. Новые процессы не запускаются автоматически.

## Приёмка

Короткий smoke, fixture UI и unit-тесты не заменяют S14/S15. Публикация 30 Гц
не доказывает ≥20 свежих измерений. Нужны точность по timestamp, wall-time
частоты, деградация, 30-минутный прогон и R01–R06. H01–H04 не начинать.
