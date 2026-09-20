# S01 — baseline симулятора и бюджет производительности

Дата: 2026-09-21  
Профиль: `ceiling_grid_baseline`, Gazebo Harmonic 8.15.0, seed 42, `steel`,
`colored`, intensity 1.0, установка камер с ошибками из `config/scenario.json`.

## Что реально проверено

Команды:

```text
gz sdf -k worlds/mocap_arena.sdf
/usr/bin/python3 -m unittest discover -s tests -v
python3 scripts/generate_world.py --style steel --lighting colored --seed 42 --output-dir /tmp/mocap-s01-baseline
./scripts/run.sh -s --headless-rendering
/usr/bin/python3 scripts/check_sim.py
```

Результаты:

- SDF валиден; 4/4 unit-теста прошли.
- Генератор воспроизводим для seed 42; геометрическая проверка покрытия дала
  0 непокрытых samples на Z=0, 0.3654 и 0.5 м (шаг 0.1 м, без occlusion).
- При runtime обнаружены все шесть RGB topics и оба world-pose topics.
- `check_sim.py`: 1600×1200 RGB для шести камер, позы обоих роверов,
  snapshots в `/tmp/mocap-camera-check`; команда завершилась с кодом 0.
- Модели роверов имеют pose и odometry topics; это не готовые локализованные
  измерения.

## Текущий поток и измерения

Сейчас реализован только Gazebo capture/world truth. Детектор AprilTag, YOLO,
трекеры, fusion, state publisher, recorder/replay и измеритель RTF отсутствуют.
Ground truth отделён топиками `/model/*/pose` и не подаётся в рабочий tracker,
поскольку такого tracker пока нет.

Конфигурация задаёт 1600×1200, 15 Гц simulation camera rate; целевые 30 Гц
состояния своего ровера и ≥10 Гц состояния второго ещё не реализованы. При
runtime `gz topic -f` наблюдал интервалы с большими разбросами (окна всего по
10 samples; например image camera_1: 0.118–0.642 с в одном запуске), поэтому
это не подтверждение capture FPS. Нужен отдельный monotonic wall-clock
benchmark с подсчётом каждого сообщения, capture timestamp, drops и RTF.

CPU по процессу arena не зафиксирован надёжно: в системе одновременно работал
другой `gz sim`, а sampling был прерван timeout. GPU/VRAM недоступны:
`nvidia-smi` сообщает `Failed to initialize NVML: Driver/library version mismatch`
(NVML 580.178); Gazebo использовал локальный workaround библиотек 595.91.07.
Эти условия занесены как ограничение, а не как оценка производительности.

## Бюджет capture → detect → fusion → publish

| Стадия | Baseline S01 | Требуемая проверка дальше |
|---|---|---|
| Capture | 6×1600×1200, nominal 15 Гц; runtime wall rate не принят | S03/S05: независимые capture FPS, timestamps, drops, RTF |
| AprilTag detect | отсутствует | S08/S14: новые observations ≥20 Гц в штатном профиле |
| YOLO detect | отсутствует, веса не включены | S10/S11/S14: новый результат ≥10 Гц, confidence sweep |
| Fusion | отсутствует | S09: состояние своего 30 Гц, prediction отдельно от observations |
| Publish | отсутствует | S09/S11: output Hz, age, P99 interval; не выдавать prediction за measurement |
| Preview/record/replay | отсутствуют | S05/S13/S14: сравнение без UI, с 6 preview и recording |

План достижения частот: сначала измерить 30 Гц capture и wall-time budget на
одной машине; затем ограничить очереди и измерить AprilTag detector отдельно;
для второго ровера подобрать YOLO batch/ROI и подтвердить ≥10 Гц новых
детекций; после этого запустить fusion на фиксированном publish tick 30/15 Гц
с отдельными counters `capture`, `detector`, `accepted_measurement`, `output`.
Прогноз и coasting не считаются новой observation.

## Сценарии и готовность данных

Готовы: покой/стартовые позы, команды `cmd_vel`, краткое движение через
`check_sim.py --move`, пять вариантов материала, neutral/colored lighting,
seed и camera installation faults, inverted tag generation, nominal и ground
truth camera JSON. Не готовы: траектории circles/eights/швы, controlled frame
drop/reorder/latency, `/clock` reset/seek, записи и replay, синтетические
AprilTag/YOLO measurement logs, accuracy evaluator и независимые test/validation
episodes.

## R01/R03: монтажные профили и footprints

Это план сравнения, не утверждение физической реализуемости:

| Профиль | Состав | Статус S01 |
|---|---|---|
| `ceiling_grid_baseline` | текущие 6 надирных камер | реализован в Gazebo; покрытие только геометрическое |
| `hybrid_research` | 2 камеры (4,6,2.9)/(8,6,2.9) и 4 у середины стен | не реализован; центральная опора — открытый hardware-вопрос |
| `perimeter_only` | 6 камер на допустимых верхних углах/рёбрах | не реализован; нужен ray/occlusion/angle sweep |

Для исследовательского pinhole пересчёта из R01 при Sx=5.02 мм, Sy=3.75 мм,
f=2 мм, z_camera=2.9 м получаем приблизительно 6.65×4.97 м на высоте метки
0.25 м и 6.36×4.75 м на 0.3654 м. Это не заменяет calibrated boundary rays
широкоугольной камеры и не равно гарантированной детекции. Текущая конфигурация
Gazebo использует другую HFOV-модель и заявляет идеальное поле 8.2×6.15 м;
расхождение должно быть разобрано в S06, не скрыто усреднением.

## Ограничения и следующий шаг

S01 не подтверждает целевые частоты, accuracy, доступность, latency, RTF,
CPU/GPU budget или replay readiness. Следующий этап: **S02 — общие контракты,
конфигурация и launch**, путь `docs/localization_plan/prompts/simulation/02_contracts.md`.
