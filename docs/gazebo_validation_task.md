# ТЗ: прогон в Gazebo и проверка ветки `fix/camera-model-and-async-fusion`

Это задание для исполнителя, который **только запускает и измеряет**. Не
меняй код, не «чини» упавшие проверки, не подбирай пороги. Если что-то
падает — зафиксируй факт, приложи файлы и остановись.

Рабочий каталог: `/home/popka/Drone/mocap-rover`
Ветка: `fix/camera-model-and-async-fusion`

---

## 0. Подготовка (1 минута)

```bash
cd /home/popka/Drone/mocap-rover && git branch --show-current && git status --short | head
```

Ожидается `fix/camera-model-and-async-fusion`. Если ветка другая:

```bash
cd /home/popka/Drone/mocap-rover && git checkout fix/camera-model-and-async-fusion
```

---

## Шаг 1. Офлайн-проверки (обязательны до Gazebo)

### 1.1 Юнит-тесты

```bash
cd /home/popka/Drone/mocap-rover && .venv/bin/python -m pytest tests -q 2>&1 | tail -5
```

**Ожидается:** `206 passed, 2 skipped` и ровно один провал —
`tests/test_scenarios.py::Scenarios::test_styles_preserve_installation`.
Этот провал существовал до изменений (`gz sdf -k` возвращает 255 в этом
окружении) и не является регрессией.

**Если провалов больше одного** — запиши их имена и `tail -40` вывода,
дальше не иди.

### 1.2 Валидатор модели камеры

```bash
cd /home/popka/Drone/mocap-rover && .venv/bin/python scripts/validate_camera_model.py --config config/cameras_ground_truth.json
```

**Ожидается:** шесть строк `PASS`, `valid_area=100.0%`, код возврата 0.

Контрольный негативный прогон — старая калибровка обязана падать:

```bash
cd /home/popka/Drone/mocap-rover && .venv/bin/python scripts/validate_camera_model.py --config artifacts/tag_coverage_20260925_181432/camera_config.json; echo "exit=$?"
```

**Ожидается:** шесть `FAIL`, `valid_area=42.8%`, `exit=1`.

---

## Шаг 2. Основной прогон в Gazebo (0.5 м/с)

Одна команда. Займёт примерно 10–13 минут.

```bash
cd /home/popka/Drone/mocap-rover && source /opt/ros/jazzy/setup.bash && .venv/bin/python scripts/run_tag_coverage_experiment.py --output artifacts/fix_v1_slow --speed 0.5 --lane-spacing 0.5 --max-seconds 720 --headless 2>&1 | tail -40
```

Пока идёт — ничего не трогай. Gazebo и рантайм запускаются скриптом сами.

### Если не стартует

- `RuntimeError: Gazebo did not publish tag_rover pose within 45 seconds` —
  проверь, что не осталось запущенного `gz sim`:
  `pkill -f 'gz sim'; sleep 5` и повтори команду.
- `localization exited during startup` — приложи
  `artifacts/fix_v1_slow/localization.log` целиком.
- `localization runtime already running` — `pkill -f run_localization.py`,
  затем повтори.

### После завершения

```bash
cd /home/popka/Drone/mocap-rover && cat artifacts/fix_v1_slow/summary.json
```

---

## Шаг 3. Что именно проверить в `summary.json`

| Поле | Порог | Смысл |
|---|---|---|
| `valid_coverage` | **≥ 0.90** | доля сэмплов с валидной позицией |
| `p50_fused_xy_error_m` | **≤ 0.08** | |
| `p95_fused_xy_error_m` | **≤ 0.20** | |
| `p95_fused_yaw_error_deg` | **≤ 10.0** | |
| `p50_measurement_age_ms` | **≤ 80** | |
| `p95_measurement_age_ms` | **≤ 150** | ниже 66 мс физически нельзя: камеры 15 Гц |
| `id_rejections` | **= 0** | |
| `tracking_states` | `LOST` — только в первые ~0.2 с | |

Для сравнения, тот же прогон до изменений
(`artifacts/tag_coverage_20260925_181432/summary.json`):

```text
valid_estimate_samples = 28 из 21361   (0.13 %)
p50_fused_xy_error_m   = 0.1354
p95_fused_xy_error_m   = 0.8885
p95 measurement age    = 18286 мс
```

Выпиши обе колонки рядом в отчёт.

---

## Шаг 4. Две проверки по CSV

Скопируй команду целиком, она печатает готовые строки отчёта.

```bash
cd /home/popka/Drone/mocap-rover && .venv/bin/python - <<'PY'
import csv, json, math, collections
RUN='artifacts/fix_v1_slow'
def pct(v,p):
    v=sorted(v)
    if not v: return float('nan')
    i=(len(v)-1)*p/100; lo=int(i); hi=min(lo+1,len(v)-1); a=i-lo
    return v[lo]*(1-a)+v[hi]*a
def num(row,key):
    try: return float(row[key])
    except (KeyError, TypeError, ValueError): return None

print('=== 1. PnP-покрытие: детекция против решения ===')
frames=list(csv.DictReader(open(f'{RUN}/camera_frames.csv')))
by=collections.defaultdict(lambda:[0,0])
for r in frames:
    s=by[r['capture_ns']]
    s[0]|= int(r['detections'])>0
    s[1]|= int(r['pnp_valid'])>0
n=len(by)
print(f'  моментов времени: {n}')
print(f'  >=1 камера детектирует: {100*sum(v[0] for v in by.values())/n:.1f} %  (было 94.2)')
print(f'  >=1 камера решает PnP : {100*sum(v[1] for v in by.values())/n:.1f} %  (было 45.2)  ЦЕЛЬ >= 85')

print()
print('=== 2. Физически невозможная высота базы ===')
obs=list(csv.DictReader(open(f'{RUN}/observations.csv')))
z=[num(r,'pnp_base_z_m') for r in obs]
z=[v for v in z if v is not None]
if z:
    bad=sum(abs(v-0.14)>0.30 for v in z)
    print(f'  наблюдений: {len(z)}; |z-0.14|>0.30 м: {bad} ({100*bad/len(z):.1f} %)  (было 39.5 %)  ЦЕЛЬ < 2 %')
    print(f'  медиана z: {pct(z,50):.3f} м (истина 0.140)  ЦЕЛЬ 0.12..0.16')
else:
    print('  ПОЛЕ pnp_base_z_m ОТСУТСТВУЕТ — приложи первую строку observations.csv')

print()
print('=== 3. Ошибка наблюдений по камерам ===')
per=collections.defaultdict(list)
for r in obs:
    e=num(r,'xy_error_m')
    if e is not None: per[r['camera_id']].append(e)
for cam in sorted(per):
    v=per[cam]
    print(f'  {cam}: n={len(v):5d} P50={pct(v,50):.3f} P95={pct(v,95):.3f}')
print('  (было: camera_1..4 P95 ~0.079; camera_5/6 P95 0.850/0.871)')

print()
print('=== 4. Задержка и режимы камер ===')
lat=[num(r,'latency_ms') for r in frames]
lat=[v for v in lat if v is not None]
print(f'  детектор+PnP на кадр: P50={pct(lat,50):.1f} мс P95={pct(lat,95):.1f} мс  (было 150/200)  ЦЕЛЬ P95 < 25')
modes=collections.Counter(r.get('mode','') for r in frames)
print(f'  режимы кадров: {dict(modes)}')

print()
print('=== 5. Разрывы валидной позиции ===')
est=list(csv.DictReader(open(f'{RUN}/estimates.csv')))
gaps=[]; cur=0; prev=None
for r in est:
    t=int(r['stamp_ns'])
    if r['valid']=='1':
        if cur: gaps.append(cur)
        cur=0
    else:
        cur += (t-prev) if prev is not None else 0
    prev=t
if cur: gaps.append(cur)
print(f'  сэмплов: {len(est)}, валидных: {sum(r["valid"]=="1" for r in est)}')
print(f'  разрывов: {len(gaps)}; самый длинный: {max(gaps)/1e6 if gaps else 0:.0f} мс  ЦЕЛЬ < 1500')
states=collections.Counter(r['tracking_state'] for r in est)
print(f'  состояния: {dict(states)}')
PY
```

---

## Шаг 5. Контрольный прогон A/B без ROI (по возможности)

Нужен, чтобы отделить эффект ROI-трекинга от остального. Ещё ~10 минут.

```bash
cd /home/popka/Drone/mocap-rover && source /opt/ros/jazzy/setup.bash && .venv/bin/python scripts/run_tag_coverage_experiment.py --output artifacts/fix_v1_noroi --speed 0.5 --lane-spacing 0.5 --max-seconds 720 --headless --no-roi-tracking 2>&1 | tail -20
```

Затем:

```bash
cd /home/popka/Drone/mocap-rover && for d in fix_v1_slow fix_v1_noroi; do echo "-- $d"; .venv/bin/python -c "import json;d=json.load(open('artifacts/$d/summary.json'));print({k:d[k] for k in ('valid_coverage','p50_fused_xy_error_m','p95_fused_xy_error_m','p50_measurement_age_ms','p95_measurement_age_ms')})"; done
```

---

## Шаг 6. Что прислать

1. Вывод шага 1.1 (последние 5 строк) и обоих прогонов шага 1.2.
2. `artifacts/fix_v1_slow/summary.json` целиком.
3. `artifacts/fix_v1_slow/manifest.json` целиком.
4. Весь вывод скрипта из шага 4.
5. Таблицу шага 5, если он выполнен.
6. `artifacts/fix_v1_slow/localization.log` — **всегда**, даже при успехе.
7. Если хоть один порог не выполнен — дополнительно:
   `artifacts/fix_v1_slow/camera_summary.csv` и первые 3 строки
   `artifacts/fix_v1_slow/observations.csv`.

---

## Чего делать НЕЛЬЗЯ

- Не менять пороги в коде и не править тесты, чтобы они прошли.
- Не увеличивать `--coast-ms`, `--lost-ms` или `--identity-max-age-s`,
  чтобы поднять `valid_coverage`. Это ровно та подмена, которую вся работа
  устраняет: устаревшая позиция не становится верной от смягчения таймаута.
- Не запускать с `--xy-source pnp` в основном прогоне.
- Не коммитить каталог `artifacts/` (он в `.gitignore`).
- Не делать выводов о геометрии камер по одному прогону.

## Известные ограничения этого прогона

- Лидар Unitree L2 **не подключён** к рантайму. Всё измеренное — чисто
  визуальная одометрия по метке.
- `valid_coverage` порядка 0.93–0.95 ожидаем: в покрытии шести камер есть
  геометрическая дыра примерно на 0.7 с, которую камеры закрыть не могут.
  Если разрыв один и короче 1.5 с — это не дефект софта.
- Камеры в мире работают на 15 Гц, поэтому `measurement_age` физически не
  может быть ниже 66 мс. Порог 50 мс из черновика критериев приёмки
  недостижим при этой частоте кадров.
