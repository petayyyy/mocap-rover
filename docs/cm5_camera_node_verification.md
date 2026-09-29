# Методика проверки камерного узла на CM5 + IMX219-160

Порядок жёсткий: каждый шаг заканчивается числом, и следующий шаг не
начинается, пока число не получено. Все числа переписываются в
`docs/cm4_camera_node_report.md` в колонку «Железо».

## Этап 0. Что нужно на столе

- CM5 на IO-плате, Raspberry Pi OS Bookworm 64-bit, IMX219-160 на разъёме
  CAM (на IO-плате CM5 разъёмы 22-pin, у камеры 15-pin: нужен переходной
  шлейф 22→15).
- Светодиод (белый или красный, яркий) и резистор 220–330 Ом на GPIO 17 и
  GND. Светодиод должен светить прямо в объектив с 10–30 см или в белый лист
  в поле зрения.
- Ethernet: пока LAN-кабелей нет, узел можно проверять по Wi-Fi или через
  USB-Ethernet, но задержки и PTP по Wi-Fi не засчитываются. Для этапов 5–8
  нужен провод, лучше напрямую в ноутбук или через свитч с PTP.
- На ноутбуке репозиторий с `.venv` (тесты там уже зелёные).

Этапы 1–4 делаются с одним узлом и без LAN-кабеля; этапы 5–8 — с проводом;
этап 8 — со вторым узлом.

## Этап 1. Камера и режим (без моего кода)

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-numpy python3-simplejpeg linuxptp ethtool python3-lgpio rsync
rpicam-hello --list-cameras
```

Ожидается строка с `imx219` и режим `1640x1232 [83.70 fps]` в списке
`SRGGB8`. Если камеры нет: в `/boot/firmware/config.txt` должно быть
`camera_auto_detect=1`; на IO-плате CM5 для второго разъёма
`dtoverlay=imx219,cam0`. Перезагрузка.

Затем старый проверенный скрипт:

```bash
python3 imx219_fov_stream.py
curl -s http://localhost:8080/api/status | python3 -m json.tool
```

Ожидается `sensor_fps` ≈ 83, `sensor_config.bit_depth` = 8,
`output_size` = [1640, 1232]. Если 10 бит и 41.8 к/с — libcamera выбрала не
тот режим, дальше идти бессмысленно; проверить версию libcamera
(`libcamera-hello --version`) и picamera2.

Записать: модель CM (`cat /proc/device-tree/model`), версию ОС
(`cat /etc/os-release`), libcamera и picamera2 (`dpkg -l | grep -E "libcamera|picamera2"`).

## Этап 2. Установка узла и первый запуск

С ноутбука:

```bash
rsync -a pi_cam/ pi@<ip-узла>:/tmp/pi_cam/
```

На узле:

```bash
sudo mkdir -p /opt/mocap-rover /etc/mocap-rover
sudo rsync -a --delete /tmp/pi_cam/ /opt/mocap-rover/pi_cam/
sudo cp /opt/mocap-rover/pi_cam/node_config.example.json /etc/mocap-rover/node_config.json
sudo nano /etc/mocap-rover/node_config.json    # camera_id этого узла, exposure_us, analogue_gain
cd /opt/mocap-rover/pi_cam
LIBCAMERA_LOG_LEVELS=RPI:INFO python3 camera_node.py --config /etc/mocap-rover/node_config.json --verbose
```

Ожидается в выводе:

- строка libcamera `Selected sensor format: 1640x1232-SBGGR8_1X8/RAW` — её
  целиком в отчёт;
- моя строка `camera camera_1: 1640x1232, line_time 9452 ns (register_model), port 5600, jpeg via simplejpeg`;
- раз в секунду при `--verbose`: `fps 83.xx missed 0 dropped 0 client None`.

Если узел падает с `libcamera picked sensor mode ...` — режим не тот, см.
этап 1. Если падает внутри `MappedArray` — старая picamera2, я предусмотрел
запасной путь, но пришлите трассировку.

Пусть работает минуту: `missed` должен остаться 0. Затем Ctrl-C и
установить службу:

```bash
sudo cp /opt/mocap-rover/pi_cam/camera_node.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now camera_node
journalctl -u camera_node -f
```

Проверка автоперезапуска: `sudo pkill -9 -f camera_node.py` — через 2 с
служба должна подняться сама (`systemctl status camera_node`).

## Этап 3. Протокол с ноутбука (можно по Wi-Fi)

```bash
.venv/bin/python scripts/lan_camera_timing.py --nodes <ip-узла>:5600 --seconds 30
```

Что смотреть в таблице:

| Строка | Ожидание | Если нет |
|---|---|---|
| `sensor fps` | 83.0 ± 0.1 | режим или FrameDurationLimits |
| `node missed` | 0 | захват не успевает, смотреть `capture_cost_us` в статусе |
| `node dropped` | 0 по проводу; по Wi-Fi может быть > 0 | канал; не дефект узла |
| `window latency` P95 | по Wi-Fi любое; по проводу ≤ 20 мс | этап 5 |
| `node exp->send` P50 | ≈ 12–14 мс (считывание 11.65 мс + экспозиция + окно) | если сильно больше — кодирование/копии, прислать статус |
| `line_time 9452 ns, exposure 800000 ns` | как в конфиге | — |

Дополнительно снять сырой статус и приложить к отчёту:

```bash
.venv/bin/python - <<'EOF'
import sys, json, time; sys.path.insert(0, ".")
from localization_contracts.lan_capture import LanCameraSource
s = LanCameraSource(["<ip-узла>:5600"]); print(s.wait_connected(5)); time.sleep(2.5)
print(json.dumps(s.status("camera_1"), indent=2)); print(json.dumps(s.hello("camera_1"), indent=2)); s.close()
EOF
```

В статусе важны `capture_cost_us` (ожидаю < 500 мкс), `soc_temp_c`,
`throttled.flags` (должно быть пусто), `cpu_percent`.

Проверка картинки, что окна режутся из нужного места: запросить полный
кадр и окно и сохранить оба в PNG:

```bash
.venv/bin/python - <<'EOF'
import sys, time, cv2; sys.path.insert(0, ".")
from localization_contracts.lan_capture import LanCameraSource
s = LanCameraSource(["<ip-узла>:5600"]); s.wait_connected(5)
s.request_windows("camera_1", [(376, 580, 480, 480)]); s.request_full("camera_1")
for _ in range(50):
    g = s.take("camera_1", 0.5)
    if g and any(f.is_full for f in g):
        for f in g: cv2.imwrite(f"/tmp/{'full' if f.is_full else 'window'}.png", f.array)
        full = [f for f in g if f.is_full][0]; win = [f for f in g if not f.is_full][0]
        print("identical:", (full.array[376:856, 580:1060] == win.array).all()); break
s.close()
EOF
```

`identical: True` обязателен: окно и полный кадр одного `stamp_ns` — одни
пиксели. Открыть `/tmp/full.png`: картинка должна быть резкой при
экспозиции 0.8 мс (иначе поднять `analogue_gain` в конфиге, не экспозицию).

## Этап 4. Светодиод: что такое SensorTimestamp и `line_time_ns`

Это главное измерение. На узле, служба остановлена, светодиод на GPIO 17:

```bash
sudo systemctl stop camera_node
cd /opt/mocap-rover/pi_cam
python3 led_timestamp_probe.py --config /etc/mocap-rover/node_config.json \
    --gpio 17 --flashes 50 --pulse-us 100 --output /tmp/led_probe.json
```

Сначала без `--columns`: в логе на каждую вспышку строка
`flash N: rows A..B, flash-stamp X ms`. Если вспышки не видны
(`flash N not seen`) — светодиод тусклый или не в кадре: снять кадр
`rpicam-still -o /tmp/led.jpg --shutter 800 --gain 4`, найти столбцы, где
светодиод, и передать их через `--columns c0,c1` (ширина полосы засветки
должна быть ≈ 85 строк при экспозиции 0.8 мс: 800 мкс / 9.45 мкс).

В конце печатается `fit`. Критерии:

| Поле | Ожидание |
|---|---|
| `fit.reference` | одно из `readout_start_first_row` / `exposure_start_first_row` / `frame_end`, не `unknown` |
| `line_time_error_percent` | в пределах ±5 % (порог ТЗ), ожидаю ±1 % |
| `fit.line_time_from_band_ns` | согласуется с `fit.line_time_ns` в пределах 5 % |
| `fit.residual_p95_ns` | < 100 000 (100 мкс); при 100-мкс импульсе ожидаю 20–50 мкс |
| `misses` | ≤ 5 из 50 |

`recommended_config` переписать в `/etc/mocap-rover/node_config.json`
(`stamp_reference`, `stamp_correction_ns`, `line_time_ns`), запустить
службу, повторить пробу ещё раз на 20 вспышек: после правки конфига
интерпретация не меняется (пробу конфиг не читает для поправки, она меряет
сырой SensorTimestamp), но `line_time_error_percent` должен стать ≈ 0.

Если `reference = unknown` — прислать `/tmp/led_probe.json`, я посмотрю
остатки: возможно, SensorTimestamp на CFE привязан к чему-то ещё, и надо
добавить вариант.

## Этап 5. Задержки по проводу, 10 минут

Только по Ethernet. На ноутбуке статический IP, на узле тоже.

```bash
.venv/bin/python scripts/lan_camera_timing.py --nodes <ip-узла>:5600 \
    --seconds 600 --json artifacts/lan_timing_1node.json | tee artifacts/lan_timing_1node.txt
```

Пороги ТЗ: `sensor fps` ≥ 83, `node missed` 0, `node dropped` 0,
`window latency` P95 ≤ 20 мс, `throttled none`, температура — записать
max. Обратите внимание: без PTP (этап 6) `window latency` включает
разбег часов узла и ноутбука и может быть даже отрицательной; для
приёмки нужен прогон после этапа 6. Прогон до PTP тоже сохранить — он
покажет узел сам по себе (`node exp->send`).

Параллельно на узле `top -d 5` две минуты: загрузка `python3` и
температура `vcgencmd measure_temp` в отчёт.

## Этап 6. PTP

Сначала есть ли аппаратные метки:

```bash
ethtool -T eth0            # узел
ethtool -T <iface>         # ноутбук
```

`PTP Hardware Clock: 0` и `hardware-transmit/receive` — аппаратные метки;
`none` — программные (`-S`). Обе строки в отчёт.

Ноутбук (мастер):

```bash
sudo ptp4l -i <iface> -m --masterOnly 1 --priority1 10 [-S]
sudo phc2sys -c <iface> -s CLOCK_REALTIME -w -m     # только при аппаратных метках
```

Узел:

```bash
sudo ptp4l -i eth0 -m -s [-S]
sudo phc2sys -s eth0 -c CLOCK_REALTIME -w -m        # только при аппаратных метках
pmc -u -b 0 'GET CURRENT_DATA_SET' 'GET PORT_DATA_SET'
```

Ожидание через минуту: `portState SLAVE`, `offsetFromMaster` стабильно
< 10 000 нс при аппаратных, < 200 000 нс при программных метках. В статусе
узла (`lan_camera_timing`) строка `ptp state SLAVE, |offset| P95 ...`.

После этого повторить этап 5 — это и есть приёмочный прогон задержек.

## Этап 7. Запись датасета и реплей

```bash
.venv/bin/python scripts/record_lan_dataset.py --nodes <ip-узла>:5600 \
    --config config/mocap_arena_imx219/runtime_cameras.json \
    --output artifacts/dataset_lan_01 --seconds 60 --divisor 8 --allow-missing
.venv/bin/python scripts/replay_dataset.py artifacts/dataset_lan_01 \
    --output artifacts/replay_lan_01 --cameras camera_1 --no-lidar
```

В `meta.json` → `achieved.camera_1`: `fps` ≈ 10.4, `mbit_s` ≈ 170,
`node_frames_dropped_queue` 0, `laptop_dropped` 0. Если `node_frames_dropped_queue`
> 0 — канал не тянет, поднять `--divisor` до 16 и записать, какой прошёл.
Оценка на шесть камер: `mbit_s × 6` должно быть < 700 Мбит/с.

Реплей должен завершиться без ошибок; маркера в кадре нет, поэтому
`tag_hits` 0 — это нормально, проверяется только чтение формата.

## Этап 8. Два узла и синхронизация

Когда есть второй CM5 (или CM4): этапы 2–4 и 6 на нём, `camera_id: camera_2`.
Обе камеры смотрят на один светодиод; светодиод мигает от третьего Pi или
от любого Pi без службы:

```bash
python3 /opt/mocap-rover/pi_cam/led_timestamp_probe.py --blink-only --gpio 17 --period-s 0.4 --pulse-us 100
```

На ноутбуке, окно у каждой камеры там, где светодиод (см. этап 3, `/tmp/full.png`):

```bash
.venv/bin/python scripts/lan_sync_check.py --nodes <ip1>:5600 <ip2>:5600 \
    --camera-window camera_1=<r,c,480,480> --camera-window camera_2=<r,c,480,480> \
    --flashes 50 --json artifacts/lan_sync.json | tee artifacts/lan_sync.txt
```

Ожидание: `camera_1 - camera_2: n≥50 |dt| P50 ... P95 ... max ...`.
Порог ТЗ P95 ≤ 1000 мкс, цель 100 мкс. `signed median` — постоянный сдвиг;
если он > 200 мкс при малом разбросе, у одного из узлов неверен
`stamp_correction_ns` из этапа 4. Строки `line_time header ... from band ...`
должны расходиться ≤ 5 %.

## Что прислать мне после каждого этапа

1. `journalctl -u camera_node -n 200` (этап 2).
2. Вывод `lan_camera_timing` и JSON (этапы 3, 5, 6).
3. `/tmp/led_probe.json` и обновлённый `node_config.json` (этап 4).
4. `ethtool -T` с обеих сторон и `pmc` (этап 6).
5. `meta.json` записи (этап 7).
6. `lan_sync.txt` и JSON (этап 8).

По этим файлам я заполню отчёт и поправлю код, если что-то разойдётся с
ожиданиями выше.
