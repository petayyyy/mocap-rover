# Задание: шесть CM4 + IMX219-160 на коммутаторе — живая проверка канала, окон и узлов

Для коллеги, у которого стенд: 6 × Raspberry Pi CM4 на носителях
Waveshare CM4-NANO-B, 6 × Waveshare IMX219-160, коммутатор Keenetic PoE+
Switch 9 (KN-4710, 1 Гбит, неуправляемый), PoE-сплиттеры на 5 В, ноутбук.
Всё нужное ПО уже в репозитории `mocap-rover`: узел `pi_cam/camera_node.py`,
приёмник `localization_contracts/lan_capture.py`, инструменты замеров в
`scripts/`. Один узел (CM4 и CM5) уже проверен; теперь нужно то, что
можно измерить только на шести.

## Что уже известно и не проверяется заново

- Режим сенсора только `1640x1232`, 8 бит, 83.7 к/с (меньшие режимы
  режут поле зрения 160°). На живом CM4 сенсор держит 83.0 к/с, 10 минут,
  2 пропуска из 49 822 кадров.
- На CM4 ISP отдаёт только каждый второй кадр, поэтому узел читает сырой
  байеровский поток (`"stream": "raw"`, на провод идёт `bayer8`). Пол
  задержки платформы 22.6 мс от начала экспозиции до кадра в программе
  (на CM5 — 15.3 мс).
- `SensorTimestamp` = начало считывания первой строки; `line_time_ns`
  = 9452 нс по регистровой модели, светодиодом ещё не подтверждён.
- У CM4/CM5 аппаратные метки PTP есть, программные драйвер не
  поддерживает; `ptp4l` поднимается только с `--tx_timestamp_timeout 50`.
  Мастером PTP должен быть один из узлов, не ноутбук.
- Решение по окнам: маркер до 320×320 без потерь, соперник до 480×480 в
  JPEG q90 (до 640 при нехватке), полный кадр только раз в 2 с на камеру
  и для записи датасетов с делителем. Расчёт: шесть камер, маркер 320
  bayer8 + соперник 480 JPEG ≈ 571 Мбит/с из 940 доступных, задержка
  ≈ 27.5 мс. Это надо подтвердить живым замером — ради этого стенд.
- При экспозиции 0.8 мс и усилении 4 в обычной комнате кадр почти чёрный
  (18/255). Для замеров канала это не мешает; для чтения маркера нужен
  свет или временно экспозиция 2–4 мс (в конфиге `exposure_us`).

## 0. Ноутбук

1. **Гигабитный адаптер обязателен.** 100-мегабитный USB-адаптер
   блокирует всё: шесть камер в 100 Мбит не влезают ни в каком режиме.
   Нужен USB 3 → 1 GbE (Realtek RTL8153/8156 или аналог). Проверка:
   `ethtool <iface> | grep Speed` → `1000Mb/s`.
2. Ubuntu/Debian, клон репозитория, окружение:
   ```bash
   git clone <URL> ~/mocap-rover && cd ~/mocap-rover
   python3 -m venv --system-site-packages .venv && .venv/bin/python -m pip install numpy==1.26.4 opencv-python==4.10.0.84 simplejpeg
   .venv/bin/python -m pytest tests/test_lan_capture.py -q      # без железа, должно быть зелёным
   ```
3. Статические адреса: ноутбук `192.168.10.1/24`, узлы
   `192.168.10.101…106` (camera_1…camera_6). Коммутатор неуправляемый,
   адреса задаются на узлах.

## 1. Образ и настройка каждого CM4 (повторить шесть раз)

1. **Образ**: Raspberry Pi OS **Lite 64-bit** на базе Debian 12 (bookworm),
   свежий с raspberrypi.com. Именно Bookworm: `picamera2` и `libcamera`
   ставятся из его репозитория; Bullseye не подходит. Записать через
   Raspberry Pi Imager (для CM4 Lite — на microSD носителя; для CM4 с eMMC —
   через `rpiboot` и USB носителя). В Imager задать: hostname `cam-N`,
   пользователь `pi`, SSH включён, Wi-Fi не нужен.
2. Первая загрузка, обновление, пакеты:
   ```bash
   sudo apt update && sudo apt full-upgrade -y
   sudo apt install -y python3-picamera2 python3-numpy python3-simplejpeg linuxptp ethtool python3-lgpio git rsync
   ```
3. **Камера на носителе CM4-NANO-B не находится автоматически.** В
   `/boot/firmware/config.txt`:
   ```text
   camera_auto_detect=0
   dtoverlay=imx219,cam0
   ```
   Перезагрузка. Проверка: `rpicam-hello --list-cameras` показывает imx219 и
   режим `1640x1232` (SBGGR8). Шлейф не подключать на ходу.
4. Статический адрес (пример для camera_1), `/etc/dhcpcd.conf` или
   NetworkManager (`nmcli con mod "Wired connection 1" ipv4.method manual
   ipv4.addresses 192.168.10.101/24`), перезагрузка, `ping 192.168.10.1`.
5. Узел:
   ```bash
   git clone <URL> ~/mocap-rover
   sudo mkdir -p /opt/mocap-rover /etc/mocap-rover
   sudo rsync -a --delete ~/mocap-rover/pi_cam/ /opt/mocap-rover/pi_cam/
   sudo cp ~/mocap-rover/pi_cam/node_config.example.json /etc/mocap-rover/node_config.json
   ```
   В `/etc/mocap-rover/node_config.json` для CM4 обязательно:
   `"camera_id": "camera_N"`, `"stream": "raw"`, `"buffer_count": 4`,
   `"exposure_us": 800`, `"analogue_gain": 4.0`, `"send_queue": 3`.
   Служба:
   ```bash
   sudo cp ~/mocap-rover/pi_cam/camera_node.service /etc/systemd/system/
   sudo systemctl daemon-reload && sudo systemctl enable --now camera_node
   journalctl -u camera_node -n 30      # ждать строку Selected sensor format: 1640x1232-SBGGR8_1X8
   ```
6. **PTP.** Проверить метки: `ethtool -T eth0` должен показать
   `PTP Hardware Clock: 0`. Узел camera_1 — мастер, остальные — slave.
   `/etc/linuxptp/ptp4l.conf` на всех: добавить `tx_timestamp_timeout 50`;
   на camera_1 `masterOnly 1`, `priority1 10`; на остальных `slaveOnly 1`.
   ```bash
   sudo systemctl enable --now ptp4l phc2sys
   sudo pmc -u -b 0 'GET CURRENT_DATA_SET'        # offsetFromMaster на slave — единицы мкс
   ```
   Если `ptp4l` уходит в `FAULTY`, значит таймаут не применился. Одна
   строка NOPASSWD для `pmc` в sudoers (узел опрашивает его под `pi`):
   `pi ALL=(root) NOPASSWD: /usr/sbin/pmc`.
7. Проверка одного узла с ноутбука перед тем, как подключать следующий:
   ```bash
   .venv/bin/python scripts/lan_camera_timing.py --nodes 192.168.10.101:5600 --seconds 60
   ```
   Ожидается `sensor fps 83.0`, `node missed` ≈ 0, `throttled none`,
   температура < 70 °C. Записать вывод.

Отдельно на **одном** узле (служба остановлена) — светодиод на GPIO 17,
светящий в объектив, для привязки времени и `line_time_ns`:
```bash
sudo systemctl stop camera_node
python3 /opt/mocap-rover/pi_cam/led_timestamp_probe.py --config /etc/mocap-rover/node_config.json --gpio 17 --flashes 50 --pulse-us 100 --columns 780,860 --output /tmp/led_probe.json
sudo systemctl start camera_node
```
Результат `recommended_config` переписать во все шесть конфигов
(`stamp_reference`, `stamp_correction_ns`, `line_time_ns`) и приложить
JSON к отчёту. Это единственный замер, требующий пайки: светодиод +
резистор 220 Ом на GPIO 17 и GND.

## 2. Что измерить на шести узлах

Все замеры — с ноутбука, инструментом `scripts/lan_camera_timing.py`; JSON
каждого прогона сохранять в `artifacts/field_<дата>/` (в git не попадает),
текстовый вывод — в отчёт целиком. Перед каждым прогоном все шесть узлов в
`systemctl status camera_node` активны и `pmc` показывает slave-узлы в
состоянии `SLAVE`.

| № | Режим | Команда (узлы `N=192.168.10.101:5600 … 192.168.10.106:5600`) | Что ждём |
|---|---|---|---|
| A | по одному узлу, 2 окна 480 JPEG, 60 с | `lan_camera_timing.py --nodes <один> --seconds 60` (×6) | 83 к/с, задержка P95 ≈ 28–35 мс на каждом |
| B | 6 узлов, 2 окна 240×240 bayer8, 10 мин | `--nodes $N --seconds 600 --window-size 240 --window-format y8 --json artifacts/field_<дата>/B.json` | 459 Мбит/с, 0 дропов, P95 ≈ 24 мс |
| C | 6 узлов, 2 окна 320×320 bayer8, 10 мин | `--window-size 320 --window-format y8` | 816 Мбит/с, 87 % порта: считаем дропы и хвост задержки |
| D | 6 узлов, 2 окна 480×480 JPEG, 10 мин | `--window-size 480 --window-format jpeg` | 327 Мбит/с, P95 ≈ 28 мс, CPU ноутбука |
| E | 6 узлов, 2 окна 480×480 bayer8, 2 мин | `--window-size 480 --window-format y8` | не влезает (1837 Мбит/с): сколько сбрасывают узлы, что с задержкой |
| F | 6 узлов, полный кадр каждый 12-й, bayer8, 2 мин | `--stream-full 12 --full-format y8` | 671 Мбит/с, задержка полного кадра |
| G | запись датасета, 60 с | `record_lan_dataset.py --nodes $N --config config/mocap_arena_imx219/runtime_cameras.json --output artifacts/field_<дата>/dataset_lan --seconds 60 --divisor 12` | `meta.json` → `achieved`, читается `replay_dataset.py` |
| H | синхронизация узлов, светодиод виден двум камерам | `lan_sync_check.py --nodes <два> --window 376,580,480,480 --columns 200,280 --flashes 50 --json artifacts/field_<дата>/H.json`, светодиод мигает с третьего Pi: `led_timestamp_probe.py --blink-only --gpio 17 --period-s 0.4` | P95 ≤ 1 мс, цель 100 мкс; повторить для трёх пар |

Во всех прогонах два окна на камеру (умолчание инструмента), полный кадр раз в 2 с (`--full-period 2`).

В каждом 10-минутном прогоне записать: `sensor fps`, `node missed`,
`node dropped`, `laptop drops`, задержки окна P50/P95/max, `send->receive`,
загрузку порта (`link`, Мбит/с), CPU ноутбука P50/P95, температуру и
`throttled` каждого узла, состояние PTP (`offset`). Отдельно: `top` на
ноутбуке во время D и `ethtool -S <iface> | grep -i drop` до и после.

## 3. Что не делать

Не менять код узла и приёмника (нашли дефект — описать в отчёте с логом,
не чинить молча). Не менять режим сенсора, экспозицию ниже 0.5 мс и выше
4 мс, `stream` на `main`. Не ставить ноутбук мастером PTP. Не запускать
два `lan_camera_timing.py` одновременно.

## 4. Отчёт

Файл `docs/field_report_<дата>.md` в репозитории (коммит в ветку
`field/<дата>`), JSON и логи — архивом `artifacts/field_<дата>.tar.gz`
отдельно. Структура:

1. Железо и ПО: модель носителя, ревизия CM4, образ и дата, версии
   `libcamera`/`picamera2`/`linuxptp` (`dpkg -l | grep -E "libcamera|picamera2|linuxptp"`),
   адаптер ноутбука и `ethtool` (скорость, `-T`), ядро ноутбука.
2. Настройка узла: `config.txt`, `node_config.json`, `ptp4l.conf` одного
   узла целиком; `journalctl -u camera_node -n 30` одного узла.
3. Таблица A–H с измеренными числами против ожидаемых из таблицы выше;
   полный текстовый вывод инструмента для каждого прогона.
4. Светодиод: `led_probe.json` целиком и что записано в конфиги.
5. Что не получилось: точная команда, вывод, лог. Замеры, которые не
   сделаны, перечислить явно с причиной.
6. Наблюдения: нагрев узлов в PoE-корпусах, стабильность PTP за 10 минут,
   поведение коммутатора под 800+ Мбит/с (потери по `ethtool -S`).

Без пунктов 3 и 5 отчёт не принимается.
