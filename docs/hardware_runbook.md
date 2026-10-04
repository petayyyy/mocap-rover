# Запуск системы на железе: установка и запуск

> **Основной сценарий с 04.10.2026 — C++-узел, один поток H.264 на камеру.**
> Установка узла: `sudo scripts/hardware/setup_cm4_node.sh N --cpp`; приём и
> проверка: `run_hardware.sh rx | rx-check | rx-record | wall | configure`;
> задание на стенд — [field_test_cpp_h264.md](field_test_cpp_h264.md),
> сравнение сценариев — [camera_scenarios.md](camera_scenarios.md). Ниже —
> прежний Python-путь (окна), он остаётся для калибровки и как запасной.

Стенд: 6 × Raspberry Pi CM4 (Waveshare CM4-NANO-B) + 6 × IMX219-160,
коммутатор 1 Гбит с PoE, ноутбук Ubuntu 24.04 с NVIDIA (проверено на RTX 3070, 8 GB),
лидар RoboSense Airy. Скрипты лежат в `scripts/hardware/`:

| Скрипт | Где запускать | Что делает |
|---|---|---|
| `setup_laptop.sh` | ноутбук | пакеты, драйвер NVIDIA, `.venv` (OpenCV, torch CUDA, SAM2), веса SAM2, адрес в сети камер |
| `setup_cm4_node.sh N` | каждый CM4 | пакеты, оверлей камеры, адрес `192.168.10.10N`, узел `camera_node`, PTP |
| `run_hardware.sh` | ноутбук | проверка стенда, замер канала, запись, локализация по записи, бенчмарк SAM2 |

## 0. Что работает на железе сейчас, а что нет

Честно, по состоянию кода на этот коммит:

- **Работает:** узлы камер (`pi_cam/camera_node.py`), приём по сети
  (`localization_contracts/lan_capture.py`), живой замер канала
  (`lan_camera_timing.py`), синхронизация PTP, запись датасета с узлов
  (`record_lan_dataset.py`), локализация `tag_rover` по маркеру на такой
  записи (`replay_dataset.py`), SAM2 на GPU.
- **Нет живого тракта «узлы → алгоритмы → выход в реальном времени».**
  `scripts/run_localization.py` читает только Gazebo (`gz.transport`).
  На железе алгоритмы сейчас запускаются **по записи**: записать → прогнать.
- **Лидара Airy в этом репозитории на железе нет.** Драйвер и разбор облака —
  в отдельном проекте `br_lidar/airy_py`; `record_lan_dataset.py` лидар не пишет.
  Поэтому на железе прогон идёт с `--no-lidar`.
- **Соперник не стартует на живой записи.** Его трек запускается от
  прямоугольника оператора, который в реплее берётся из эталона
  (`operator_box_from_truth`); у живой записи эталона нет. Нужен ввод
  оператора (клик в UI), его пока нет.
- **Калибровка.** Делается станцией калибровки: внутренняя (K, D) —
  [calibration_intrinsics.md](calibration_intrinsics.md), взаимное положение
  шести камер — [calibration_extrinsics.md](calibration_extrinsics.md). Результат —
  `runtime_cameras.json`, его передать в запись: `CAM_CONFIG=<путь> run_hardware.sh record …`.
  Без неё используется `config/mocap_arena_imx219/runtime_cameras.json` из
  симуляции, и координаты будут неточными.

## 1. Сеть

```
ноутбук 192.168.10.1/24 ── коммутатор 1 Гбит PoE ── cam-1 … cam-6: 192.168.10.101 … 106, порт 5600
```

- Адаптер ноутбука только гигабитный (USB 3 → 1 GbE, RTL8153/8156):
  `ethtool <iface> | grep Speed` → `1000Mb/s`. Шесть камер в 100 Мбит не влезают.
- Мастер PTP — `camera_1`, не ноутбук.

## 2. Ноутбук

```bash
git clone <URL> ~/mocap-rover && cd ~/mocap-rover
```

Если драйвера NVIDIA ещё нет (`nvidia-smi` не работает) — поставить и перезагрузиться:

```bash
scripts/hardware/setup_laptop.sh --driver
```

```bash
sudo reboot
```

Затем всё остальное, `<iface>` — имя гигабитного порта к коммутатору (`ip -br link`):

```bash
scripts/hardware/setup_laptop.sh --iface <iface>
```

Что ставится:

- apt: `git python3-venv ffmpeg ethtool network-manager rsync curl`;
- драйвер: `nvidia-driver-580-open` (пакет Ubuntu, ветка 580, CUDA 13.0 в
  `nvidia-smi`; ветки не смешивать, см. `docs/nvidia_driver_plan.md`);
- `.venv` (`--system-site-packages`): `numpy==1.26.4`, `opencv-python==4.10.0.84`,
  `simplejpeg`, `scipy`, `matplotlib`, `pytest`;
- `torch==2.6.0` / `torchvision==0.21.0` с колёсами `cu124`, `sam2` из
  `github.com/facebookresearch/sam2`;
- веса `sam2.1_hiera_tiny.pt` и `sam2.1_hiera_small.pt` в `models/sam2/` (не в git).

Без GPU всё, кроме SAM2, работает на CPU: `--no-sam2`.

Системный `/usr/bin/python3` для алгоритмов не годится: OpenCV 4.6 из Ubuntu не
умеет `aruco.generateImageMarker`. Всегда `.venv/bin/python`.

## 3. Узлы камер (каждый из шести)

1. **Образ:** Raspberry Pi OS **Lite 64-bit, Bookworm**. В Raspberry Pi Imager:
   hostname `cam-N`, пользователь `pi`, SSH включён. CM4 Lite — microSD;
   CM4 с eMMC — через `rpiboot` и USB носителя.
2. Шлейф камеры в `CAM0`, подключать только при выключенном питании.
3. На Pi:

```bash
git clone <URL> ~/mocap-rover && cd ~/mocap-rover
```

```bash
sudo scripts/hardware/setup_cm4_node.sh N
```

```bash
sudo reboot
```

`N` — номер камеры 1…6. Для CM5 добавить `--cm5`. Скрипт:

- ставит `python3-picamera2 python3-numpy python3-simplejpeg linuxptp ethtool python3-lgpio python3-opencv`;
- пишет в `/boot/firmware/config.txt` `camera_auto_detect=0` и `dtoverlay=imx219,cam0`
  (на CM4-NANO-B камера сама не находится);
- задаёт адрес `192.168.10.10N/24` через NetworkManager;
- копирует `pi_cam/` в `/opt/mocap-rover/pi_cam`, создаёт
  `/etc/mocap-rover/node_config.json` (`camera_id`, `"stream": "raw"` на CM4 —
  ISP CM4 отдаёт только каждый второй кадр, `buffer_count 4`, `exposure_us 800`,
  `analogue_gain 4.0`, `send_queue 3`); существующий конфиг не трогает;
- включает службы `camera_node`, `ptp4l` (`tx_timestamp_timeout 50`, на
  `camera_1` `masterOnly`, на остальных `slaveOnly`) и `phc2sys`;
- даёт `pi` право `sudo pmc` без пароля (узел опрашивает PTP).

Проверка после перезагрузки:

```bash
rpicam-hello --list-cameras
```

```bash
journalctl -u camera_node -n 30
```

```bash
sudo pmc -u -b 0 'GET PORT_DATA_SET'
```

Ждём: `imx219` с режимом `1640x1232`; в журнале
`Selected sensor format: 1640x1232-SBGGR8_1X8`; `portState MASTER` на camera_1 и
`SLAVE` на остальных. Если `ptp4l` уходит в `FAULTY` — не применился
`tx_timestamp_timeout`.

**Свет.** При 0.8 мс и усилении 4 в обычной комнате кадр почти чёрный; для
маркера нужен свет арены или временно `exposure_us` 2000–4000 в
`/etc/mocap-rover/node_config.json`, затем `sudo systemctl restart camera_node`.

**Привязка времени (один раз, нужен светодиод на GPIO 17 + 220 Ом):**
`pi_cam/led_timestamp_probe.py`, порядок — в `docs/field_test_six_cm4.md`, раздел 1.
`recommended_config` переписать во все шесть конфигов.

Обновить код узла после `git pull`: повторить `setup_cm4_node.sh N` (конфиг
сохранится) и `sudo systemctl restart camera_node`.

## 4. Запуск с ноутбука

Все команды — из корня репозитория. Узлы по умолчанию `192.168.10.101…106:5600`,
другой набор: `NODES="192.168.10.101:5600 192.168.10.102:5600" scripts/hardware/run_hardware.sh …`.

**Стенд жив:** пинг узлов, скорость порта, GPU, место на диске:

```bash
IFACE=<iface> scripts/hardware/run_hardware.sh status
```

**Канал вживую** (окно маркера 320 raw + малый поток 640×480@30, 60 с):

```bash
scripts/hardware/run_hardware.sh check 60
```

Ждём `sensor fps` ≈ 83, `node missed` ≈ 0, `throttled none`, температура < 70 °C.

**Пустая арена** (фон для соперника; роверов на арене нет):

```bash
scripts/hardware/run_hardware.sh background arena_bg 20
```

**Запись с роверами:**

```bash
scripts/hardware/run_hardware.sh record run_01 60
```

Пишется каждый 12-й полный кадр (~7 к/с на камеру, `DIVISOR=` меняет),
около 5 GB на минуту шести камер — скрипт проверяет свободное место и не
перезаписывает существующие каталоги. В `meta.json` → `achieved` — сколько
реально прошло через канал.

**Алгоритмы (локализация по записи):**

```bash
scripts/hardware/run_hardware.sh track run_01
```

Результат в `artifacts/track_run_01/`: `odometry.jsonl` (позы роверов), `observations.jsonl`,
`timing.json`. Сейчас это `tag_rover` по маркеру, без лидара. С фоном
(`track run_01 arena_bg`) включается модель фона для соперника, но без
прямоугольника оператора он не стартует (раздел 0). Любые флаги
`replay_dataset.py` дописываются в конец, например
`--sam2-mode backup` (рекомендованный режим из
`docs/dataset_tz/reports/08_sam2_report.md`: tiny, 512 px, bf16, 2 камеры × 15 Гц).

**Скорость SAM2 на этой GPU:**

```bash
scripts/hardware/run_hardware.sh sam2-bench --dataset artifacts/dataset_run_01
```

На RTX 3070: 2 камеры × 15 Гц — P95 15.3 мс, загрузка 49 %, ~450 MB VRAM.

## 5. Замеры стенда и что делать дальше

- Полная программа замеров канала (A–H, PTP, светодиод): `docs/field_test_six_cm4.md`,
  малый поток: `docs/field_test_small_stream.md`.
- Чтобы получить настоящий рантайм на железе, не хватает:
  1. живого источника кадров для `camera_worker` из `lan_capture` вместо датасета
     (контракт кадра уже общий — `localization_contracts/frame_source.py`);
  2. приёма лидара Airy (из `br_lidar/airy_py`) в формате `lidar_pipeline`;
  3. ввода прямоугольника оператора для старта соперника;
  4. проверки калибровки станцией на реальном стенде (код готов, на железе не запускался).
