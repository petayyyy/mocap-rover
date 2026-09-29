# Веб-стрим IMX219

## Установка на Raspberry Pi

```bash
sudo apt update
sudo apt install -y python3-flask python3-picamera2 python3-pil python3-numpy
```

Скопируйте `imx219_fov_stream.py` на Raspberry Pi и запустите:

```bash
python3 imx219_fov_stream.py
```

Затем откройте в браузере:

```text
http://IP_RASPBERRY_PI:8080
```

По умолчанию выбран режим `1640x1232`, 8-bit, полная матрица, целевая частота 83 FPS.

## Как читать показания FPS

Страница показывает две разные величины, и путать их нельзя:

- **FPS сенсора** — медиана `FrameDuration` из метаданных камеры. Это истинный
  такт матрицы. Он не зависит от того, успевает ли Python кодировать JPEG.
  Именно это число сравнивайте с заявленными 83.7 FPS.
- **Доставлено в поток** — сколько кадров реально прошло через Python и попало
  в MJPEG. Эта величина всегда ниже, потому что JPEG-кодирование в PIL на
  1640x1232 занимает порядка 17 мс на кадр.

Счётчик «Кадров / пропущено» показывает, сколько кадров кодировщик не успел
забрать. Пропуски здесь — норма и на частоту сенсора не влияют.

Если нужен более высокий *доставленный* FPS, увеличьте `PREVIEW_DIVISOR`
в начале скрипта (2 → 4): превью уменьшится, кодирование станет вчетверо дешевле.

## Порядок цветовых каналов

В picamera2 имя формата читается в обратном порядке байт: `"RGB888"` отдаёт
numpy-массив в порядке **BGR**, а `"BGR888"` — в порядке **RGB**. Скрипт
использует `"BGR888"`, потому что `PIL.Image.fromarray(..., "RGB")` ждёт именно
R,G,B. Если поставить «интуитивный» `"RGB888"`, красный и синий поменяются
местами.

Карточка «Проверка цветов» показывает значения центрального пикселя прямо из
numpy-массива, до кодирования в JPEG. Наведите камеру на красный объект: канал
R должен быть наибольшим. Это проверяет камеру, а не то, как браузер отрисовал
картинку.

## Угол обзора

Для оценки угла обзора введите расстояние до объекта и его ширину. Чтобы
измерить весь угол объектива, направьте камеру на стену и введите ширину всей
видимой области стены.

## Исправленные дефекты и результаты проверки

Проверено на живом железе 29.09.2026: Raspberry Pi 5 + IMX219, `pi@192.168.1.189`.

### 1. Красный и синий каналы менялись местами

Было `"format": "RGB888"`, стало `"format": "BGR888"`.

В picamera2 имя формата читается в обратном порядке байт, поэтому `"RGB888"`
отдаёт numpy-массив в порядке BGR. `PIL.Image.fromarray(..., "RGB")` принимал
его как RGB — отсюда синие лица и холодный оттенок дерева.

Проверка после фикса: средние по кадру `R=76.0 G=66.6 B=74.7`, тёплые
поверхности выглядят тёплыми: дерево коричнево-оранжевое, а не холодное.

### 2. FPS 57 вместо заявленных 83.7

Причина была не в камере. `_capture_loop` выполнял захват и JPEG-кодирование
в PIL последовательно в одном потоке; кодирование кадра 1640x1232 занимает
около 17 мс, что и давало потолок ~57 FPS. При этом счётчик частоты считал
по таймстемпам доставленных кадров, поэтому показывал скорость кодировщика
под видом скорости матрицы.

Что сделано:

- захват и кодирование разнесены по разным потокам;
- `make_array()` вызывается только когда кодировщик свободен — иначе кадр
  пропускается без копирования 6 МБ;
- частота считается по медиане `FrameDuration` из метаданных, то есть
  отражает такт сенсора независимо от пропусков;
- превью уменьшается вдвое (`PREVIEW_DIVISOR`), кодирование дешевле вчетверо.

Замер после фикса:

```text
сенсор=83.05 | доставлено=82.79 | целевой=83 | режим=1640x1232 @ 8-bit | дроп=0
```

Лог libcamera при старте подтверждает выбор нужного режима матрицы:

```text
Selected sensor format: 1640x1232-SBGGR8_1X8/RAW
```

Это SRGGB8 — режим с потолком 83.7 FPS. Если бы libcamera выбрала 10-битный
вариант, потолок был бы 41.85 FPS. Текущий выбранный режим всегда виден на
странице в поле «Режим сенсора».

## Развёртывание на Pi

```bash
scp imx219_fov_stream.py pi@192.168.1.189:~/
ssh pi@192.168.1.189 'python3 ~/imx219_fov_stream.py'
```

Быстрый замер без браузера:

```bash
curl -s http://192.168.1.189:8080/api/status | python3 -m json.tool
```

---

# Камерный узел `camera_node.py`: окна по LAN с честным временем

Рабочая программа узла на CM4/CM5. `imx219_fov_stream.py` выше остаётся
инструментом проверки FPS и цветов; в бою на узле работает `camera_node.py`
как служба systemd. Приёмник на ноутбуке —
`localization_contracts/lan_capture.py` (`LanCameraSource`).

## Что делает узел

- Сенсор всегда работает в режиме `1640x1232`, 8 бит, `YUV420`, только
  плоскость Y, `FrameDurationLimits` под 83 к/с, экспозиция и усиление
  фиксированы из конфига, `AeEnable=False`, `AwbEnable=False`. Если libcamera
  выбирает другой режим матрицы, узел не стартует (ошибка в журнале).
- Захват в своём потоке через `MappedArray` без копирования буфера; из
  отображённого кадра вырезаются только окна (копия ≈ 0.23 МБ на окно
  480×480), полный кадр (2 МБ) копируется только по запросу.
- По одному TCP-соединению узел отдаёт то, что просит ноутбук:
  `set_windows` — список окон на каждый следующий кадр; `full_frame` — один
  полный кадр; `stream_full` — каждый N-й полный кадр (запись датасета);
  `configure` — экспозиция/усиление/частота; `status` — статус немедленно.
  Все окна одного кадра вырезаны из одного и того же буфера и несут один
  `stamp_ns`.
- Без клиента захват продолжается, кадры считаются и выбрасываются; при
  подключении клиент получает `HELLO` с идентичностью узла из конфига, а
  не из IP.
- Раз в секунду `STATUS`: `sensor_fps` по медиане `FrameDuration`,
  захвачено/пропущено сенсором/сброшено из-за отставания канала, задержка
  «экспозиция → отправка» P50/P95, `ptp` (offset и состояние порта из
  `pmc`), смещение `REALTIME − BOOTTIME`, температура SoC,
  `vcgencmd get_throttled`, загрузка CPU.

## Время в заголовке кадра

`stamp_ns` в каждом кадре — **начало экспозиции первой строки** в общей с
ноутбуком шкале `CLOCK_REALTIME`, которую держит PTP. Считается так:

```text
stamp_ns = SensorTimestamp                      # CLOCK_BOOTTIME, из libcamera
         + (REALTIME − BOOTTIME)                # пара clock_gettime вокруг кадра
         + поправка за то, что означает SensorTimestamp на платформе
         + stamp_correction_ns                  # остаток, измеренный светодиодом
```

Поправка задаётся в конфиге полем `stamp_reference`:

| `stamp_reference` | Что помечает SensorTimestamp | Поправка |
|---|---|---|
| `exposure_start_first_row` | начало экспозиции первой строки | 0 |
| `readout_start_first_row` | начало считывания первой строки (по умолчанию: FS-прерывание CSI-2 на unicam/CFE) | `−exposure_ns` |
| `frame_end` | конец считывания кадра | `−(exposure_ns + H·line_time_ns)` |

Значение по умолчанию — гипотеза. Её обязана заменить измеренная
светодиодом (`led_timestamp_probe.py`, ниже). Время центра маркера в строке
`row` тракт считает как `stamp_ns + row · line_time_ns + exposure_ns / 2`.

`line_time_ns` (период строки rolling shutter) по умолчанию берётся из
регистровой модели IMX219 для этого режима: `LINE_LENGTH_PCK = 3448`
пиксельных тактов при 182.4 МГц, а в биннинге 2×2 с 8-битным выводом
строки идут вдвое быстрее — 9452 нс на строку, 1232 строки за 11.65 мс, что
и даёт потолок 83.7 к/с при 1264 строках с гашением. Поле `line_time_ns` в
конфиге переопределяет модель измеренным значением.

## Установка на CM4/CM5

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-numpy python3-simplejpeg linuxptp ethtool
# опционально, для пробы светодиода:
sudo apt install -y python3-lgpio
sudo mkdir -p /opt/mocap-rover /etc/mocap-rover
sudo rsync -a --delete pi_cam/ /opt/mocap-rover/pi_cam/
sudo cp pi_cam/node_config.example.json /etc/mocap-rover/node_config.json
sudo nano /etc/mocap-rover/node_config.json     # camera_id, exposure_us, analogue_gain
sudo cp pi_cam/camera_node.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now camera_node
journalctl -u camera_node -f
```

JPEG окон кодируется `simplejpeg` (libjpeg-turbo, идёт вместе с picamera2);
если его нет, узел берёт `cv2.imencode` (тоже libjpeg-turbo). PIL не
используется. Ручной запуск без службы:

```bash
python3 /opt/mocap-rover/pi_cam/camera_node.py --config /etc/mocap-rover/node_config.json --verbose
```

Без камеры, для проверки протокола на любой машине:

```bash
python3 pi_cam/camera_node.py --synthetic --camera-id camera_1 --port 5600
```

## PTP: общая шкала времени

Ноутбук — grandmaster, узлы — slave. Сначала узнать, есть ли аппаратные
метки времени у сетевых карт:

```bash
ethtool -T eth0        # на узле
ethtool -T enp4s0      # на ноутбуке (имя интерфейса своё)
```

Строки `hardware-transmit`/`hardware-receive` и `PTP Hardware Clock: 0` —
аппаратные метки (CM4 с BCM54210PE и CM5 их поддерживают). Если PHC нет —
программные метки (`-S`), точность падает до сотен микросекунд, и это надо
честно записать в отчёт.

Ноутбук (мастер, аппаратные метки, если есть; иначе добавить `-S`):

```bash
sudo ptp4l -i enp4s0 -m --masterOnly 1 --priority1 10
sudo phc2sys -c enp4s0 -s CLOCK_REALTIME -w -m      # PHC ← системные часы мастера
```

Узел (slave):

```bash
sudo ptp4l -i eth0 -m -s          # добавить -S, если ethtool не показал PHC
sudo phc2sys -s eth0 -c CLOCK_REALTIME -w -m
```

Для постоянной работы: `/etc/linuxptp/ptp4l.conf` c `slaveOnly 1` на узлах и
`systemctl enable --now ptp4l phc2sys`. Узел опрашивает `pmc -u -b 0 -s
/var/run/ptp4l 'GET CURRENT_DATA_SET' 'GET PORT_DATA_SET'` раз в секунду и
кладёт `offsetFromMaster` в статус и в заголовок каждого кадра
(`ptp_offset_ns`); если `pmc` недоступен, в статусе `ptp.state =
unavailable`. Цель — ≤ 100 мкс между любыми узлами, порог приёмки 1 мс;
проверяется не по `pmc`, а светодиодом (`scripts/lan_sync_check.py`).

## Протокол

Один TCP-поток на узел, узел — сервер (порт из конфига, по умолчанию 5600).
Формат описан и реализован в `pi_cam/lan_protocol.py`; этот файл общий для
узла и ноутбука и зависит только от стандартной библиотеки.

```text
MessageHeader  14 байт  <HBBHII  magic 0x4D43, version 1, type, header_len, json_len, data_len
fixed header   header_len байт (только для кадров, 92 байта, один struct.unpack)
JSON           json_len байт (команды, статус, ack; у кадров пусто)
data           data_len байт (Y8 построчно или JPEG)
```

Типы: `FRAME=1` (узел → ноутбук), `STATUS=2`, `COMMAND=3` (ноутбук → узел),
`ACK=4`, `HELLO=5`. Фиксированный заголовок кадра (little-endian):

| Поле | Тип | Смысл |
|---|---|---|
| `camera_id` | 16s | идентичность узла из конфига |
| `frame_seq` | u64 | счётчик кадров сенсора |
| `stamp_ns` | i64 | общая шкала, начало экспозиции первой строки |
| `exposure_ns`, `line_time_ns`, `frame_duration_ns` | u32 | из метаданных и модели |
| `row0`, `col0`, `width`, `height` | u16 | окно в полном кадре |
| `format` | u8 | 0 = y8, 1 = jpeg |
| `window_index`, `window_count` | u8 | окна одного кадра |
| `sensor_width`, `sensor_height` | u16 | 1640, 1232 |
| `request_id` | u32 | какой команде отвечает |
| `node_send_ns` | i64 | REALTIME узла перед отправкой |
| `sensor_stamp_ns` | i64 | сырой SensorTimestamp (BOOTTIME) |
| `clock_offset_ns` | i64 | REALTIME − BOOTTIME узла |
| `ptp_offset_ns` | i64 | `offsetFromMaster` из pmc, INT64_MIN = неизвестно |

Команды (JSON, поле `cmd`; ответ `ACK` с тем же `token` и `request_id`):

```json
{"cmd": "set_windows", "windows": [{"row0": 100, "col0": 200, "w": 480, "h": 480, "format": "y8"}]}
{"cmd": "full_frame", "format": "y8"}
{"cmd": "stream_full", "divisor": 8, "format": "y8"}
{"cmd": "configure", "exposure_us": 800, "gain": 4.0, "fps": 83.0}
{"cmd": "status"}
```

Окна режутся до границ сенсора и выравниваются на чётные `row0`/`col0`
(ячейка Байера биннинга). Пустой список окон выключает окна.

## Приёмник на ноутбуке

```python
from localization_contracts.lan_capture import LanCameraSource
src = LanCameraSource(["192.168.1.101:5600", "192.168.1.102:5600"])
src.wait_connected(5.0)                       # ['camera_1', 'camera_2'] из HELLO
src.request_windows("camera_1", [(100, 200, 480, 480), (600, 900, 480, 480, "jpeg")])
group = src.take("camera_1", timeout=0.2)     # список LanFrame одного кадра, один stamp_ns
for camera_id, array, stamp_ns, row0, col0, line_time_ns, exposure_ns, receive_ns in group:
    ...
src.request_full("camera_1")                  # watchdog / захват
```

Семантика доставки как у `capture.LatestFrames`: на камеру хранится
последний полный кадр (все его окна), отставший потребитель теряет старый,
потери считаются (`src.stats()["dropped"]`). При обрыве приёмник
переподключается сам и заново отправляет последние `set_windows` /
`stream_full`.

## Инструменты и порядок измерений

1. **Режим и частота.** `journalctl -u camera_node` показывает выбранный
   режим (`Selected sensor format: 1640x1232-SBGGR8_1X8`), статус —
   `sensor_fps` и `frames_missed`.
2. **Что такое SensorTimestamp и `line_time_ns`** — на узле, служба
   остановлена, светодиод на GPIO 17 светит в объектив:

   ```bash
   sudo systemctl stop camera_node
   python3 /opt/mocap-rover/pi_cam/led_timestamp_probe.py --config /etc/mocap-rover/node_config.json \
       --gpio 17 --flashes 50 --pulse-us 100 --columns 780,860 --output /tmp/led_probe.json
   ```

   Вспышка длиной 100 мкс засвечивает полосу строк; последняя засвеченная
   строка против `flash − SensorTimestamp` ложится на прямую: наклон —
   `line_time_ns`, пересечение говорит, что помечает SensorTimestamp.
   Результат (`recommended_config`) переписать в `node_config.json`:
   `stamp_reference`, `stamp_correction_ns`, `line_time_ns`. Расхождение
   `line_time_error_percent` с моделью должно быть ≤ 5 %. Сухой прогон без
   железа: `--synthetic`.
3. **Задержки и дропы в штатном режиме** (2 окна 480×480 + полный кадр раз
   в 2 с), 10 минут, вывод целиком в отчёт:

   ```bash
   .venv/bin/python scripts/lan_camera_timing.py --nodes 192.168.1.101:5600 192.168.1.102:5600 \
       --seconds 600 --json artifacts/lan_timing.json
   ```
4. **Синхронизация двух узлов.** Один светодиод виден обеим камерам, мигает
   от третьего Pi (или любого узла с остановленной службой):
   `led_timestamp_probe.py --blink-only --gpio 17 --period-s 0.4`. На ноутбуке:

   ```bash
   .venv/bin/python scripts/lan_sync_check.py --nodes 192.168.1.101:5600 192.168.1.102:5600 \
       --window 376,580,480,480 --columns 200,280 --flashes 50 --json artifacts/lan_sync.json
   ```

   Разность `stamp_ns + row·line_time_ns + exposure/2` двух узлов для одной
   вспышки — измеренная ошибка синхронизации; печатаются P50/P95/max по
   ≥ 50 вспышкам и проверка `line_time_ns` по высоте полосы.
5. **Запись датасета** в формате `scripts/record_dataset.py`:

   ```bash
   .venv/bin/python scripts/record_lan_dataset.py --nodes 192.168.1.101:5600 ... \
       --config config/mocap_arena_imx219/runtime_cameras.json --output artifacts/dataset_lan_01 \
       --seconds 60 --divisor 8
   ```

   Узлы переводятся в `stream_full` с делителем (83/8 ≈ 10 полных кадров/с
   на камеру, 162 Мбит/с на камеру в Y8); реально прошедшая частота и объём
   пишутся в `meta.json` (`achieved`). `truth.jsonl` пустой. Читается
   `scripts/replay_dataset.py` без правок.

Тесты всего этого без железа: `.venv/bin/python -m pytest tests/test_lan_capture.py -q`.

## Проверено на живом железе (CM5 Lite, 30.09.2026)

Полный отчёт с числами — `docs/cm4_camera_node_report.md`. Здесь только то,
что меняет порядок установки.

**Режим и частота.** libcamera выбрала `1640x1232-SBGGR8_1X8/RAW`,
`sensor_fps` 83.05, за 10 минут в режиме «2 окна 480×480 + полный кадр раз в
2 с» сенсор не пропустил ни одного кадра, узел не сбросил ни одного,
температура максимум 50 °C, троттлинга нет, CPU узла 9 %.

**`SensorTimestamp` — начало считывания первой строки.** Проверено
развёрткой по экспозиции: при её изменении с 0.5 до 4 мс задержка
«метка → выдача кадра» постоянна (14.18 мс, наклон 0.000), то есть метка
экспозицию не включает. Поэтому `stamp_reference` по умолчанию оставлен
`readout_start_first_row`. Светодиодный тест всё равно нужен: он даёт
абсолютную привязку и меряет `line_time_ns`.

**Задержка.** От начала экспозиции до появления кадра в Python проходит
16.0 мс, из которых 11.64 мс — считывание матрицы, а ≈ 3.6 мс — ISP и
передача буфера. Узел добавляет 2.3 мс. Бюджет 20 мс это почти исчерпывает,
поэтому на гигабите берите JPEG-окна, а не y8. На CM4 стоит проверить захват
мимо ISP (там сырой `SBGGR8` не сжат, в отличие от `PISP_COMP1` на CM5) —
это может снять около 3 мс.

**PTP: обязательный `tx_timestamp_timeout`.** Драйвер eth0 на CM5 не
поддерживает программные метки вообще (`ptp4l -S` отказывается стартовать), а
с аппаратными и таймаутом по умолчанию уходит в `FAULTY`
(`timed out while polling for tx timestamp`). Рабочий запуск:

```bash
sudo ptp4l -i eth0 -m --tx_timestamp_timeout 50          # добавить -s на slave-узлах
sudo phc2sys -s eth0 -c CLOCK_REALTIME -w -m
```

**Мастером делайте узел, а не ноутбук.** У CM4 и CM5 есть PHC
(`PTP Hardware Clock: 0`, режимы до `onestep-sync`), а у USB-адаптера
ноутбука аппаратных меток нет вовсе. Ошибка между узлами по ТЗ важнее
абсолютной шкалы ноутбука, поэтому grandmaster — один из узлов. Для
ноутбука нужна карта с PHC или свитч с PTP.

**Доступ к `pmc`.** Он лежит в `/usr/sbin` (нет в PATH службы) и не может
привязать свой сокет от имени `pi`. Конфиг по умолчанию вызывает его через
`sudo -n`; добавьте одну строку и уберите шум в журнале:

```bash
echo 'pi ALL=(root) NOPASSWD: /usr/sbin/pmc' | sudo tee /etc/sudoers.d/mocap-pmc
echo 'Defaults!/usr/sbin/pmc !syslog' | sudo tee -a /etc/sudoers.d/mocap-pmc
sudo chmod 0440 /etc/sudoers.d/mocap-pmc
```

Без этого поле `ptp.offset_ns` останется пустым, а метки времени кадров
не пострадают: они идут по `CLOCK_REALTIME`, который синхронизирует
`phc2sys`. Опрос по умолчанию раз в 5 с.

**Освещение.** При штатной экспозиции 0.8 мс и усилении 4 кадр в обычной
комнате почти чёрный (средняя яркость 18 из 255). Арену нужно освещать
примерно в шесть раз ярче; поднимать экспозицию нельзя из-за смаза на
11 м/с.

**Пропускная способность на стенде.** USB-адаптер даёт 100 Мбит/с: два окна
480×480 в y8 (306 Мбит/с) на полной частоте не проходят, JPEG-окна проходят
свободно (11.5 Мбит/с на тёмной сцене). Для записи датасета полными кадрами
прошло 67 Мбит/с на камеру при делителе 20 (4.15 к/с).
