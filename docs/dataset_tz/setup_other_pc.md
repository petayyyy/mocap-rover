# Методичка: проверка и запуск на другом компьютере (Intel iGPU)

Проверено на ноутбуке с RTX 3070. На машине с Intel Core Ultra 7 155H и
встроенной графикой работает всё, кроме CUDA-алгоритмов (CUDA MOG2, любые
TensorRT/torch-на-GPU шаги). Gazebo рендерит через Mesa на iGPU: кадры те же,
только медленнее, поэтому все замеры задержки с такой машины сравнивать с
ноутбуком нельзя, а точность и логику — можно.

## 1. Система

Ubuntu 24.04. Установить ROS 2 Jazzy и Gazebo Harmonic из vendor-пакетов ROS
(так же, как на основной машине: `gz` берётся из `/opt/ros/jazzy/opt/gz_tools_vendor`):

```bash
sudo apt update
sudo apt install -y ros-jazzy-ros-base ros-jazzy-ros-gz ffmpeg git python3-venv python3-opencv python3-numpy python3-scipy python3-yaml mesa-utils
```

Проверка Gazebo и iGPU:

```bash
source /opt/ros/jazzy/setup.bash && gz sim --version
```

```bash
glxinfo -B | grep -E "OpenGL renderer|OpenGL version"
```

Ожидается `Gazebo Sim, version 8.x` и рендерер `Mesa Intel(R) Arc(TM) Graphics`.
Если `gz` не находится, значит не выполнен `source /opt/ros/jazzy/setup.bash`.

## 2. Репозиторий и окружение

```bash
git clone <URL репозитория> ~/Drone/mocap-rover && cd ~/Drone/mocap-rover
```

```bash
/usr/bin/python3 -m venv --system-site-packages .venv && .venv/bin/python -m pip install -r requirements-simulation.txt
```

`requirements-simulation.txt` тянет torch и ultralytics. Они нужны только
для старого YOLO-пути и на этой машине не используются; если установка
долгая, их можно пропустить:

```bash
.venv/bin/python -m pip install numpy==1.26.4 opencv-python==4.10.0.84 threadpoolctl==3.6.0 matplotlib==3.9.2
```

Проверка привязок Gazebo из системного Python (они не ставятся через pip):

```bash
/usr/bin/python3 -c "import gz.transport13, gz.msgs10, cv2; print('ok', cv2.__version__)"
```

## 3. Автономные проверки, без Gazebo

```bash
source /opt/ros/jazzy/setup.bash && .venv/bin/python -m unittest discover -s tests -q
```

Ожидается `Ran 255 tests ... OK`. Именно `.venv/bin/python` и именно с
`source`: системный OpenCV 4.6 не имеет `cv2.aruco.generateImageMarker`
(42 ошибки), а один тест вызывает `gz sdf`, которого без ROS-окружения нет
в PATH. Затем пересборка мира и проверка SDF:

```bash
/usr/bin/python3 scripts/generate_world.py --profile imx219_160 --layout final --lidar airy --ideal-cameras --world-name mocap_arena_imx219 --output-dir /tmp/regen
```

```bash
source /opt/ros/jazzy/setup.bash && gz sdf -k /tmp/regen/worlds/mocap_arena_imx219.sdf
```

Ожидается `Valid.` и в выводе генератора `Z=0.3654: 28 uncovered samples`.
Сгенерированный файл в `/tmp/regen` должен совпасть с
`worlds/mocap_arena_imx219.sdf` из репозитория (`diff` пустой; в файле
абсолютные пути к мешам лидара, поэтому различие только в них допустимо).

## 4. Датасеты

Датасеты не в git (4.7 ГБ + 0.3 ГБ). Перенести с основной машины:

```bash
rsync -av --progress popka@<IP ноутбука>:~/Drone/mocap-rover/artifacts/dataset_imx219_01 popka@<IP ноутбука>:~/Drone/mocap-rover/artifacts/dataset_imx219_01_background artifacts/
```

Независимая проверка датасета (декодирование, детектор маркера, PnP, лидар
против эталона):

```bash
/usr/bin/python3 scripts/check_dataset.py artifacts/dataset_imx219_01 --frames 40 --scans 4
```

Ожидается по камерам `xy_error_p50_m` около 0.02 и `yaw_error_p95_deg` < 2
там, где есть принятые наблюдения, и `check_report.json` в каталоге датасета.
Время `detect_ms_full_frame_p50` на этом процессоре будет больше, чем 27–36
мс на ноутбуке; это ожидаемо.

## 5. Gazebo на iGPU: запись короткого датасета

Отдельный раздел транспорта, чтобы не пересекаться с другими симуляциями на
той же машине:

```bash
source /opt/ros/jazzy/setup.bash && export GZ_PARTITION=imx219test GZ_SIM_RESOURCE_PATH="$PWD/models:$PWD" && gz sim -s -r --headless-rendering worlds/mocap_arena_imx219.sdf
```

Предупреждения `libEGL warning` при старте нормальны. Во втором терминале
(тот же `GZ_PARTITION`) проверить темп камер:

```bash
source /opt/ros/jazzy/setup.bash && GZ_PARTITION=imx219test gz topic -l | grep -E "cameras|robosense|clock"
```

Затем езда и запись 5 сим-секунд:

```bash
source /opt/ros/jazzy/setup.bash && export GZ_PARTITION=imx219test && /usr/bin/python3 scripts/drive_random.py --seconds 600 --speed 5 --seed 7
```

```bash
source /opt/ros/jazzy/setup.bash && export GZ_PARTITION=imx219test && /usr/bin/python3 scripts/record_dataset.py --config config/mocap_arena_imx219/runtime_cameras.json --output artifacts/dataset_igpu_smoke --seconds 5
```

Рекордер печатает `rtf`. На RTX 3070 он 0.11–0.12; на iGPU ожидается
0.02–0.05, то есть 5 сим-секунд займут 2–4 минуты. В `meta.json` должно быть
по ~415 кадров на камеру, ~50 сканов и `image_size_errors: 0`. Далее
`check_dataset.py` на этом каталоге, как в п. 4. Остановить драйвер и сервер
через Ctrl+C; убедиться, что ничего не осталось:

```bash
pgrep -af "gz sim|drive_random|record_dataset"
```

## 6. Что на этой машине не проверять

- CUDA-варианты фона (`cv2.cuda`), любые запуски с `device=0` и torch на GPU.
- Абсолютные значения задержки из промпта 04 (`timing.json`): критерии в
  `docs/dataset_tz/README.md` заданы для ноутбука с 3070. На iGPU-машине
  принимаются только точность, доля valid, подмены идентичности и тесты.

## 7. Работа исполнителя по ТЗ

Исполнителю отдаётся `docs/dataset_tz/README.md` и по одному промпту
`01..04`. Его реплей (`scripts/replay_dataset.py`, появится по промпту 01)
работает без Gazebo и без ROS, поэтому на этой машине его можно гонять
целиком, включая оценку:

```bash
/usr/bin/python3 scripts/evaluate_recording.py --runtime <каталог реплея> --truth artifacts/dataset_imx219_01/truth.jsonl --output /tmp/eval.json
```
