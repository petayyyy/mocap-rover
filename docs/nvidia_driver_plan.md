# План NVIDIA-драйвера для mocap-rover

Дата аудита: 2026-09-21. Аудит выполнен только чтением; пакеты, репозитории,
Secure Boot, графическая сессия и загрузка не изменялись.

## Итог

Основной вариант: **Ubuntu `nvidia-driver-580-open`, ветка 580, пакет Ubuntu**.
На машине эта ветка уже установлена как `nvidia-driver-580-open`
`580.178.04-0ubuntu0.24.04.1`, а модуль для текущего ядра собран как
предсобранный Canonical-модуль `linux-modules-nvidia-580-open-7.0.0-31-generic`.
Наиболее вероятно, что после пакетного обновления модуль 595 остался загруженным
до перезагрузки; повторная переустановка не нужна до проверки результата reboot.

Запасной вариант: **`nvidia-driver-595-open` из Ubuntu noble-updates**, только если
для конкретной версии PyTorch/TensorRT потребуется CUDA 13.x или ветка 580 будет
неприемлема после чистой перезагрузки. Это не первый выбор для стабильного Gazebo:
595 сейчас присутствует лишь остаточными config-файлами/firmware, а рабочая
библиотечная часть системы 580. Смешивать 580 и 595 нельзя.

## Фактический аудит

- ОС: Ubuntu 24.04.5 LTS, x86_64; ядро `7.0.0-31-generic` (HWE/custom HWE-мета-
  пакет). Заголовки именно этого ядра перед установкой надо проверить.
- GPU: ASUS NVIDIA GeForce RTX 3070 LHR, GA104, PCI ID `10de:2488`, Ampere,
  compute capability 8.6, **8 GB GDDR6**. Это обычная дискретная карта, не
  hybrid/Optimus: в PCI присутствует один NVIDIA VGA-адаптер; AMD-чипсет — не
  графический адаптер. Официальные характеристики RTX 3070: [NVIDIA](https://www.nvidia.com/en-me/geforce/graphics-cards/30-series/rtx-3070-3070ti/).
- Загруженный модуль: `NVIDIA UNIX Open Kernel Module 595.91.07`; `modinfo`
  для установленного файла 580 показывает `580.178.04`, значит kernel/userspace
  mismatch подтверждён.
- Пакеты 580 установлены: `nvidia-driver-580-open`, `nvidia-kernel-source-580-open`,
  `linux-modules-nvidia-580-open-7.0.0-31-generic`, GL/EGL, compute, utils.
  Пакеты 595 имеют статус `deinstall ok config-files` или остаточный firmware;
  это не означает, что полноценный 595 установлен.
- DKMS: стороннего NVIDIA DKMS нет; `dkms status` показывает только AmneziaWG.
  Модуль NVIDIA — предсобранный и подписанный Canonical (`signer: Canonical Ltd.`).
- Secure Boot: disabled. Сессия: GNOME/X11 (`DISPLAY=:1`), Wayland не используется.
- `nvidia-smi`: `Failed to initialize NVML: Driver/library version mismatch`,
  NVML library `580.178`; это ожидаемый симптом текущего рассогласования.
- OpenGL: GLX падает; EGL не инициализируется на X11/GBM и выбирает Mesa
  `swrast`, renderer `llvmpipe`. GPU-ускоренный Gazebo сейчас не подтверждён.
- Переменные `CUDA`, `NVIDIA`, `LD_LIBRARY_PATH`, `__GLX*` в текущем окружении не
  выставлены. `ldconfig` показывает NVIDIA GL/CUDA библиотеки 580.
- `command -v nvcc` не найден: системный CUDA Toolkit не установлен/не виден.
  В системном Python не установлены `torch`, `tensorrt`, `onnxruntime`,
  `ultralytics`; найденных venv/conda-окружений в проекте не обнаружено.
  Поэтому CUDA из PyTorch и TensorRT runtime пока отсутствуют для проверки.
- Источники: Ubuntu noble/noble-updates/security (restricted/multiverse), ROS 2,
  OSRF Gazebo и NVIDIA Container Toolkit. Отдельного NVIDIA CUDA apt-репозитория
  нет. В источниках не обнаружено временных библиотек 595; workaround проекта
  всё ещё существует в `scripts/run.sh` и обращается к
  `~/.cache/mocap-rover/nvidia-595.91.07`.

## Почему 580-open

RTX 3070 — Ampere, то есть open kernel modules совместимы. NVIDIA указывает, что
с 560 open flavor является рекомендуемым, а Turing и новее его поддерживают:
[NVIDIA kernel modules](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/610/kernel-modules.html).
Ubuntu-пакеты предпочтительнее `.run`-инсталлятора: они обслуживают ядро и GL/EGL
через apt. NVIDIA также рекомендует distribution-specific пакеты и требует
заголовки текущего ядра: [installation guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html).

580 покрывает CUDA 12.x и является минимальной веткой для CUDA 13.x. Согласно
официальной CUDA compatibility matrix, CUDA 12.x требует driver >=525, CUDA 13.x
— >=580: [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
Следовательно, 580 подходит для PyTorch CUDA 12.6/12.8 и TensorRT с CUDA 12.x,
а также оставляет путь к CUDA 13.x без перехода на экспериментально более новую
ветку. Официальная страница PyTorch предлагает CUDA 11.8, 12.6 и 12.8 wheels:
[PyTorch install](https://pytorch.org/get-started/locally/). TensorRT требует
Linux driver r535+, а для CUDA 13.x — r580+: [TensorRT prerequisites](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/prerequisites.html).

Для данного приоритета я бы начинал с PyTorch wheel `cu126` или `cu128`, а не с
системного Toolkit: wheel содержит свой CUDA runtime, драйвер предоставляет kernel
driver/API. `nvidia-smi`'s “CUDA Version” после исправления будет верхней
поддерживаемой версией драйвера, а не доказательством установленного Toolkit.

## План, только после подтверждения

### 1. Повторный снимок и симуляция

```bash
uname -r
apt-cache policy nvidia-driver-580-open nvidia-driver-595-open
apt-mark showhold
dpkg -l | grep -E 'nvidia|cuda|tensorrt' | sort
sudo apt-get update
apt-get -s install --reinstall nvidia-driver-580-open \
  linux-modules-nvidia-580-open-$(uname -r)
```

`apt-get -s` ничего не меняет. Перед применением надо убедиться, что simulation
не предлагает удалить ROS/Gazebo и не выбирает 595. Если пакет модуля для текущего
ядра отсутствует, сначала установить matching headers и модуль через Ubuntu HWE.

### 2. Согласовать уже выбранную ветку

```bash
sudo apt-get install --reinstall nvidia-driver-580-open \
  linux-modules-nvidia-580-open-$(uname -r)
sudo update-initramfs -u -k $(uname -r)
```

Ожидаемое изменение: переустановка 580 userspace/GL/EGL и initramfs; ROS, Gazebo,
CUDA Toolkit и Python-пакеты не должны удаляться. Не выполнять `rmmod`, не
останавливать display manager и не менять Secure Boot.

### 3. Перезагрузка обязательна для проверки

Старый модуль 595 нельзя заменить безопасно внутри работающей графической сессии.
После согласования пакетов:

```bash
sudo reboot
```

После входа проверить согласованность:

```bash
cat /proc/driver/nvidia/version
nvidia-smi
modinfo nvidia | grep -E 'version|filename|signer|vermagic'
glxinfo -B
eglinfo | grep -E 'EGL vendor|EGL version|OpenGL renderer'
ldconfig -p | grep -E 'libcuda|libnvidia-gl|libEGL'
```

Ожидается одна версия 580, успешный NVML, renderer NVIDIA (не llvmpipe), рабочие
X11 GLX и EGL. Только после этого можно удалить остаточные 595 config-пакеты:

```bash
dpkg-query -W -f='${binary:Package}\t${db:Status-Status}\n' \
  'nvidia*595*' 'libnvidia*595*' 2>/dev/null
# затем, после проверки списка:
sudo apt-get purge <только перечисленные пакеты со статусом config-files>
sudo apt-get autoremove --purge
```

### 4. Workaround Gazebo

До успешной проверки после reboot оставить `scripts/run.sh` без изменений. После
успешного `nvidia-smi`, GLX/EGL и теста Gazebo удалить/отключить ветку, которая
добавляет `~/.cache/mocap-rover/nvidia-595.91.07` в `LD_LIBRARY_PATH`; запускать
Gazebo с системными библиотеками. Не удалять cache до этой проверки. Минимальная
проверка:

```bash
env -u LD_LIBRARY_PATH gz sim -r worlds/mocap_arena.sdf
```

Затем подтвердить GUI, шесть камер, GPU renderer и отсутствие mismatch. В текущем
репозитории полноценная six-camera acceptance ещё не считается доказанной только
по наличию пакетов; нужен фактический smoke test.

## Восстановление

Если после reboot нет графического входа, перейти в TTY `Ctrl+Alt+F3`, войти и
собрать диагностику:

```bash
journalctl -b -p err..alert
systemctl status gdm3 --no-pager
nvidia-smi
```

Не удалять драйвер вслепую. Сначала выбрать предыдущий kernel в GRUB → Advanced
options. Если потребуется откат, он должен быть отдельным подтверждённым шагом;
рабочий вариант — снова согласовать весь стек 580 и обновить initramfs, а не
подмешивать библиотеки 595.

## Критерии готовности после установки

1. `/proc/driver/nvidia/version`, `modinfo`, `nvidia-smi` и все `libnvidia*`
   имеют одну ветку/версию.
2. `nvidia-smi` видит RTX 3070 и 8 GB VRAM; `glxinfo`/`eglinfo` показывают NVIDIA.
3. В выбранном venv `torch.cuda.is_available()` == true, `torch.version.cuda`
   записан отдельно (это runtime wheel), а `nvcc --version` проверен отдельно
   (это Toolkit).
4. TensorRT проверен только если он реально установлен, с его CUDA/driver
   требованиями; engine не переносится между несовместимыми версиями без rebuild.
5. Gazebo Harmonic GUI стартует без временного `LD_LIBRARY_PATH`, шесть камер
   отображаются, а тестовая камера/рендер действительно использует GPU.

До этих пяти проверок нельзя объявлять драйвер исправленным.
