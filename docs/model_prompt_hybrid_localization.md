# Готовый запрос для модели

Ты выступаешь как ведущий инженер по computer vision, sensor fusion, робототехнике и высокопроизводительным ROS 2 системам. Проанализируй существующую систему локализации ровера, экспериментальные записи и исходный код. Затем предложи максимально эффективное, точное и практически реализуемое решение для комбинированной локализации по потолочным камерам, ArUco/AprilTag и потолочному 3D-лидару Unitree L2.

Рабочий репозиторий:

```text
/home/popka/Drone/mocap-rover
```

Сначала полностью прочитай основной технический бриф:

```text
/home/popka/Drone/mocap-rover/docs/hybrid_localization_optimization_brief.md
```

Не ограничивайся пересказом брифа. Самостоятельно проверь его выводы по исходному коду, CSV и JSONL. Если найдёшь ошибку или неверное предположение, явно укажи это и докажи по данным.

## Обязательные исходные файлы

```text
/home/popka/Drone/mocap-rover/scripts/run_localization.py
/home/popka/Drone/mocap-rover/scripts/run_tag_coverage_experiment.py
/home/popka/Drone/mocap-rover/scripts/analyze_tag_coverage.py
/home/popka/Drone/mocap-rover/localization_contracts/detector.py
/home/popka/Drone/mocap-rover/localization_contracts/apriltag.py
/home/popka/Drone/mocap-rover/localization_contracts/fusion.py
/home/popka/Drone/mocap-rover/localization_contracts/hybrid_rover.py
/home/popka/Drone/mocap-rover/localization_contracts/cuboid.py
/home/popka/Drone/mocap-rover/localization_contracts/image_pipeline.py
/home/popka/Drone/mocap-rover/localization_contracts/contracts.py
/home/popka/Drone/mocap-rover/worlds/mocap_arena_l2.sdf
/home/popka/Drone/mocap-rover/models/rpi_camera/model.sdf
/home/popka/Drone/mocap-rover/models/unitree_l2/model.sdf
/home/popka/Drone/mocap-rover/models/imx219_1280x960.yaml
/home/popka/Drone/mocap-rover/docs/center_sensor_layout.md
/home/popka/Drone/mocap-rover/docs/hybrid_rover_tracking.md
```

## Основной новый эксперимент

Каталог:

```text
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_181432
```

Прочитай все перечисленные файлы:

```text
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_181432/manifest.json
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_181432/summary.json
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_181432/truth.csv
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_181432/camera_frames.csv
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_181432/observations.csv
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_181432/estimates.csv
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_181432/camera_summary.csv
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_181432/spatial_bins.csv
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_181432/comparison_previous.csv
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_181432/odometry_diagnostics.csv
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_181432/ODOMETRY_REPORT.md
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_181432/gazebo.log
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_181432/localization.log
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_181432/runtime/initial_calibration.json
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_181432/runtime/runtime_parameters.json
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_181432/runtime/camera_frames.jsonl
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_181432/runtime/observations.jsonl
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_181432/runtime/odometry.jsonl
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_181432/runtime/status.json
```

## Предыдущий эксперимент для сравнения

```text
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_001059/manifest.json
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_001059/summary.json
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_001059/truth.csv
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_001059/camera_frames.csv
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_001059/observations.csv
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_001059/estimates.csv
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_001059/camera_summary.csv
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_001059/spatial_bins.csv
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_001059/runtime/camera_frames.jsonl
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_001059/runtime/observations.jsonl
/home/popka/Drone/mocap-rover/artifacts/tag_coverage_20260925_001059/runtime/odometry.jsonl
```

Если какого-либо необязательного runtime-файла старого эксперимента нет, продолжай по агрегированным CSV и явно отметь ограничение.

## Условия задачи

Арена приблизительно 12×12 м. На потолке установлены шесть камер IMX219-160 и Unitree L2. Четыре угловые камеры нельзя перемещать, разрешено менять только ориентацию. Две центральные камеры можно разнести друг относительно друга не более чем на 60 см и свободно поворачивать. Лидар расположен по центру и смотрит вниз.

Максимальная скорость tagged rover:

```text
40 км/ч = 11.11 м/с
```

Поддерживаемые маркеры:

```text
AprilTag tag36h11
ArUco DICT_4X4_50
```

Метка нужна для идентификации и абсолютной коррекции, но не обязана быть видимой постоянно. После захвата трек должен продолжаться по 3D-лидару и/или обычному изображению. В системе может быть второй ровер, поэтому запрещено переключать identity на другой объект без подтверждения маркером.

Система должна выдавать в координатах `arena` не реже 100 Гц:

```text
x, y, yaw, vx, vy, yaw_rate, timestamp, covariance, tracking_state, source_mask
```

## Известные результаты, которые нужно перепроверить

Предыдущий прогон:

```text
valid estimates: 8425 / 16814
P50 XY: 0.0854 м
P95 XY: 0.3680 м
P95 yaw: 15.52°
```

Новый прогон после изменения центральных камер:

```text
camera_5: (5.70, 6.00, 2.90), tilt 40° from nadir, yaw 35°
camera_6: (6.30, 6.00, 2.90), tilt 40° from nadir, yaw 215°
valid estimates: 28 / 21361
P50 XY: 0.1354 м
P95 XY: 0.8885 м
P95 yaw: 13.79°
fusion-accepted observations: 3555
median measurement age: 564 мс
P95 measurement age: 18.286 с
dropout horizon: 200 мс
```

Центральные камеры стали чаще видеть метку, но P95 ошибки их отдельных наблюдений выросли приблизительно до 0.85–0.87 м. Было принято физически невозможное PnP-наблюдение с оценкой высоты базы около 1.08 м при реальной высоте около 0.14 м и reprojection error около 1.47 px.

3D-лидар в этих результатах не участвовал: текущий `run_localization.py` использует только RGB и `/clock`.

L2 выдаёт полный скан с частотой 5.55 Гц. При 40 км/ч ровер проходит примерно 2 м между полными сканами. Нужно определить возможность использования timestamp отдельных точек/секторов, deskew и частичных обновлений.

## Требуемый анализ

### 1. Аудит данных

Самостоятельно вычисли:

- частоту кадров каждой камеры;
- detector, PnP и fusion accept rate;
- capture-to-process и capture-to-publish latency;
- распределение measurement age;
- длительность интервалов без валидной позиции;
- ошибки XY/yaw во времени, по камерам и клеткам арены;
- зависимость ошибки от размера метки, reprojection error, качества, края кадра и угла наблюдения;
- частоту неправильной ветви planar PnP;
- связь нагрузки детектора с задержкой;
- причины различий двух прогонов.

### 2. Целевая архитектура

Спроектируй полный поток:

```text
marker acquisition
→ identity confirmation
→ high-rate ROI tracking
→ lidar association
→ asynchronous multi-camera/multi-sensor fusion
→ prediction to current timestamp
→ 100 Hz output
```

### 3. Устранение задержки

Предложи замену `SynchronousObservationArbiter`, учитывая:

- асинхронные updates;
- окно группировки не больше 10–15 мс;
- историю состояний для out-of-sequence measurements;
- репропагацию к текущему timestamp;
- queue depth 1 и latest-frame-wins;
- независимые camera workers/processes;
- C++ для горячего image path;
- ROS 2 best-effort, shared memory и zero-copy;
- обработку только камер, в которых находится прогноз.

Нельзя предлагать увеличение dropout timeout как основное решение. При 40 км/ч задержка 100 мс соответствует 1.11 м движения, 564 мс — 6.27 м.

### 4. Marker acquisition и ROI tracking

Предложи конкретный алгоритм и параметры:

- полный поиск 10–15 Гц;
- projected ROI порядка 160×160 или 240×240 после захвата;
- детектор в ROI до 90 Гц;
- KLT/optical flow или другой markerless tracker между детекциями;
- расширение ROI и переход к reacquisition;
- состояния TRACKING, COASTING, REACQUIRING и LOST;
- работа нескольких камер без double-counting.

### 5. Исправление позы по метке

Проверь подход, в котором IPPE PnP является только одним кандидатом, а XY независимо вычисляется пересечением луча через центр метки с плоскостью известной высоты. Yaw можно оценивать отдельно.

Предложи конкретные gates:

- диапазон base Z;
- Mahalanobis/innovation gate;
- максимальные скорость и ускорение;
- минимальный projected tag size;
- контроль формы четырёхугольника;
- согласованность разных камер;
- зависимость covariance от размера, края кадра, incidence angle и reprojection error.

### 6. Подключение Unitree L2

Спроектируй:

- преобразование scan в `arena`;
- timestamp policy и deskew;
- статическую voxel-карту пустой арены;
- удаление пола, стен и потолочных конструкций;
- prediction crop до кластеризации;
- кластеризацию и проверку габаритов ровера;
- association только с подтверждённым tagged track;
- lidar covariance;
- сопровождение остановившегося ровера;
- защиту от второго ровера;
- частичные обновления по секторам до полного оборота, если это физически возможно.

Объясни, какую точность и частоту реально получить при 5.55 Гц полного скана.

### 7. Fusion/filter

Сравни EKF, UKF, IMM с CV/CA/CTRV и fixed-lag smoother/factor graph. Выбери конкретный вариант для реального времени и обоснуй стоимость.

Минимальное состояние:

```text
[x, y, vx, vy, yaw, yaw_rate]
```

Опиши state transition, process noise, модели tag/lidar/optical-flow, delayed updates, robust loss, covariance propagation, инициализацию и повторный захват identity. Приведи формулы или точный псевдокод.

### 8. Повторная оптимизация камер

Не оптимизируй только геометрический frustum. Целевая функция должна учитывать:

- вероятность детекции;
- projected marker size;
- incidence angle;
- motion blur;
- расстояние до края кадра;
- эмпирическую PnP covariance;
- число одновременно видящих камер;
- вычислительную стоимость;
- вклад лидара.

Определи, сохранить ли текущие 40°, вернуться к 25–30° или выбрать промежуточный наклон. Позиции угловых камер фиксированы. Центральные можно разнести максимум на 60 см. Дай несколько ранжированных вариантов с численным обоснованием.

### 9. Реальный режим IMX219

Сравни:

1. захват 1280×960 и resize до 640×480;
2. прямой аппаратный режим 640×480 ради 90 FPS.

Проверь sensor crop, изменение FOV, необходимость отдельной калибровки и предложи процедуру измерения фактического FOV всех камер.

### 10. План реализации

Составь file-level план именно для `/home/popka/Drone/mocap-rover`:

- какие файлы и классы изменить;
- какие ROS/Gazebo topics использовать;
- какие параметры вынести в конфигурацию;
- какие поля добавить в CSV;
- какие unit, replay и integration tests написать.

Для каждого этапа укажи ожидаемый эффект, риск, проверку, критерий успешности и возможность отката.

## Предварительные критерии приёмки

Проверь реалистичность и предложи обоснованную корректировку при необходимости.

Симуляция на 0.5 м/с:

```text
valid coverage >= 95%
P50 XY <= 0.08 м
P95 XY <= 0.20 м
P95 yaw <= 10°
P95 measurement age <= 50 мс
ID switches = 0
```

Режим до 40 км/ч:

```text
output rate >= 100 Гц
camera ROI tracking >= 90 Гц
P50 end-to-end latency <= 15 мс
P95 end-to-end latency <= 30 мс
честные состояния TRACKING/COASTING/REACQUIRING/LOST
ID switches = 0
```

## Формат ответа

Ответ должен содержать:

1. Краткий итог и главные причины текущего провала.
2. Аудит данных с самостоятельно вычисленными метриками.
3. Таблицу latency budget.
4. Целевую архитектуру и потоки данных.
5. Математику fusion: state, модели, gates, covariance и delayed updates.
6. Камерный тракт: acquisition, ROI, optical flow и multi-camera aggregation.
7. Лидарный тракт: preprocessing, deskew, clustering и association.
8. Несколько вариантов углов камер с численным сравнением.
9. Таблицу рекомендуемых параметров и диапазонов autotuning.
10. File-level план реализации.
11. Воспроизводимый план экспериментов и необходимых CSV.
12. Риски, неизвестные и недостающие данные.
13. Приоритет реализации по эффекту и риску.

Не давай только общих советов. Связывай предложения с экспериментальными данными, физикой, формулами или явно обозначенными предположениями. Не считай геометрический FOV доказательством детектируемости. Не объявляй устаревшую позицию валидной увеличением timeout. Не предполагай, что лидар улучшил существующие CSV: он в них не участвовал.
