# Завершение SIM-этапов после перезагрузки

Статус: **NOT ACCEPTED**. Новые прогоны Gazebo приостановлены по прямому
указанию пользователя: симулятор занят другой задачей. Не запускать сервер,
клиенты локализации, управляющие сценарии или проверки топиков до разрешения.
Автономные тесты и анализ сохранённых записей разрешены.

## Что фактически проверено до приостановки

- RTX 3070, NVIDIA 580.178.04: kernel/userspace согласованы.
- GUI арены, шесть RGB-потоков, перемещение обоих роверов.
- Устранено повторное копирование всего protobuf image.data для каждой строки.
  Обработка вынесена из transport callback в отдельные workers с очередью
  последнего кадра. Новый профиль по умолчанию: 1440×1080, 30 sim Hz.
  Все девять цветных источников сохранены, тени считают два источника;
  исходные девять теневых источников доступны через `--shadow-lights 9`.
- Калибровка шести камер по отрендеренным шахматным доскам:
  12 обучающих и 3 контрольных положения на камеру, 648 training corners.
  K/D получены calibrateCamera, extrinsics — PnP/RANSAC + LM по измеренным
  углам и известным размерам/положениям калибровочных досок.
  P95 hold-out reprojection: 0.289, 0.321, 0.318, 0.296, 0.319, 0.350 px.
  Camera ground truth не является входом решателя.
- YOLO11n обучен 40 эпох на 720 изображениях Gazebo, сцены разделены между
  train/val/test вместе со всеми шестью камерами. На test: opponent
  precision 0.963, recall 1.0, mAP50 0.992, mAP50–95 0.853.
  Проверен только steel/colored, остальные стили/свет не приняты.
- Исправлены отдельные transforms ID 0 и ID 1. Печатные рисунки повёрнуты
  на 180° для согласования канонических углов декодера с заданными осями tag.
  Верхний ID 0 в живом прогоне давал yaw около нуля при yaw ровера 0.
  Нижний ID 1 проверен автономным тестом геометрии; новый live flip ещё нужен.
- Минутный live runtime: GUI + шесть JPEG previews, AprilTag, YOLO,
  независимые таймеры, ROS2 publishers, запись observations/odometry.
  Выходы по wall-time: **30.010 / 15.000 Гц**; новые принятые измерения:
  **19.974 / 12.003 Гц** (свой/второй).
  P99 интервала публикации: **37.496 / 70.829 мс**.
  Ошибок runtime и потерь очереди записи в этом прогоне не было.
  19.974 не округляется до выполнения порога ≥20.

## Почему это ещё не приёмка

В evaluator-рекордере обнаружена ошибка: timestamp читался из заголовка
Pose_V, который пуст, вместо вложенного Pose.header.stamp. Сохранённый
`static_truth3.jsonl` имеет нулевые timestamps и не годится для временного
сопоставления. `static_evaluation.json` честно содержит 0 aligned outputs,
null accuracy и FAIL. Подмена времени временем прихода не выполнялась.
Чтение исправлено и покрыто регрессионным тестом, новый прогон отложен.

Семантика timestamp подтверждена в
[исходнике Gazebo PosePublisher](https://github.com/gazebosim/gz-sim/blob/gz-sim8/src/systems/pose_publisher/PosePublisher.cc).

После этого live-прогона внесены дополнительные правки: ограничение BLAS
потоков, ограничение запуска YOLO до 15 batch/s, clock-stall invalidation,
защита от старой версии калибровки, atomic apply/rollback, корректные отдельные
поля wall/sim Hz, контролируемое завершение записи и защита от повторного
запуска/перезаписи сессии. Они проверены автономно, **не повторно в Gazebo**.

Ковариации и геометрия cuboid пока требуют проверки на траекториях и швах.
ROS-выход сейчас planar yaw + измеренный Z; полный quaternion перевёрнутого
ровера и TF ещё не являются принятым интерфейсом. В наблюдениях PnP сохранена
полная матрица rotation_arena_base. Панель предоставляет импорт/откат JSON;
полные формы настроек, remap работающих subscriptions и синхронизация ещё
требуют завершения S12. Запись содержит наблюдения и одометрию; шесть JPEG
превью перезаписываются, полного RGB rosbag пока нет.

## Автономная проверка последних правок

Системный Python: 90 tests, OK, skipped=2; `.venv`: 90 tests, OK, skipped=2.
Пропущены только два теста на старых снимках из `/tmp`. Есть регрессии на
единственное чтение protobuf-буфера, bounded latest-frame queue, различные
transforms меток, оси печатных кодов, восстановление после пропуска,
недопустимость будущих измерений, шум скорости, cuboid projection,
отсутствие изменения active config при отказе runtime, nested Pose timestamp
и JSON-отчёт evaluator. Проверены shell/Python syntax и diff whitespace.

Chrome headless с сохранённой сессией подтвердил отрисовку обоих роверов и
шести JPEG-превью. Процесс браузера и fixture HTTP server завершены. Тест не
подключался к Gazebo. Текущие live-изменения ещё требуют прогона после разрешения.

## Артефакты

Всё ниже сохранено в проекте, не в `/tmp`:

- `artifacts/post-reboot/calibrated_cameras.json`
- `artifacts/post-reboot/calibrated_cameras.observations.json`
- `artifacts/post-reboot/calibration_views/`
- `artifacts/post-reboot/yolo_dataset/{manifest.json,data.yaml,images,labels}`
- `artifacts/post-reboot/training/rover/weights/best.pt`
- `artifacts/post-reboot/training/rover/results.csv`
- `artifacts/post-reboot/training/holdout/`
- `artifacts/post-reboot/pretrained/yolo11n.pt`
- `artifacts/post-reboot/live_static/`
- `artifacts/post-reboot/static_truth3.jsonl` — timestamps непригодны
- `artifacts/post-reboot/static_evaluation.json`
- `artifacts/post-reboot/dashboard_offline/{dashboard.png,page.html,report.json}`
- `artifacts/post-reboot/offline_tests.log`, `offline_venv_tests.log`

`artifacts/` и `.venv/` исключены из Git; при переносе проекта их надо сохранить
отдельно или воспроизвести. Нельзя удалять эти файлы как временный мусор перед
продолжением. Новый запуск требует нового каталога записи.

## Что дальше после разрешения

1. Короткий статический прогон текущего кода: valid timestamps, качество
   сопоставления, метрики wall/sim, оба ROS topic, шесть обновляемых JPEG.
2. Траектории до 1 м/с, швы/границы, динамическая точность, скорость, yaw;
   переворот и ID 1, маскирование меток, сближение роверов.
3. Стили/свет и независимые seeds; дополнение датасета и обучение при провалах.
4. Fault matrix на живом тракте: задержки, drop/reorder, отключение камеры,
   clock pause/reset, восстановление, смена/откат калибровки.
5. Доработать настройки, remap и replay/запись изображений; проверить нагрузку.
6. Настоящие 30 минут wall-time с UI/превью/записью и сравнение без UI.
7. Обязательные R01–R06 сравнения. Только затем решение S15/SIM_ACCEPTED.

Аппаратная фаза IMX296/udev/libcamera не начата.
