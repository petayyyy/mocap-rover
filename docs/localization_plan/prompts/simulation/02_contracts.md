# Промпт 02 — Каркас и контракты

> Фаза SIMULATION. Все работы и критерии этого этапа выполняются на Gazebo,
> синтетических данных и replay. Драйверы IMX296/libcamera, udev, реальные камеры,
> сеть узлов захвата и аппаратный trigger — только фаза HARDWARE после S15.
> Общие интерфейсы и поля конфигурации допустимы; отсутствие оборудования
> не блокирует этап и не требует спрашивать пользователя о подключении камер.


Сначала выполни инструкции из `docs/localization_plan/prompts/00_context.md`.
Прочитай результаты предыдущего S-этапа в STATUS.md и используй существующие интерфейсы.

Создай минимальный workspace/пакеты для общего ядра, ROS интерфейсов,
конфигурации и launch. Реализуй версии схем и структуры Observation, TrackStatus,
CameraStatus и CalibrationSet по 02_architecture.md. Не создавай фиктивные детекторы.
Вынеси геометрию/время в тестируемые библиотеки независимо от UI.

Определи frame IDs, метрические единицы, семантику timestamps и covariance,
object identity, session/reset IDs. У Odometry pose и twist разные frame semantics;
покрой преобразование скорости и ковариации тестами. Невалидная ориентация должна
быть явной. Определи TF владельцев и поведение LOST.

Сделай schema validation, load/save, version/hash, staged config и atomic apply.
Тесты: неверные K/размеры/единицы, несовместимая версия, частичное обновление,
прыжок времени. Критерий: минимальный launch и конфигурация проходят проверку,
контракты документированы и пригодны для sim/hardware/replay.

В конце обнови STATUS.md и дай команды воспроизведения результата.

## Дополнение из согласованного ресерча

Прочитай `docs/localization_plan/05_research_integration.md`.
R02/R04: добавь marker registry (family+ID, board, bundle, T_base_marker),
6D pose, attitude_state, observation uncertainty и covariance с единицами м²/рад².
Раздели observation/video planes, existence state и quality state. Контракт
per-camera пиксельных признаков допустим, глобальная ассоциация — в метрах.
