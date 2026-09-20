# Промпт 05 — Время, синхронизация и replay

> Фаза SIMULATION. Все работы и критерии этого этапа выполняются на Gazebo,
> синтетических данных и replay. Драйверы IMX296/libcamera, udev, реальные камеры,
> сеть узлов захвата и аппаратный trigger — только фаза HARDWARE после S15.
> Общие интерфейсы и поля конфигурации допустимы; отсутствие оборудования
> не блокирует этап и не требует спрашивать пользователя о подключении камер.


Сначала выполни инструкции из `docs/localization_plan/prompts/00_context.md`.
Прочитай результаты предыдущего S-этапа в STATUS.md и используй существующие интерфейсы.

Реализуй общий time domain и сохранение capture/receive/processed timestamps,
source clock, exposure duration, uncertainty и sequence. Для driver timestamp
установи, что он означает: start/end/mid exposure или delivery; не угадывай.
При нескольких узлах поддержи offset/drift и диагностику коррекции часов.

Моделируй skew, offset/drift, длительность экспозиции, trigger counter и пропуски
в адаптере симуляции. Для идеально синхронного и free-running режимов используй
явные simulated capabilities. Не реализуй аппаратный trigger, PTP/NTP на узлах
или драйверные timestamps: это H03. Не жди все шесть кадров для каждого объекта.

Запись: кадры + метаданные + config/calibration/model hashes; воспроизведение
с исходным временем и независимым arrival scheduling. При seek/reset новая session.
Тесты: skew, reorder, jitter, clock jump, пропуск camera_6; latency остальных
не растёт. Критерий: reproducible replay и отчёт качества времени, без ложного
статуса synchronized по одному лишь равенству FPS.

В конце обнови STATUS.md и дай команды воспроизведения результата.

## Дополнение из согласованного ресерча

Прочитай `docs/localization_plan/05_research_integration.md`.
R03/R04: sweep timestamp uncertainty 0/1/5/20 мс отдельно от arrival delay
и exposure skew; сравни ошибку при 2 м/с. Добавь виртуальные observation/video
сетевые каналы с независимыми delays/drops. ≤50 мс до приводов — отдельная
гипотеза полного контура; измеренный odometry latency не подменяет её.
