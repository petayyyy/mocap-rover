# Промпт 13 — Визуализация и диагностика

> Фаза SIMULATION. Все работы и критерии этого этапа выполняются на Gazebo,
> синтетических данных и replay. Драйверы IMX296/libcamera, udev, реальные камеры,
> сеть узлов захвата и аппаратный trigger — только фаза HARDWARE после S15.
> Общие интерфейсы и поля конфигурации допустимы; отсутствие оборудования
> не блокирует этап и не требует спрашивать пользователя о подключении камер.


Сначала выполни инструкции из `docs/localization_plan/prompts/00_context.md`.
Прочитай результаты предыдущего S-этапа в STATUS.md и используй существующие интерфейсы.

Добавь карту 12×12 м с траекториями, скоростями, направлениями при наличии,
ковариациями, источниками наблюдений и покрытиями камер. Разделяй measured,
filtered и predicted. Истинные траектории показывай только в режиме sim evaluation.

Шесть previews + увеличенный выбранный кадр, tag corners, bbox/mask/ID,
время экспозиции, age, drops, stale overlay и calibration version. Графики
capture/detector/accepted/output Hz, latency percentiles, skew, residuals,
CPU/GPU/VRAM и queue depths. Для LOST не оставляй зелёную статичную точку.

Запись/воспроизведение с timeline, pause/seek и session reset; экспорт config
и отчёта. Ограничь preview FPS/resolution отдельно от capture. При отключении
браузера одометрия продолжает работу. Не передавай все raw кадры через JSON.
Критерий: видимое UI подтверждено скриншотом/ручным проходом, данные реальные,
сравнены latency и rates с открытой панелью и без неё.

В конце обнови STATUS.md и дай команды воспроизведения результата.

## Дополнение из согласованного ресерча

Прочитай `docs/localization_plan/05_research_integration.md`.
R03/R05: показывай геометрический FOV отдельно от читаемости, места
occlusion/слепых зон, pixels/marker, exposure uncertainty, attitude validity.
Добавь measured TRACKING availability по времени/кадрам; COASTING не засчитывать.
Panorama возможна только как UI-preview, MOG2 — необязательный индикатор,
не источник мировой позы. Покажи ошибки по каждой контрольной точке и швам.
