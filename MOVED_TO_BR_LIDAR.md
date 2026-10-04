# mocap-rover переехал в br_lidar

С 4 октября 2026 года **итоговый репозиторий — [mi1en/br_lidar](https://github.com/mi1en/br_lidar)**.
Здесь отлаживались алгоритмы и камеры; в br_lidar этот код лежит в каталоге `mocap/`
и подключён к навигации и пульту. Новую работу делать там. Этот репозиторий — архив
и место, откуда ещё не перенесённые коммиты забираются в br_lidar.

## Куда что переехало

Путь меняется одинаково для всех файлов: `mocap-rover/<путь>` → `br_lidar/mocap/<путь>`.

| В mocap-rover | В br_lidar |
|---|---|
| `localization_contracts/`, `scripts/`, `pi_cam/`, `tools/`, `tests/`, `config/`, `docs/` … | `mocap/` + тот же путь |
| `.venv` (numpy 1.26, OpenCV 4.10) | `mocap/.venv`, ставится из `mocap/requirements-field.txt` |
| — | `navigation/`, `airy_py/`, `ui/` — навигация, Airy, пульт (свой `navigation/.venv`, numpy 2.5) |

Как mocap связан с навигацией, порядок подготовки стенда и запуск —
[`br_lidar/mocap/docs/INTEGRATION.md`](https://github.com/mi1en/br_lidar/blob/migrate-mocap-rover/mocap/docs/INTEGRATION.md).

## Что уже перенесено (на 4 октября 2026)

| mocap-rover | Что | В br_lidar |
|---|---|---|
| всё до `6fcb0ad` | основной код, срез дерева без истории | `c7b9f52` на `main` (mi1en, вместе с её доработками стенда: оператор `mocap`, PTP, выпрямление кадров, полевые отчёты) |
| `b3b2427` | SAM2 как второе мнение для соперника | `11183c0` на `migrate-mocap-rover` |
| `9a15543` | инструкция по железу и скрипты установки | `397109f` на `migrate-mocap-rover` |
| `35da4b6` | станция калибровки ChArUco | `8556cda` на `migrate-mocap-rover` |

Ветка `migrate-mocap-rover` в br_lidar = `main` + влитая `pult-ui` + три коммита выше +
интеграция: живой тракт `mocap/scripts/run_mocap_live.py`, приём поз в пульте
(`navigation/runtime/mocap_source.py`), калибровка в системе кольца лидара. Ветка
запушена, **в `main` br_lidar ещё не влита** — нужен PR и согласие mi1en.

## Что ещё НЕ перенесено

| Где | Коммиты | Что |
|---|---|---|
| ветка `field/round2-h264` | `4053997`, `29416f5` | полевой тест, раунд 2: поиск неисправности коммутатора, профиль с одним окном, проба H.264 (`pi_cam/h264_probe.py`) |
| рабочая копия на ноутбуке (не закоммичено) | — | правки `localization_contracts/camera_worker.py`, `scripts/replay_dataset.py` |

Ветка `fix/camera-model-and-async-fusion` уже влита в `main`, переносить нечего.

Незакоммиченные правки сначала закоммитить (в `field/round2-h264` или новую ветку) и
запушить в mocap-rover, потом переносить по порядку ниже. Оба файла в br_lidar уже
изменены интеграцией (там добавлен живой режим), так что при переносе возможен конфликт:
разрешать, сохраняя обе стороны.

## Как перенести коммит из mocap-rover в br_lidar

Переносить через `git am` с префиксом каталога: так сохраняются автор, сообщение и
дата, а трёхстороннее слияние разбирает пересечения с правками в br_lidar.

1. В br_lidar взять свежую ветку интеграции:

   ```bash
   cd ~/br_lidar
   ```

   ```bash
   git fetch origin && git checkout migrate-mocap-rover && git pull --ff-only
   ```

2. Достать объекты mocap-rover (нужны для трёхстороннего слияния):

   ```bash
   git fetch https://github.com/petayyyy/mocap-rover.git field/round2-h264:refs/tmp/mocap-rover
   ```

3. Сделать патчи и наложить их с префиксом `mocap/` (пример — два коммита
   `field/round2-h264`):

   ```bash
   git format-patch -o /tmp/mr-patches 35da4b6..refs/tmp/mocap-rover
   ```

   ```bash
   git am -3 --directory=mocap /tmp/mr-patches/*.patch
   ```

   При конфликте: поправить файлы, `git add <файлы>`, `git am --continue`.
   Отменить всё: `git am --abort`.

4. Проверить в двух окружениях. Тесты mocap гоняются из `mocap/` его окружением:

   ```bash
   cd mocap && .venv/bin/python -m pytest tests -q
   ```

   Тесты навигации, `airy_py` и пульта — из корня окружением навигации:

   ```bash
   cd .. && navigation/.venv/bin/python -m pytest navigation/runtime/tests airy_py ui/tests -q
   ```

   Падения `test_check_charuco_poses.py` и `test_scenarios.py::test_styles_preserve_installation`
   есть и на `main`, они не от переноса.

5. Убрать временную ссылку и запушить:

   ```bash
   git update-ref -d refs/tmp/mocap-rover
   ```

   ```bash
   git push origin migrate-mocap-rover
   ```

6. Дописать строку в таблицу «Что уже перенесено» выше (и в
   `br_lidar/mocap/docs/IMPORT_NOTES.md`), чтобы было видно, что забрано.

Не копировать файлы руками поверх `mocap/`: так теряется авторство и молча
затираются правки, сделанные уже в br_lidar.

## Как это влить в main br_lidar (с mi1en)

1. PR `migrate-mocap-rover` → `main` в mi1en/br_lidar:
   https://github.com/mi1en/br_lidar/pull/new/migrate-mocap-rover
   В описании дать ссылку на `mocap/docs/INTEGRATION.md` и раздел «Проверено и не проверено».
2. Если `main` ушёл вперёд, влить его в ветку и разобрать конфликты у себя,
   а не в веб-интерфейсе:

   ```bash
   git fetch origin && git checkout migrate-mocap-rover && git merge origin/main
   ```

3. После слияния PR работать от `main` br_lidar: новые ветки от него, PR в него же.
   Сюда, в mocap-rover, больше не коммитить; если пришлось — перенести по разделу выше
   и отметить в таблице.

## Чего не переносить

`artifacts/` (датасеты, записи, результаты реплея), `models/` с весами (SAM2, YOLO),
`.venv`. Они в git не хранятся ни здесь, ни там; нужные датасеты копировать
между машинами отдельно.
