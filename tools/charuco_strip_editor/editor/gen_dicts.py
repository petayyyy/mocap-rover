#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Одноразовый скрипт для выгрузки битовых матриц предопределённых словарей
ArUco/AprilTag из OpenCV в компактный JS-литерал, который вставляется
в aruco_map_editor.html (константа ARUCO_DICTS). Сам редактор от этого
скрипта не зависит -- запускать только при необходимости пересобрать данные
(например, при появлении новых версий OpenCV или новых словарей).

Требует: pip install opencv-contrib-python (модуль cv2.aruco).

Способ извлечения бит -- рендер эталонного изображения метки функцией
generateImageMarker() (без рамки, borderBits=0) и посэмплирование центра
каждой ячейки. Это гарантирует пиксель-в-пиксель соответствие тому, что
реально сгенерирует/задетектирует OpenCV, в отличие от попытки вручную
распаковать внутренний формат bytesList (там неочевидный порядок бит,
завязанный на реализацию).

Запуск:
    python gen_dicts.py dicts_data.js
"""
import sys
import cv2
import cv2.aruco as aruco

# Список словарей, которые нужны редактору (см. ТЗ).
DICTS = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
    "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11",
]

CELL_PX = 8  # разрешение рендера на одну ячейку кода (без рамки)


def extract_marker_bits(dictionary, marker_id, n):
    """Возвращает n x n сетку 0/1 (1 = белая ячейка) для метки marker_id."""
    # borderBits=0 недопустим в OpenCV 4.13, поэтому рендерим с рамкой в 1
    # ячейку и сэмплируем только внутреннюю область кода, пропуская рамку.
    side = (n + 2) * CELL_PX
    img = dictionary.generateImageMarker(marker_id, side, borderBits=1)
    bits = 0
    for r in range(n):
        for c in range(n):
            y = int((r + 1 + 0.5) * CELL_PX)
            x = int((c + 1 + 0.5) * CELL_PX)
            val = img[y, x]
            bit = 1 if val > 127 else 0
            bits = (bits << 1) | bit
    return bits


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "dicts_data.js"
    out = open(out_path, "w", encoding="utf-8", newline="\n")
    out.write("// Автосгенерировано gen_dicts.py -- НЕ редактировать руками.\n")
    out.write("// Формат: {n: сторона в ячейках, mcb: maxCorrectionBits, codes: [hex...]}\n")
    out.write("// codes[id] -- hex-строка из n*n бит, старший бит = ячейка (0,0) (левый верхний угол),\n")
    out.write("// далее по строкам слева направо, сверху вниз.\n")
    out.write("const ARUCO_DICTS = {\n")
    for name in DICTS:
        d = aruco.getPredefinedDictionary(getattr(aruco, name))
        n = d.markerSize
        count = d.bytesList.shape[0]
        mcb = int(d.maxCorrectionBits)
        hexw = (n * n + 3) // 4
        codes = []
        for mid in range(count):
            bits = extract_marker_bits(d, mid, n)
            codes.append(format(bits, "0%dx" % hexw))
        out.write('  "%s": {n:%d, mcb:%d, codes:[%s]},\n' % (
            name, n, mcb, ",".join('"%s"' % c for c in codes)
        ))
        sys.stderr.write("%s: n=%d count=%d mcb=%d ok\n" % (name, n, count, mcb))
    out.write("};\n")
    out.close()


if __name__ == "__main__":
    main()
