#!/usr/bin/env python3
"""
Веб-интерфейс для Raspberry Pi 5 + IMX219.

Запуск:
    python3 imx219_fov_stream.py
Открыть в браузере:
    http://<IP_RASPBERRY_PI>:8080
"""

from collections import deque
from io import BytesIO
import statistics
import threading
import time

from flask import Flask, Response, jsonify, render_template_string, request
from PIL import Image
from picamera2 import Picamera2

import numpy as np


app = Flask(__name__)


# Режимы из вывода rpicam-hello --list-cameras для IMX219 пользователя.
# full_sensor=True означает чтение всей области 3280x2464 с биннингом/масштабированием.
MODES = {
    "1640x1232": {
        "width": 1640,
        "height": 1232,
        "sensor_size": (1640, 1232),
        "bit_depth": 8,
        "max_fps": 83.70,
        "full_sensor": True,
        "description": "вся матрица, 2x2 binning, 8-bit",
    },
    "3280x2464": {
        "width": 3280,
        "height": 2464,
        "sensor_size": (3280, 2464),
        "bit_depth": 10,
        "max_fps": 21.19,
        "full_sensor": True,
        "description": "вся матрица, полный размер, 10-bit",
    },
    "640x480": {
        "width": 640,
        "height": 480,
        "sensor_size": (640, 480),
        "bit_depth": 8,
        "max_fps": 206.65,
        "full_sensor": False,
        "description": "кроп 1280x960, 8-bit",
    },
    "1920x1080": {
        "width": 1920,
        "height": 1080,
        "sensor_size": (1920, 1080),
        "bit_depth": 10,
        "max_fps": 47.57,
        "full_sensor": False,
        "description": "кроп 1920x1080, 10-bit",
    },
}

# Во сколько раз уменьшать картинку перед JPEG для веб-превью.
# Кодирование в PIL — самая медленная часть; уменьшение в 2 раза делает его
# вчетверо дешевле и снимает нагрузку с потока захвата.
PREVIEW_DIVISOR = 2


class CameraService:
    def __init__(self):
        self.camera = Picamera2()
        self.lock = threading.RLock()
        self.condition = threading.Condition()

        self.latest_jpeg = None
        self.frame_id = 0

        # Сырой кадр, ожидающий кодирования. Если кодировщик занят,
        # новый кадр просто заменяет старый — поток захвата никогда не ждёт.
        self.pending_frame = None
        self.pending_condition = threading.Condition()

        self.timestamps = deque(maxlen=240)     # SensorTimestamp доставленных кадров
        self.frame_durations = deque(maxlen=240)  # FrameDuration, мкс — истинный такт сенсора
        self.captured_frames = 0
        self.dropped_frames = 0
        self.center_pixel = None

        self.last_error = None
        self.running = True
        self.mode_name = "1640x1232"
        self.requested_fps = 83.0
        self.configured = False
        self.sensor_config = {}
        self._configure_locked(self.mode_name, self.requested_fps)

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        self.encoder_thread = threading.Thread(target=self._encode_loop, daemon=True)
        self.encoder_thread.start()

    def _configure_locked(self, mode_name, requested_fps):
        mode = MODES[mode_name]
        if self.configured:
            self.camera.stop()
        config = self.camera.create_video_configuration(
            main={
                "size": (mode["width"], mode["height"]),
                # ВАЖНО: в picamera2 имя формата читается в обратном порядке байт.
                # "BGR888" отдаёт numpy-массив с каналами в порядке R,G,B —
                # именно это ждёт PIL. "RGB888" дал бы BGR и поменял местами
                # красный с синим.
                "format": "BGR888",
            },
            sensor={
                "output_size": mode["sensor_size"],
                "bit_depth": mode["bit_depth"],
            },
            controls={"FrameRate": float(requested_fps)},
            buffer_count=6,
        )
        self.camera.configure(config)
        self.camera.start()
        self.configured = True
        self.mode_name = mode_name
        self.requested_fps = float(requested_fps)

        # Какой режим сенсора libcamera выбрала на самом деле.
        actual = self.camera.camera_configuration().get("sensor", {})
        self.sensor_config = {
            "output_size": list(actual.get("output_size", ())),
            "bit_depth": actual.get("bit_depth"),
        }

        self.timestamps.clear()
        self.frame_durations.clear()
        self.captured_frames = 0
        self.dropped_frames = 0
        self.last_error = None

    def configure(self, mode_name, requested_fps):
        if mode_name not in MODES:
            raise ValueError("Неизвестный режим камеры")
        if requested_fps <= 0 or requested_fps > MODES[mode_name]["max_fps"] + 1:
            raise ValueError("Запрошенная частота не поддерживается этим режимом")
        with self.lock:
            self._configure_locked(mode_name, requested_fps)

    def _capture_loop(self):
        """Забирает каждый кадр сенсора и снимает метаданные.

        Тяжёлое JPEG-кодирование вынесено в отдельный поток, поэтому этот цикл
        успевает за сенсором и измеренный FPS отражает реальный такт матрицы.
        """
        while self.running:
            try:
                with self.lock:
                    request_obj = self.camera.capture_request()
                    metadata = request_obj.get_metadata()

                    # Кадр копируем только если кодировщик готов его принять,
                    # иначе make_array зря тратит сотни МБ/с на memcpy.
                    with self.pending_condition:
                        encoder_idle = self.pending_frame is None
                        if encoder_idle:
                            self.pending_frame = request_obj.make_array("main")
                            self.pending_condition.notify_all()
                        else:
                            self.dropped_frames += 1
                    request_obj.release()

                self.captured_frames += 1

                sensor_timestamp = metadata.get("SensorTimestamp")
                if sensor_timestamp is not None:
                    self.timestamps.append(sensor_timestamp)

                frame_duration = metadata.get("FrameDuration")
                if frame_duration:
                    self.frame_durations.append(frame_duration)
            except Exception as exc:  # продолжать работу после временной ошибки камеры
                self.last_error = str(exc)
                time.sleep(0.2)

    def _encode_loop(self):
        while self.running:
            with self.pending_condition:
                self.pending_condition.wait_for(
                    lambda: self.pending_frame is not None, timeout=1.0
                )
                frame = self.pending_frame
                self.pending_frame = None
            if frame is None:
                continue
            try:
                # Центральный пиксель — для объективной проверки порядка каналов.
                h, w = frame.shape[:2]
                self.center_pixel = [int(v) for v in frame[h // 2, w // 2][:3]]

                if PREVIEW_DIVISOR > 1:
                    # Срез даёт несмежный массив — PIL требует смежный буфер.
                    frame = np.ascontiguousarray(
                        frame[::PREVIEW_DIVISOR, ::PREVIEW_DIVISOR]
                    )

                image = Image.fromarray(frame, "RGB")
                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=78, optimize=False)
                jpeg = buffer.getvalue()

                with self.condition:
                    self.latest_jpeg = jpeg
                    self.frame_id += 1
                    self.condition.notify_all()
            except Exception as exc:
                self.last_error = str(exc)
                time.sleep(0.2)

    def sensor_fps(self):
        """Истинная частота сенсора по FrameDuration — не зависит от дропов."""
        if not self.frame_durations:
            return 0.0
        median_us = statistics.median(self.frame_durations)
        return 1_000_000 / median_us if median_us > 0 else 0.0

    def delivered_fps(self):
        """Сколько кадров реально прошло через поток захвата."""
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = (self.timestamps[-1] - self.timestamps[0]) / 1_000_000_000
        return (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    def status(self):
        mode = MODES[self.mode_name]
        with self.condition:
            frame_id = self.frame_id
        return {
            "mode": self.mode_name,
            "width": mode["width"],
            "height": mode["height"],
            "full_sensor": mode["full_sensor"],
            "description": mode["description"],
            "requested_fps": self.requested_fps,
            "sensor_fps": round(self.sensor_fps(), 2),
            "delivered_fps": round(self.delivered_fps(), 2),
            "captured_frames": self.captured_frames,
            "dropped_frames": self.dropped_frames,
            "sensor_config": self.sensor_config,
            "center_pixel": self.center_pixel,
            "frame_id": frame_id,
            "error": self.last_error,
        }

    def mjpeg_stream(self):
        last_id = -1
        while True:
            with self.condition:
                self.condition.wait_for(
                    lambda: self.latest_jpeg is not None and self.frame_id != last_id,
                    timeout=2.0,
                )
                jpeg = self.latest_jpeg
                last_id = self.frame_id
            if jpeg is None:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n\r\n"
                + jpeg
                + b"\r\n"
            )


camera = CameraService()


HTML = r"""
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IMX219 — FPS и угол обзора</title>
  <style>
    body { font-family: sans-serif; max-width: 1100px; margin: 20px auto; padding: 0 14px; background:#f4f6f8; color:#17202a; }
    .card { background:white; border-radius:12px; padding:16px; margin:12px 0; box-shadow:0 2px 8px #0001; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:12px; }
    label { display:block; margin:8px 0; } input, select, button { font-size:16px; padding:8px; width:100%; box-sizing:border-box; }
    button { cursor:pointer; background:#1769aa; color:white; border:0; border-radius:6px; margin-top:8px; }
    button:hover { background:#0e4f83; }
    #stream { width:100%; max-height:70vh; object-fit:contain; background:#111; }
    .value { font-weight:bold; font-size:1.15em; }
    .ok { color:#16803c; } .warn { color:#a55b00; }
    small { color:#5d6870; }
    #swatch { display:inline-block; width:40px; height:20px; border:1px solid #999; vertical-align:middle; }
  </style>
</head>
<body>
  <h1>IMX219: поток, FPS и угол обзора</h1>

  <div class="card">
    <img id="stream" src="/video_feed" alt="Поток камеры">
  </div>

  <div class="card">
    <h2>Настройки камеры</h2>
    <div class="grid">
      <label>Разрешение
        <select id="mode"></select>
      </label>
      <label>Целевая частота, FPS
        <input id="fps" type="number" min="1" max="207" step="0.1" value="83">
      </label>
      <label>Режим матрицы
        <select id="fullSensor">
          <option value="any">Все режимы</option>
          <option value="full">Только полная матрица</option>
          <option value="crop">Только кроп</option>
        </select>
      </label>
    </div>
    <button onclick="applySettings()">Применить настройки</button>
    <p id="applyResult"></p>
  </div>

  <div class="card">
    <h2>Фактическая частота</h2>
    <div class="grid">
      <div>Режим: <span class="value" id="currentMode">—</span></div>
      <div>Матрица: <span class="value" id="currentSensor">—</span></div>
      <div>Режим сенсора: <span class="value" id="sensorConfig">—</span></div>
      <div>FPS сенсора: <span class="value" id="sensorFps">—</span></div>
      <div>Доставлено в поток: <span class="value" id="deliveredFps">—</span></div>
      <div>Кадров / пропущено: <span class="value" id="frameCount">—</span></div>
    </div>
    <p><small><b>FPS сенсора</b> считается по медиане FrameDuration из метаданных камеры — это истинный такт матрицы, он не зависит от того, успевает ли JPEG-кодировщик.
    <b>Доставлено в поток</b> — сколько кадров прошло через Python. Разница между ними — потери на кодирование, а не проблема камеры.</small></p>
  </div>

  <div class="card">
    <h2>Проверка цветов</h2>
    <p>Наведите камеру на заведомо <b>красный</b> объект так, чтобы он закрывал центр кадра, и сверьте значения. Красный канал должен быть наибольшим.</p>
    <p>Центральный пиксель: <span class="value" id="centerPixel">—</span> <span id="swatch"></span></p>
    <small>Значения берутся из numpy-массива до кодирования в JPEG, поэтому показывают порядок каналов на стороне камеры, а не то, как их отрисовал браузер.</small>
  </div>

  <div class="card">
    <h2>Расчёт угла</h2>
    <p>Введите расстояние до объекта и его видимую ширину. Для проверки объектива укажите ширину всей стены/сцены, попавшей в кадр.</p>
    <div class="grid">
      <label>Расстояние до объекта
        <input id="distance" type="number" min="0.0001" step="any" placeholder="например, 1.0">
      </label>
      <label>Ширина объекта в кадре
        <input id="objectWidth" type="number" min="0" step="any" placeholder="например, 5.7">
      </label>
      <label>Единицы измерения
        <select id="units"><option value="м">метры</option><option value="см">сантиметры</option><option value="мм">миллиметры</option></select>
      </label>
    </div>
    <button onclick="calculateFov()">Рассчитать угол</button>
    <p>Угол по ширине кадра: <span class="value" id="angle">—</span></p>
    <small>Формула: 2 × atan(ширина / (2 × расстояние)). Для сильно искажённого fisheye это практическая оценка, а не точная калибровка.</small>
  </div>

<script>
let modes = {};
async function loadModes() {
  const response = await fetch('/api/modes');
  modes = await response.json();
  fillModes();
}
function fillModes() {
  const filter = document.getElementById('fullSensor').value;
  const select = document.getElementById('mode');
  const old = select.value;
  select.innerHTML = '';
  for (const [name, mode] of Object.entries(modes)) {
    if (filter === 'full' && !mode.full_sensor) continue;
    if (filter === 'crop' && mode.full_sensor) continue;
    const option = document.createElement('option');
    option.value = name;
    option.textContent = `${name} — до ${mode.max_fps} FPS — ${mode.description}`;
    select.appendChild(option);
  }
  if ([...select.options].some(o => o.value === old)) select.value = old;
}
document.getElementById('fullSensor').addEventListener('change', fillModes);
async function applySettings() {
  const response = await fetch('/api/configure', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({mode:document.getElementById('mode').value, fps:parseFloat(document.getElementById('fps').value)})
  });
  const data = await response.json();
  const el = document.getElementById('applyResult');
  el.textContent = data.error || 'Настройки применены';
  el.className = data.error ? 'warn' : 'ok';
}
async function updateStatus() {
  const data = await (await fetch('/api/status')).json();
  document.getElementById('currentMode').textContent = `${data.mode} (${data.description})`;
  document.getElementById('currentSensor').textContent = data.full_sensor ? 'ПОЛНАЯ' : 'КРОП';
  document.getElementById('currentSensor').className = data.full_sensor ? 'value ok' : 'value warn';

  const sc = data.sensor_config || {};
  const size = (sc.output_size || []).join('x');
  document.getElementById('sensorConfig').textContent = size ? `${size} @ ${sc.bit_depth}-bit` : '—';

  const target = data.requested_fps;
  const sensorEl = document.getElementById('sensorFps');
  sensorEl.textContent = `${data.sensor_fps.toFixed(2)} / целевой ${target}`;
  sensorEl.className = (data.sensor_fps >= target - 1) ? 'value ok' : 'value warn';

  document.getElementById('deliveredFps').textContent = data.delivered_fps.toFixed(2);
  document.getElementById('frameCount').textContent = `${data.captured_frames} / ${data.dropped_frames}`;

  const px = data.center_pixel;
  if (px) {
    document.getElementById('centerPixel').textContent = `R=${px[0]}  G=${px[1]}  B=${px[2]}`;
    document.getElementById('swatch').style.background = `rgb(${px[0]},${px[1]},${px[2]})`;
  }
}
function calculateFov() {
  const distance = parseFloat(document.getElementById('distance').value);
  const width = parseFloat(document.getElementById('objectWidth').value);
  const angle = 2 * Math.atan(width / (2 * distance)) * 180 / Math.PI;
  document.getElementById('angle').textContent = Number.isFinite(angle) ? `${angle.toFixed(2)}°` : 'проверьте значения';
}
loadModes();
setInterval(updateStatus, 1000);
updateStatus();
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(HTML)


@app.get("/video_feed")
def video_feed():
    return Response(
        camera.mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/modes")
def modes():
    return jsonify(MODES)


@app.get("/api/status")
def status():
    return jsonify(camera.status())


@app.post("/api/configure")
def configure():
    data = request.get_json(silent=True) or {}
    try:
        camera.configure(str(data["mode"]), float(data["fps"]))
        return jsonify({"ok": True, "status": camera.status()})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


if __name__ == "__main__":
    print("Откройте в браузере: http://<IP_RASPBERRY_PI>:8080")
    app.run(host="0.0.0.0", port=8080, threaded=True)
