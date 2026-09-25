"""Localhost-only dashboard endpoint for simulation diagnostics/settings."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit


class DashboardHandler(BaseHTTPRequestHandler):
    settings_provider = staticmethod(lambda: {})
    settings_apply = None
    status_provider = staticmethod(
        lambda: {"phase": "SIMULATION", "hardware_verified": False}
    )
    cameras_provider = staticmethod(lambda: [])
    previews_provider = staticmethod(lambda: [])
    preview_files_provider = staticmethod(lambda: {})

    def do_GET(self):
        self.path = urlsplit(self.path).path
        if self.path == "/api/settings":
            self._json(self.settings_provider())
            return
        if self.path == "/api/status":
            self._json(self.status_provider())
            return
        if self.path == "/api/cameras":
            self._json(self.cameras_provider())
            return
        if self.path == "/api/previews":
            self._json(self.previews_provider())
            return
        if self.path.startswith("/preview/"):
            camera_id = self.path[len("/preview/") :]
            path = self.preview_files_provider().get(camera_id)
            if not path or not Path(path).is_file():
                self.send_error(404)
                return
            body = Path(path).read_bytes()
            if Path(path).suffix.lower() == ".ppm":
                import cv2
                import numpy as np

                decoded = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
                if decoded is None:
                    self.send_error(503)
                    return
                ok, encoded = cv2.imencode(
                    ".jpg", decoded, [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                if not ok:
                    self.send_error(503)
                    return
                body = encoded.tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/":
            slots = "".join(
                f"<figure><figcaption>camera_{i}</figcaption><img src='/preview/camera_{i}' alt='camera_{i} preview' width='320'><small id='camera_{i}-state'>metadata pending</small></figure>"
                for i in range(1, 7)
            )
            body = (
                Path(__file__)
                .with_name("dashboard.html")
                .read_text()
                .replace("{{CAMERA_SLOTS}}", slots)
                .encode()
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self):
        if self.path != "/api/settings" or self.settings_apply is None:
            self.send_error(404)
            return
        if self.headers.get("Content-Type") != "application/json":
            self.send_error(415)
            return
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            self.send_error(403)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1_000_000:
                raise ValueError("invalid request size")
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise ValueError("settings must be a JSON object")
            self._json(self.settings_apply(value))
        except (ValueError, KeyError, TypeError) as exc:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": False, "error": str(exc)}).encode())

    def _json(self, value):
        body = json.dumps(value, sort_keys=True).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def serve(host="127.0.0.1", port=8080):
    return ThreadingHTTPServer((host, port), DashboardHandler)
