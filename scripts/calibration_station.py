#!/usr/bin/env python3
"""Calibration station: live 1640x1232 frames from all camera nodes and a web page.

    .venv/bin/python scripts/calibration_station.py --nodes 192.168.10.101:5600 ...
    .venv/bin/python scripts/calibration_station.py --demo        # no hardware

Open http://localhost:8090.  Two tabs:

* "Взаимное положение" -- the floor strip in every camera, snapshots of each
  strip placement, the joint solve of the six camera poses ->
  ``<session>/extrinsics/runtime_cameras.json`` + ``calibration_report.json``.
* "Внутренние параметры" -- one chosen camera, a hand-held ChArUco board,
  ~35 captures (manual or automatic), three fisheye calibrations
  (1640x1232, 820x616, 640x480) -> ``<session>/intrinsics/<camera>/``.

Every node streams full frames as grey JPEG on a timer: 20 Hz, or 10 Hz when
the link is loaded (``--rate auto``), each node shifted by period / N on the
common time scale so the frames reach the switch one after another instead of
in one burst.  Nothing is deleted; a session directory is only ever added to.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import logging
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from localization_contracts import charuco_calibration as cc  # noqa: E402
from localization_contracts import lan_capture  # noqa: E402

log = logging.getLogger("calibration_station")
HTML = Path(__file__).with_name("calibration_station.html")
DEFAULT_NODES = [f"192.168.10.{100 + i}:5600" for i in range(1, 7)]
DEFAULT_BASE = ROOT / "config" / "mocap_arena_imx219" / "runtime_cameras.json"
BACKGROUND_HZ = 2.0            # other cameras while one is being calibrated
NS = 1_000_000_000


def now_s():
    return time.monotonic()


class CameraState:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.frames = collections.deque(maxlen=4)      # (seq, gray, stamp_ns, receive_ns)
        self.seq = 0
        self.arrivals = collections.deque(maxlen=200)  # (monotonic s, payload bytes)
        self.latency_ms = collections.deque(maxlen=50)
        self.strip = None                              # (Detection, monotonic s, ms)
        self.hand = None
        self.hz = 0.0
        self.phase_ns = 0
        self.jpeg_cache = {}

    def latest(self):
        return self.frames[-1] if self.frames else None

    def fps(self, window=3.0):
        t = now_s()
        recent = [a for a in self.arrivals if t - a[0] <= window]
        if len(recent) < 2:
            return 0.0
        span = max(recent[-1][0] - recent[0][0], 1e-3)
        return (len(recent) - 1) / span

    def mbps(self, window=3.0):
        t = now_s()
        recent = [a for a in self.arrivals if t - a[0] <= window]
        return sum(b for _, b in recent) * 8 / 1e6 / window


class Station:
    def __init__(self, args, demo=None):
        self.args = args
        self.demo = demo
        self.lock = threading.RLock()
        self.messages = collections.deque(maxlen=30)
        self.session = Path(args.session) if args.session else (
            ROOT / "artifacts" / "calibration" / time.strftime("%Y%m%d_%H%M%S"))
        (self.session / "extrinsics").mkdir(parents=True, exist_ok=True)
        (self.session / "intrinsics").mkdir(parents=True, exist_ok=True)
        self.base_config = json.loads(Path(args.base_config).read_text())
        self.strip = cc.strip_target()
        self.hand = hand_target(args)
        self.cams: dict[str, CameraState] = {}
        self.mode = args.tab
        self.rate_mode = args.rate
        self.rate_hz = 20.0 if args.rate in ("auto", "20") else 10.0
        self.rate_reason = "старт"
        self.rate_strikes = 0
        self.load = 0.0
        self.node_drops = {}
        self.placements = self._load_manifest()
        self.solving = False
        self.solution = None
        self.selected = args.camera
        self.auto_capture = False
        self.captures = {}                 # camera -> list of (path, Detection)
        self.last_capture_t = 0.0
        self.calibrating = False
        self.calib_progress = ""
        self.intrinsic_results = {}
        self._load_intrinsic_captures()
        self.stop_event = threading.Event()
        nodes = demo.addresses if demo else args.nodes
        self.source = lan_capture.LanCameraSource(nodes, bayer_demosaic=True)
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.say(f"сессия {self.session}")

    # ------------------------------------------------------------ helpers

    def say(self, text):
        log.info(text)
        with self.lock:
            self.messages.append({"t": time.strftime("%H:%M:%S"), "text": text})

    def camera_order(self):
        with self.lock:
            return sorted(self.cams, key=lambda c: (len(c), c))

    def _load_manifest(self):
        path = self.session / "extrinsics" / "manifest.json"
        if path.is_file():
            return json.loads(path.read_text())["placements"]
        return []

    def _save_manifest(self):
        path = self.session / "extrinsics" / "manifest.json"
        path.write_text(json.dumps({"strip": self.strip.to_dict(), "placements": self.placements},
                                   indent=2))

    def _load_intrinsic_captures(self):
        for cam_dir in sorted((self.session / "intrinsics").glob("*/captures")):
            camera = cam_dir.parent.name
            items = []
            for png in sorted(cam_dir.glob("*.png")):
                gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
                items.append((png, self.hand.detect(gray)))
            self.captures[camera] = items

    # --------------------------------------------------------------- loops

    def start(self):
        for target in (self.receive_loop, self.control_loop, self.strip_loop, self.hand_loop):
            threading.Thread(target=target, daemon=True, name=target.__name__).start()

    def receive_loop(self):
        while not self.stop_event.is_set():
            got = self.source.take_any(0.2)
            if got is None:
                continue
            cid, group = got
            for frame in group:
                if not frame.is_full:
                    continue
                with self.lock:
                    cam = self.cams.get(cid)
                    if cam is None:
                        continue
                    cam.seq += 1
                    cam.frames.append((cam.seq, frame.array, frame.stamp_ns, frame.receive_ns))
                    cam.arrivals.append((now_s(), frame.payload_bytes))
                    cam.latency_ms.append((frame.receive_ns - frame.stamp_ns) / 1e6)

    def desired_rates(self):
        cams = self.camera_order()
        out = {}
        for k, cid in enumerate(cams):
            hz = self.rate_hz
            if self.mode == "intrinsic" and self.selected and cid != self.selected:
                hz = min(BACKGROUND_HZ, hz)
            period = NS / self.rate_hz
            out[cid] = (hz, int(k * period / max(len(cams), 1)))
        return out

    def apply_rates(self):
        for cid, (hz, phase) in self.desired_rates().items():
            cam = self.cams[cid]
            if (cam.hz, cam.phase_ns) == (hz, phase):
                continue
            try:
                ack = self.source.stream_full(cid, 0, "jpeg", hz=hz, phase_ns=phase, gray=True)
                if ack and ack.get("ok") and "hz" not in ack:
                    # A node from before the timed stream reads this as "divisor 0":
                    # it stops instead of streaming.  Say so once and leave it.
                    cam.hz, cam.phase_ns = hz, phase
                    self.say(f"{cid}: узел не знает поток по таймеру -- обновите pi_cam на узле "
                             f"(scripts/hardware/setup_cm4_node.sh N, затем restart camera_node)")
                elif ack and ack.get("ok"):
                    cam.hz, cam.phase_ns = hz, phase
                else:
                    self.say(f"{cid}: узел отказал: {ack}")
            except (lan_capture.LanNotConnected, TimeoutError, OSError) as exc:
                self.say(f"{cid}: команда не прошла: {exc}")

    def control_loop(self):
        """Registers cameras, keeps every node streaming, adapts 20 / 10 Hz."""
        warm_until = now_s() + 4.0
        last_dropped = {}
        while not self.stop_event.is_set():
            for cid in self.source.camera_ids:
                with self.lock:
                    if cid not in self.cams:
                        self.cams[cid] = CameraState(cid)
                        warm_until = now_s() + 4.0
                        self.say(f"{cid}: подключена")
            self.apply_rates()
            for cid in list(self.cams):
                try:
                    self.source.request_status(cid, wait=False)
                except lan_capture.LanNotConnected:
                    pass
            time.sleep(1.0)
            total = sum(c.mbps() for c in self.cams.values())
            self.load = total / self.args.link_mbit
            stats = self.source.stats()
            problems = []
            for cid, cam in self.cams.items():
                status = self.source.status(cid) or {}
                dropped = status.get("frames_dropped_queue", 0) + stats["dropped"].get(cid, 0)
                # A frame lost now and then is noise; a tenth of the rate is a jammed link.
                lost = dropped - last_dropped.get(cid, dropped)
                if lost > max(1.0, 0.1 * cam.hz):
                    problems.append(f"{cid}: {lost} кадров сброшено за 1 с")
                last_dropped[cid] = dropped
                self.node_drops[cid] = dropped
                if now_s() > warm_until and cam.hz >= 1 and cam.fps() < 0.75 * cam.hz:
                    problems.append(f"{cid}: {cam.fps():.1f} к/с из {cam.hz:g}")
            if self.load > self.args.max_load:
                problems.append(f"канал загружен на {self.load:.0%}")
            if self.rate_mode == "auto" and now_s() > warm_until:
                if problems and self.rate_hz > 10:
                    self.rate_strikes += 1
                    if self.rate_strikes >= 2:
                        self.rate_hz = 10.0
                        self.rate_reason = "; ".join(problems[:3])
                        self.say(f"частота снижена до 10 Гц: {self.rate_reason}")
                        warm_until = now_s() + 4.0
                        self.rate_strikes = 0
                else:
                    self.rate_strikes = 0
                if problems and self.rate_hz <= 10:
                    self.rate_reason = "и при 10 Гц: " + "; ".join(problems[:3])

    def strip_loop(self):
        while not self.stop_event.is_set():
            if self.mode != "extrinsic":
                time.sleep(0.2)
                continue
            jobs = []
            with self.lock:
                for cam in self.cams.values():
                    latest = cam.latest()
                    if latest is None:
                        continue
                    if cam.strip and now_s() - cam.strip[1] < 0.8:
                        continue
                    jobs.append((cam, latest[1]))
            futures = [self.pool.submit(self._detect_strip, cam, gray) for cam, gray in jobs]
            concurrent.futures.wait(futures)
            time.sleep(0.1)

    def _detect_strip(self, cam, gray):
        begin = time.perf_counter()
        det = self.strip.detect(gray)
        cam.strip = (det, now_s(), (time.perf_counter() - begin) * 1000)

    def hand_loop(self):
        last_seq = None
        while not self.stop_event.is_set():
            cid = self.selected
            if self.mode != "intrinsic" or cid not in self.cams:
                time.sleep(0.1)
                continue
            cam = self.cams[cid]
            latest = cam.latest()
            if latest is None or latest[0] == last_seq:
                time.sleep(0.01)
                continue
            last_seq = latest[0]
            begin = time.perf_counter()
            det = self.hand.detect(latest[1])
            previous = cam.hand
            cam.hand = (det, now_s(), (time.perf_counter() - begin) * 1000)
            if self.auto_capture:
                self._maybe_auto_capture(cid, latest[1], det, previous)

    # ------------------------------------------------------- intrinsics

    @staticmethod
    def _mean_shift(a, b):
        common, ia, ib = np.intersect1d(a.ids, b.ids, return_indices=True)
        if len(common) < 6:
            return None
        return float(np.linalg.norm(a.points[ia] - b.points[ib], axis=1).mean())

    def _maybe_auto_capture(self, cid, gray, det, previous):
        if det.count < self.hand.min_corners or now_s() - self.last_capture_t < self.args.auto_interval:
            return
        if previous is None or now_s() - previous[1] > 0.6:
            return
        still = self._mean_shift(det, previous[0])
        if still is None or still > self.args.still_px:
            return
        for _, old in self.captures.get(cid, []):
            shift = self._mean_shift(det, old)
            if shift is not None and shift < self.args.novelty_px:
                return
        self.capture_intrinsic(cid, gray, det, auto=True)
        if len(self.captures.get(cid, [])) >= cc.RECOMMENDED_VIEWS:
            self.auto_capture = False
            self.say(f"{cid}: набрано {cc.RECOMMENDED_VIEWS} кадров, автозахват выключен")

    def capture_intrinsic(self, cid, gray=None, det=None, auto=False):
        cam = self.cams.get(cid)
        if gray is None:
            latest = cam.latest() if cam else None
            if latest is None:
                return {"ok": False, "error": "нет кадров"}
            gray = latest[1]
            det = self.hand.detect(gray)
        if det.count < self.hand.min_corners:
            return {"ok": False, "error": f"доска не найдена ({det.count} углов)"}
        folder = self.session / "intrinsics" / cid / "captures"
        folder.mkdir(parents=True, exist_ok=True)
        with self.lock:
            items = self.captures.setdefault(cid, [])
            index = max([int(p.stem) for p, _ in items] + [-1]) + 1
            path = folder / f"{index:03d}.png"
            cv2.imwrite(str(path), gray)
            items.append((path, det))
            self.last_capture_t = now_s()
        self.say(f"{cid}: кадр {len(items)} ({det.count} углов{', авто' if auto else ''})")
        if self.demo:
            self.demo.next_hand_view()
        return {"ok": True, "count": len(items), "corners": det.count}

    def undo_intrinsic(self, cid):
        with self.lock:
            items = self.captures.get(cid, [])
            if not items:
                return {"ok": False, "error": "нечего убирать"}
            path, _ = items.pop()
        rejected = path.parent.parent / "rejected"
        rejected.mkdir(exist_ok=True)
        path.rename(rejected / path.name)          # kept on disk, out of the set
        return {"ok": True, "count": len(items)}

    def calibrate_intrinsic(self, cid):
        if self.calibrating:
            return {"ok": False, "error": "калибровка уже идёт"}
        items = list(self.captures.get(cid, []))
        if len(items) < 5:
            return {"ok": False, "error": f"мало кадров: {len(items)}"}
        self.calibrating = True

        def run():
            try:
                images = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p, _ in items]
                results = cc.calibrate_intrinsics(
                    images, self.hand, max_frame_err=self.args.max_frame_err,
                    progress=lambda text: setattr(self, "calib_progress", text))
                out = self.session / "intrinsics" / cid
                cc.write_intrinsics(out, cid, results, self.hand, sensor_name=self.args.sensor_name)
                self.intrinsic_results[cid] = [self._result_view(r) for r in results]
                summary = ", ".join(f"{r['size'][0]}x{r['size'][1]}: "
                                    + (f"{r['reprojection_error']:.3f} px" if "K" in r else r["error"])
                                    for r in results)
                self.say(f"{cid}: калибровка готова ({summary}) -> {out}")
            except Exception as exc:     # shown on the page, the station keeps running
                log.exception("intrinsic calibration")
                self.say(f"{cid}: калибровка не удалась: {exc}")
            finally:
                self.calibrating = False
                self.calib_progress = ""

        threading.Thread(target=run, daemon=True).start()
        return {"ok": True}

    @staticmethod
    def _result_view(r):
        if "K" not in r:
            return {"size": r["size"], "error": r.get("error"), "views_detected": r["views_detected"]}
        k = np.asarray(r["K"])
        return {"size": r["size"], "rms": r["rms"], "reprojection_error": r["reprojection_error"],
                "fx": k[0, 0], "fy": k[1, 1], "cx": k[0, 2], "cy": k[1, 2],
                "D": np.asarray(r["D"]).tolist(), "views_used": len(r["views_used"]),
                "views_detected": r["views_detected"], "views_captured": r["views_captured"],
                "dropped_outliers": len(r["dropped_outliers"]),
                "dropped_illcond": len(r["dropped_illcond"]), "fov_deg": r["fov_deg"],
                "scaled_check": r.get("K_scaled_from")}

    def coverage(self, cid, cols=16, rows=12):
        grid = np.zeros((rows, cols), int)
        w, h = cc.SENSOR_SIZE
        for _, det in self.captures.get(cid, []):
            if det.count == 0:
                continue
            c = np.clip((det.points[:, 0] / w * cols).astype(int), 0, cols - 1)
            r = np.clip((det.points[:, 1] / h * rows).astype(int), 0, rows - 1)
            np.add.at(grid, (r, c), 1)
        return grid.tolist()

    # ------------------------------------------------------- extrinsics

    def capture_placement(self, x0, y0, yaw_deg, known, replace=None):
        snapshot = {}
        with self.lock:
            for cid, cam in self.cams.items():
                fresh = [f for f in cam.frames if f[3] and (time.time_ns() - f[3]) < 3 * NS]
                if fresh:
                    stack = np.mean([f[1].astype(np.float32) for f in fresh], axis=0)
                    snapshot[cid] = np.clip(stack + 0.5, 0, 255).astype(np.uint8)
        if not snapshot:
            return {"ok": False, "error": "нет свежих кадров ни с одной камеры"}
        index = len(self.placements) if replace is None else int(replace)
        folder = self.session / "extrinsics" / f"placement_{index}"
        folder.mkdir(parents=True, exist_ok=True)
        entry = {"index": index, "x0": float(x0), "y0": float(y0), "yaw_deg": float(yaw_deg),
                 "known": bool(known), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "cameras": {}}
        for cid, gray in snapshot.items():
            path = folder / f"{cid}.png"
            cv2.imwrite(str(path), gray)
            det = self.strip.detect(gray)
            entry["cameras"][cid] = {"image": str(path.relative_to(self.session)),
                                     "corners": det.count, "markers": det.markers}
        with self.lock:
            if replace is None:
                self.placements.append(entry)
            else:
                self.placements[index] = entry
            self._save_manifest()
        counts = ", ".join(f"{c}: {v['corners']}" for c, v in sorted(entry["cameras"].items()))
        self.say(f"положение {index} снято ({counts})")
        if self.demo:
            self.demo.next_placement()
        return {"ok": True, "placement": entry}

    def delete_placement(self, index):
        with self.lock:
            if not 0 <= index < len(self.placements):
                return {"ok": False, "error": "нет такого положения"}
            self.placements[index]["deleted"] = True
            self._save_manifest()
        return {"ok": True}

    def intrinsics_for_solve(self):
        out, sources = {}, {}
        for cam in self.base_config["cameras"]:
            name = cam["name"]
            found = cc.load_intrinsics(self.session / "intrinsics", name)
            if found is None and self.args.intrinsics:
                found = cc.load_intrinsics(self.args.intrinsics, name)
                src = str(self.args.intrinsics)
            else:
                src = "эта сессия"
            if found is None and self.demo:
                found, src = self.demo.strip_intrinsics(), "демо: объектив демо-мира"
            if found is None:
                found = (np.asarray(cam["K"], float).reshape(3, 3), np.asarray(cam["D"], float))
                src = "базовый конфиг (НЕ калибровка объектива)"
            out[name], sources[name] = found, src
        return out, sources

    def solve(self):
        if self.solving:
            return {"ok": False, "error": "уже решается"}
        self.solving = True

        def run():
            try:
                self.solution = cc.solve_session(self.session, self.base_config, self.strip,
                                              *self.intrinsics_for_solve(),
                                              truth=self.demo.truth if self.demo else None)
                sol = self.solution
                self.say(f"решено: {len(sol['cameras'])} камер, ошибка P95 {sol['p95_px']:.2f} px"
                         + (f", не решены: {', '.join(sol['unsolved'])}" if sol["unsolved"] else ""))
            except Exception as exc:
                log.exception("extrinsic solve")
                self.solution = {"error": str(exc)}
                self.say(f"решение не удалось: {exc}")
            finally:
                self.solving = False

        threading.Thread(target=run, daemon=True).start()
        return {"ok": True}

    # ------------------------------------------------------------ state

    def state(self):
        cams = []
        for cid in self.camera_order():
            cam = self.cams[cid]
            status = self.source.status(cid) or {}
            latest = cam.latest()
            strip = cam.strip
            ptp = (status.get("ptp") or {}).get("offset_ns")
            cams.append({
                "id": cid, "hz": cam.hz, "fps": round(cam.fps(), 1), "mbps": round(cam.mbps(), 1),
                "latency_ms": round(float(np.median(cam.latency_ms)), 1) if cam.latency_ms else None,
                "age_s": round((time.time_ns() - latest[3]) / NS, 2) if latest else None,
                "seq": latest[0] if latest else 0, "phase_ms": round(cam.phase_ns / 1e6, 1),
                "strip": None if not strip else {"corners": strip[0].count, "markers": strip[0].markers,
                                                 "ms": round(strip[2])},
                "node": {"temp_c": status.get("soc_temp_c"), "cpu": status.get("cpu_percent"),
                         "sensor_fps": status.get("sensor_fps"), "dropped": self.node_drops.get(cid, 0),
                         "ptp_offset_us": None if ptp is None else round(ptp / 1000, 1),
                         "throttled": status.get("throttled")}})
        sel = self.selected
        hand = self.cams[sel].hand if sel in self.cams else None
        return {
            "session": str(self.session), "mode": self.mode, "demo": bool(self.demo),
            "rate": {"mode": self.rate_mode, "hz": self.rate_hz, "reason": self.rate_reason,
                     "load": round(self.load, 3), "link_mbit": self.args.link_mbit,
                     "max_load": self.args.max_load},
            "cameras": cams,
            "extrinsic": {"placements": [p for p in self.placements if not p.get("deleted")],
                          "solving": self.solving, "solution": self.solution,
                          "suggested": self.demo.suggested_placement() if self.demo else None,
                          "strip": self.strip.describe()},
            "intrinsic": {"camera": sel, "auto": self.auto_capture, "recommended": cc.RECOMMENDED_VIEWS,
                          "captured": len(self.captures.get(sel, [])) if sel else 0,
                          "coverage": self.coverage(sel) if sel else None,
                          "corners_now": hand[0].count if hand else 0,
                          "corners_total": self.hand.corner_count,
                          "board": self.hand.describe(), "calibrating": self.calibrating,
                          "progress": self.calib_progress,
                          "results": self.intrinsic_results.get(sel),
                          "output": str(self.session / "intrinsics" / sel) if sel else None,
                          "sizes": [list(s) for s in cc.INTRINSIC_SIZES]},
            "messages": list(self.messages)}

    def frame_jpeg(self, cid, width, overlay):
        cam = self.cams.get(cid)
        if cam is None:
            return None
        latest = cam.latest()
        if latest is None:
            return None
        key = (latest[0], width, overlay)
        cached = cam.jpeg_cache.get(key)
        if cached is not None:
            return cached
        gray = latest[1]
        scale = width / gray.shape[1]
        small = cv2.resize(gray, (width, int(round(gray.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        det = None
        if overlay == "strip" and cam.strip:
            det = cam.strip[0]
        elif overlay == "hand" and cam.hand:
            det = cam.hand[0]
        if det is not None and det.count:
            radius = max(2, int(round(3 * scale * 2)))
            for x, y in det.points * scale:
                cv2.circle(bgr, (int(round(x)), int(round(y))), radius, (60, 220, 90), -1, cv2.LINE_AA)
        ok, data = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        data = data.tobytes() if ok else None
        cam.jpeg_cache = {key: data}
        return data

    def close(self):
        self.stop_event.set()
        for cid in list(self.cams):
            try:
                self.source.stream_full(cid, 0, wait=False)
            except lan_capture.LanNotConnected:
                pass
        time.sleep(0.2)
        self.source.close()
        self.pool.shutdown(wait=False)


# ----------------------------------------------------------------- HTTP

def make_handler(station: Station):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
            if isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            if url.path in ("/", "/index.html"):
                return self._send(200, HTML.read_text(encoding="utf-8"), "text/html; charset=utf-8")
            if url.path == "/api/state":
                return self._send(200, station.state())
            if url.path.startswith("/frame/") and url.path.endswith(".jpg"):
                cid = url.path[len("/frame/"):-4]
                width = int(query.get("w", ["640"])[0])
                width = max(160, min(width, cc.SENSOR_SIZE[0]))
                data = station.frame_jpeg(cid, width, query.get("overlay", ["none"])[0])
                if data is None:
                    return self._send(404, {"error": "нет кадра"})
                return self._send(200, data, "image/jpeg")
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}") if length else {}
            path = urlsplit(self.path).path
            try:
                result = self.route(path, body)
            except Exception as exc:          # report, keep serving
                log.exception("POST %s", path)
                result = {"ok": False, "error": str(exc)}
            return self._send(200 if result is not None else 404, result or {"error": "not found"})

        def route(self, path, body):
            s = station
            if path == "/api/mode":
                s.mode = "intrinsic" if body.get("mode") == "intrinsic" else "extrinsic"
                if s.demo:
                    s.demo.set_mode(s.mode, s.selected)
                return {"ok": True}
            if path == "/api/rate":
                mode = str(body.get("mode", "auto"))
                s.rate_mode = mode if mode in ("auto", "20", "10") else "auto"
                s.rate_hz = 10.0 if s.rate_mode == "10" else 20.0
                s.rate_reason = "выбрано вручную" if s.rate_mode != "auto" else "авто: снова с 20 Гц"
                return {"ok": True}
            if path == "/api/extrinsic/capture":
                return s.capture_placement(body.get("x0", 0), body.get("y0", 0), body.get("yaw_deg", 0),
                                           body.get("known", False), body.get("replace"))
            if path == "/api/extrinsic/delete":
                return s.delete_placement(int(body["index"]))
            if path == "/api/extrinsic/solve":
                return s.solve()
            if path == "/api/intrinsic/select":
                s.selected = body.get("camera")
                s.auto_capture = False
                if s.demo:
                    s.demo.set_mode(s.mode, s.selected)
                return {"ok": True}
            if path == "/api/intrinsic/capture":
                return s.capture_intrinsic(s.selected) if s.selected else {"ok": False, "error": "камера не выбрана"}
            if path == "/api/intrinsic/auto":
                s.auto_capture = bool(body.get("enable"))
                return {"ok": True}
            if path == "/api/intrinsic/undo":
                return s.undo_intrinsic(s.selected)
            if path == "/api/intrinsic/calibrate":
                return s.calibrate_intrinsic(s.selected)
            return None

    return Handler


def add_board_args(p):
    """The hand-held board; defaults: the A4 board of sverk-ros2 camera_calibration."""
    p.add_argument("--board-squares", type=int, nargs=2, default=(11, 8), metavar=("X", "Y"))
    p.add_argument("--board-square-mm", type=float, default=22.0)
    p.add_argument("--board-marker-mm", type=float, default=16.0)
    p.add_argument("--board-dict", default="DICT_4X4_50")
    p.add_argument("--board-layout", choices=("auto", "legacy", "new"), default="auto",
                   help="ChArUco corner layout: legacy for calib.io / OpenCV < 4.6 boards")
    p.add_argument("--min-corners", type=int, default=12)
    p.add_argument("--max-frame-err", type=float, default=2.0,
                   help="drop views above this mean reprojection error (px), as sverk")


def hand_target(args):
    return cc.CharucoTarget(tuple(args.board_squares), args.board_square_mm / 1000.0,
                            args.board_marker_mm / 1000.0, args.board_dict,
                            legacy={"auto": "auto", "legacy": True, "new": False}[args.board_layout],
                            min_corners=args.min_corners)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nodes", nargs="+", default=DEFAULT_NODES, help="host:port of every camera node")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--session", help="continue this session directory (default: a new one in "
                                     "artifacts/calibration/)")
    p.add_argument("--base-config", default=str(DEFAULT_BASE),
                   help="runtime_cameras.json whose cameras get the solved poses")
    p.add_argument("--intrinsics", help="K, D of the cameras when this session has none: a "
                                        "directory of <camera>_1640x1232.json or a runtime_cameras.json")
    p.add_argument("--rate", choices=("auto", "20", "10"), default="auto",
                   help="full frames per second per camera; auto starts at 20 and drops to 10 "
                        "when the link is loaded or frames are lost")
    p.add_argument("--link-mbit", type=float, default=940.0, help="usable link capacity")
    p.add_argument("--max-load", type=float, default=0.7, help="auto: drop to 10 Hz above this share")
    add_board_args(p)
    p.add_argument("--auto-interval", type=float, default=0.8, help="s between automatic captures")
    p.add_argument("--still-px", type=float, default=1.5, help="board must move less than this")
    p.add_argument("--novelty-px", type=float, default=40.0,
                   help="a new capture must differ from every earlier one by this much")
    p.add_argument("--sensor-name", default="imx219", help="YAML file name prefix")
    p.add_argument("--tab", choices=("extrinsic", "intrinsic"), default="extrinsic",
                   help="page opens on this tab")
    p.add_argument("--camera", help="intrinsic tab: start with this camera selected")
    p.add_argument("--demo", action="store_true",
                   help="six synthetic camera nodes in this process: rendered strip and hand board")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    demo = None
    if args.demo:
        from tools.calibration_demo import DemoWorld
        demo = DemoWorld(Path(args.base_config))
        demo.start()
        demo.set_mode(args.tab, args.camera)
        args.base_config = str(demo.base_config_path)
    station = Station(args, demo)
    station.start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(station))
    print(f"calibration station: http://localhost:{args.port}  session {station.session}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        station.close()
        if demo:
            demo.stop()


if __name__ == "__main__":
    main()
