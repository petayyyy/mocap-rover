"""Frames and windows from CM4/CM5 camera nodes over LAN.

``LanCameraSource`` keeps one TCP connection per node (``pi_cam/camera_node.py``)
and hands the localization pipeline what it asked the node for: the windows
of a sensor frame, or a full frame.  Each item is a ``LanFrame``; iterating
one yields the contract the tracking pipeline consumes::

    camera_id, array, stamp_ns, row0, col0, line_time_ns, exposure_ns, receive_ns

``stamp_ns`` is the exposure start of the first sensor row on the common
(PTP) time scale; a pixel at row ``r`` of the window was exposed around
``stamp_ns + (row0 + r) * line_time_ns + exposure_ns / 2``.  ``receive_ns`` is
the laptop's CLOCK_REALTIME when the last byte arrived, on the same scale, so
``receive_ns - stamp_ns`` is the exposure-to-laptop latency.

Delivery follows ``capture.LatestFrames``: one slot per camera holding the
newest complete frame (all its windows together, because they share one
``stamp_ns``), and a consumer that falls behind loses the older frame, which is
counted, never queued.

The wire format lives in ``pi_cam/lan_protocol.py`` and is shared with the node.
"""
from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass

import numpy as np

from pi_cam import lan_protocol as proto


def _jpeg_decoder():
    try:
        import simplejpeg

        def decode(data):
            return simplejpeg.decode_jpeg(bytes(data), colorspace="GRAY")[:, :, 0]
        return decode
    except ImportError:
        pass
    try:
        import cv2

        def decode(data):
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("JPEG decode failed")
            return image
        return decode
    except ImportError:
        return None


JPEG_DECODE = _jpeg_decoder()


@dataclass(slots=True)
class LanFrame:
    camera_id: str
    array: np.ndarray
    stamp_ns: int
    row0: int
    col0: int
    line_time_ns: int
    exposure_ns: int
    receive_ns: int
    receive_mono_ns: int = 0
    frame_seq: int = 0
    frame_duration_ns: int = 0
    format: str = "y8"
    window_index: int = 0
    window_count: int = 1
    sensor_width: int = 0
    sensor_height: int = 0
    request_id: int = 0
    node_send_ns: int = 0
    sensor_stamp_ns: int = 0
    clock_offset_ns: int = 0
    ptp_offset_ns: int | None = None
    # Payload bytes as they travelled: for JPEG this is the compressed size,
    # which is what the link actually carried, not the decoded pixel count.
    payload_bytes: int = 0

    def __iter__(self):
        """The eight-field source contract, in order."""
        return iter((self.camera_id, self.array, self.stamp_ns, self.row0, self.col0,
                     self.line_time_ns, self.exposure_ns, self.receive_ns))

    @property
    def height(self):
        return int(self.array.shape[0])

    @property
    def width(self):
        return int(self.array.shape[1])

    @property
    def is_full(self):
        return self.width == self.sensor_width and self.height == self.sensor_height

    def row_stamp_ns(self, row, exposure_centre=True):
        """Exposure time of window row ``row`` (0-based inside this window)."""
        stamp = self.stamp_ns + (self.row0 + row) * self.line_time_ns
        return stamp + self.exposure_ns // 2 if exposure_centre else stamp

    @property
    def latency_ns(self):
        return self.receive_ns - self.stamp_ns


def decode_frame(header: proto.FrameHeader, data, receive_ns, receive_mono_ns) -> LanFrame:
    if header.format == proto.FORMAT_Y8:
        expected = header.width * header.height
        if len(data) != expected:
            raise proto.ProtocolError(f"y8 payload {len(data)} != {expected}")
        array = np.frombuffer(data, dtype=np.uint8).reshape(header.height, header.width)
    elif header.format == proto.FORMAT_JPEG:
        if JPEG_DECODE is None:
            raise proto.ProtocolError("no JPEG decoder available")
        array = JPEG_DECODE(data)
        if array.shape != (header.height, header.width):
            raise proto.ProtocolError(f"jpeg decoded to {array.shape}, header says "
                                      f"{(header.height, header.width)}")
    else:
        raise proto.ProtocolError(f"unknown format {header.format}")
    return LanFrame(header.camera_id, array, header.stamp_ns, header.row0, header.col0,
                    header.line_time_ns, header.exposure_ns, receive_ns, receive_mono_ns,
                    header.frame_seq, header.frame_duration_ns, header.format_name,
                    header.window_index, header.window_count, header.sensor_width,
                    header.sensor_height, header.request_id, header.node_send_ns,
                    header.sensor_stamp_ns, header.clock_offset_ns, header.ptp_offset_ns,
                    len(data))


class LanNotConnected(ConnectionError):
    pass


def parse_node_address(node):
    if isinstance(node, dict):
        return str(node["host"]), int(node["port"])
    if isinstance(node, str):
        host, _, port = node.rpartition(":")
        return host or "127.0.0.1", int(port)
    host, port = node
    return str(host), int(port)


class NodeLink(threading.Thread):
    """One node: connect, reconnect, parse the stream, send commands."""

    def __init__(self, source, host, port, *, reconnect_delay=0.5, max_reconnect_delay=5.0):
        super().__init__(daemon=True, name=f"lan-{host}:{port}")
        self.source = source
        self.host, self.port = host, port
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.stop_event = threading.Event()
        self.sock = None
        self.send_lock = threading.Lock()
        self.connected = threading.Event()
        self.camera_id = None
        self.hello = None
        self.status = None
        self.status_mono_ns = None
        self.reconnects = 0
        self.parse_errors = 0
        self.bytes_received = 0
        self.messages = 0
        self.token = 0
        self.acks = {}
        self.ack_lock = threading.Lock()
        self.last_windows = None
        self.last_stream = None

    # ------------------------------------------------------------ sending

    def send_command(self, meta: dict, wait=True, timeout=2.0):
        sock = self.sock
        if sock is None or not self.connected.is_set():
            raise LanNotConnected(f"{self.host}:{self.port} is not connected")
        with self.ack_lock:
            self.token += 1
            token = self.token
            event = threading.Event()
            self.acks[token] = [event, None]
        meta = dict(meta, token=token)
        payload = proto.encode_json(proto.MSG_COMMAND, meta)
        try:
            with self.send_lock:
                sock.sendall(payload)
        except OSError as exc:
            with self.ack_lock:
                self.acks.pop(token, None)
            raise LanNotConnected(str(exc)) from exc
        if not wait:
            return token
        if not event.wait(timeout):
            with self.ack_lock:
                self.acks.pop(token, None)
            raise TimeoutError(f"no ack from {self.host}:{self.port} for {meta.get('cmd')}")
        with self.ack_lock:
            return self.acks.pop(token)[1]

    # ---------------------------------------------------------- receiving

    def run(self):
        delay = self.reconnect_delay
        while not self.stop_event.is_set():
            try:
                sock = socket.create_connection((self.host, self.port), timeout=2.0)
            except OSError:
                self.stop_event.wait(delay)
                delay = min(delay * 2, self.max_reconnect_delay)
                continue
            delay = self.reconnect_delay
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
            sock.settimeout(0.5)
            self.sock = sock
            self.connected.set()
            try:
                self._receive(sock)
            except (OSError, proto.ProtocolError, ValueError) as exc:
                if isinstance(exc, proto.ProtocolError):
                    self.parse_errors += 1
                self.source._link_error(self, str(exc))
            finally:
                self.connected.clear()
                self.sock = None
                try:
                    sock.close()
                except OSError:
                    pass
                if not self.stop_event.is_set():
                    self.reconnects += 1

    def _receive(self, sock):
        parser = proto.StreamParser()
        while not self.stop_event.is_set():
            try:
                chunk = sock.recv(1 << 20)
            except TimeoutError:
                continue
            if not chunk:
                raise ConnectionError("node closed the connection")
            receive_ns = time.clock_gettime_ns(time.CLOCK_REALTIME)
            receive_mono_ns = time.monotonic_ns()
            self.bytes_received += len(chunk)
            for msg in parser.feed(chunk):
                self.messages += 1
                self._dispatch(msg, receive_ns, receive_mono_ns)

    def _dispatch(self, msg, receive_ns, receive_mono_ns):
        if msg.msg_type == proto.MSG_FRAME:
            header = msg.frame
            try:
                frame = decode_frame(header, msg.data, receive_ns, receive_mono_ns)
            except (proto.ProtocolError, ValueError) as exc:
                self.parse_errors += 1
                self.source._link_error(self, f"frame: {exc}")
                return
            self.source._push_window(self, frame)
        elif msg.msg_type == proto.MSG_HELLO:
            self.hello = msg.meta
            self.camera_id = msg.meta.get("camera_id")
            self.source._register(self)
            self._resend_state()
        elif msg.msg_type == proto.MSG_STATUS:
            self.status = msg.meta
            self.status_mono_ns = receive_mono_ns
            self.source._push_status(self, msg.meta)
        elif msg.msg_type == proto.MSG_ACK:
            token = msg.meta.get("token")
            with self.ack_lock:
                entry = self.acks.get(token)
                if entry is not None:
                    entry[1] = msg.meta
                    entry[0].set()

    def _resend_state(self):
        """After a node restart its window list is gone: restore ours."""
        for meta in (self.last_windows, self.last_stream):
            if meta:
                try:
                    self.send_command(meta, wait=False)
                except LanNotConnected:
                    pass

    def close(self):
        self.stop_event.set()
        sock = self.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class LanCameraSource:
    """Latest complete frame per camera from N nodes, drops counted.

    ``nodes``: ``"host:port"`` strings, ``(host, port)`` tuples or dicts with
    ``host``/``port``.  Camera identity comes from the node's hello message,
    which carries the ``camera_id`` of its config -- never from the address.
    """

    def __init__(self, nodes, *, camera_ids=None, connect=True):
        self.condition = threading.Condition()
        self.links = [NodeLink(self, *parse_node_address(n)) for n in nodes]
        self.by_camera: dict[str, NodeLink] = {}
        self.expected = list(camera_ids) if camera_ids else None
        self.pending: dict[str, list] = {}
        self.partial: dict[str, list] = {}
        self.received = {}
        self.dropped = {}
        self.windows_received = {}
        self.statuses = {}
        self.errors = []
        self.closed = False
        if connect:
            for link in self.links:
                link.start()

    # --------------------------------------------------- link callbacks

    def _register(self, link):
        with self.condition:
            cid = link.camera_id
            self.by_camera[cid] = link
            self.received.setdefault(cid, 0)
            self.dropped.setdefault(cid, 0)
            self.windows_received.setdefault(cid, 0)
            self.condition.notify_all()

    def _link_error(self, link, message):
        with self.condition:
            self.errors.append((time.monotonic_ns(), f"{link.host}:{link.port}", message))
            del self.errors[:-50]

    def _push_status(self, link, status):
        with self.condition:
            self.statuses[status.get("camera_id", link.camera_id)] = status
            self.condition.notify_all()

    def _push_window(self, link, frame: LanFrame):
        cid = frame.camera_id
        with self.condition:
            if self.closed:
                return
            if cid not in self.by_camera:
                self.by_camera[cid] = link
                self.received.setdefault(cid, 0)
                self.dropped.setdefault(cid, 0)
                self.windows_received.setdefault(cid, 0)
            self.windows_received[cid] += 1
            group = self.partial.get(cid)
            if group and group[0].frame_seq != frame.frame_seq:
                # The node skipped a window of the previous frame: deliver what came.
                self._complete_locked(cid, group)
                group = None
            if group is None:
                group = self.partial[cid] = []
            group.append(frame)
            if len(group) >= frame.window_count or frame.window_index == frame.window_count - 1:
                self.partial.pop(cid, None)
                self._complete_locked(cid, group)

    def _complete_locked(self, cid, group):
        self.received[cid] += 1
        if cid in self.pending:
            self.dropped[cid] += 1
        self.pending[cid] = group
        self.condition.notify_all()

    # ------------------------------------------------------- consumer API

    @property
    def camera_ids(self):
        with self.condition:
            return sorted(self.by_camera)

    def wait_connected(self, timeout=5.0, camera_ids=None):
        """Block until every expected node has said hello; return the ids seen."""
        expected = camera_ids or self.expected
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                if expected is None:
                    ok = len(self.by_camera) >= len(self.links)
                else:
                    ok = set(expected) <= set(self.by_camera)
                remaining = deadline - time.monotonic()
                if ok or remaining <= 0:
                    return sorted(self.by_camera)
                self.condition.wait(remaining)

    def take(self, camera_id, timeout=0.2):
        """Newest complete frame of ``camera_id`` as a list of ``LanFrame`` windows."""
        with self.condition:
            self.condition.wait_for(lambda: camera_id in self.pending or self.closed, timeout)
            return self.pending.pop(camera_id, None)

    def take_any(self, timeout=0.2):
        """``(camera_id, windows)`` of any camera with a pending frame, or None."""
        with self.condition:
            self.condition.wait_for(lambda: bool(self.pending) or self.closed, timeout)
            if not self.pending:
                return None
            cid = next(iter(self.pending))
            return cid, self.pending.pop(cid)

    def status(self, camera_id):
        with self.condition:
            return self.statuses.get(camera_id)

    def hello(self, camera_id):
        link = self._link(camera_id)
        return link.hello

    def stats(self):
        with self.condition:
            return {
                "received": dict(self.received), "dropped": dict(self.dropped),
                "windows": dict(self.windows_received),
                "links": {f"{l.host}:{l.port}": {
                    "camera_id": l.camera_id, "connected": l.connected.is_set(),
                    "reconnects": l.reconnects, "parse_errors": l.parse_errors,
                    "bytes": l.bytes_received, "messages": l.messages} for l in self.links},
                "errors": list(self.errors),
            }

    # -------------------------------------------------------- commands

    def _link(self, camera_id) -> NodeLink:
        with self.condition:
            link = self.by_camera.get(camera_id)
        if link is None:
            raise LanNotConnected(f"no node has announced camera {camera_id!r}")
        return link

    def request_windows(self, camera_id, windows, wait=True, timeout=2.0):
        """Windows to cut from every following frame: ``(row0, col0, w, h[, format])``."""
        link = self._link(camera_id)
        normalized = []
        for w in windows:
            if isinstance(w, dict):
                normalized.append(w)
            else:
                row0, col0, width, height = (int(v) for v in w[:4])
                fmt = w[4] if len(w) > 4 else "y8"
                normalized.append({"row0": row0, "col0": col0, "w": width, "h": height, "format": fmt})
        meta = {"cmd": "set_windows", "windows": normalized}
        link.last_windows = meta
        return link.send_command(meta, wait=wait, timeout=timeout)

    def request_full(self, camera_id, fmt="y8", wait=True, timeout=2.0):
        """One full sensor frame (watchdog / acquisition)."""
        return self._link(camera_id).send_command({"cmd": "full_frame", "format": fmt},
                                                  wait=wait, timeout=timeout)

    def stream_full(self, camera_id, divisor, fmt="y8", wait=True, timeout=2.0):
        """Every ``divisor``-th full frame, continuously; 0 stops (dataset recording)."""
        link = self._link(camera_id)
        meta = {"cmd": "stream_full", "divisor": int(divisor), "format": fmt}
        link.last_stream = meta if divisor > 0 else None
        return link.send_command(meta, wait=wait, timeout=timeout)

    def configure(self, camera_id, exposure_us=None, gain=None, fps=None, wait=True, timeout=2.0):
        meta = {"cmd": "configure"}
        if exposure_us is not None:
            meta["exposure_us"] = int(exposure_us)
        if gain is not None:
            meta["gain"] = float(gain)
        if fps is not None:
            meta["fps"] = float(fps)
        return self._link(camera_id).send_command(meta, wait=wait, timeout=timeout)

    def request_status(self, camera_id, wait=True, timeout=2.0):
        return self._link(camera_id).send_command({"cmd": "status"}, wait=wait, timeout=timeout)

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        for link in self.links:
            link.close()
        for link in self.links:
            link.join(timeout=2.0)
