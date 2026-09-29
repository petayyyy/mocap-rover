"""Wire protocol between a camera node (CM4/CM5) and the laptop.

One TCP stream per node.  Every message is::

    MessageHeader (14 bytes, little-endian, fixed)
    type-specific fixed header (``header_len`` bytes; frames only)
    JSON metadata (``json_len`` bytes; may be empty)
    binary payload (``data_len`` bytes; image data for frames)

The frame header is a fixed ``struct`` so that a frame costs one
``struct.unpack`` to parse -- microseconds -- and the JSON field is reserved
for things that are rare or free-form (commands, status, acknowledgements).
Everything the localization pipeline needs from a frame is in the fixed
header; the JSON field of a frame is normally empty.

This module is imported by both ends and therefore depends on nothing but the
standard library.  Keep it that way.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass

PROTOCOL_VERSION = 1
MAGIC = 0x4D43  # "CM"

# Message types.
MSG_FRAME = 1     # node -> laptop: one window or one full frame
MSG_STATUS = 2    # node -> laptop: JSON status, once per second and on request
MSG_COMMAND = 3   # laptop -> node: JSON command
MSG_ACK = 4       # node -> laptop: JSON answer to a command
MSG_HELLO = 5     # node -> laptop: JSON, first message after connect

# Pixel formats carried by a frame.
FORMAT_Y8 = 0
FORMAT_JPEG = 1
FORMAT_NAMES = {FORMAT_Y8: "y8", FORMAT_JPEG: "jpeg"}
FORMAT_CODES = {name: code for code, name in FORMAT_NAMES.items()}

# magic u16, version u8, type u8, header_len u16, json_len u32, data_len u32
MESSAGE_HEADER = struct.Struct("<HBBHII")
MESSAGE_HEADER_SIZE = MESSAGE_HEADER.size  # 14

# The fixed frame header.  Field order is the order of FrameHeader below.
FRAME_HEADER = struct.Struct(
    "<"
    "16s"   # camera_id, NUL padded ASCII
    "Q"     # frame_seq: sensor frame counter on the node
    "q"     # stamp_ns: common time scale, start of exposure of the first sensor row
    "I"     # exposure_ns
    "I"     # line_time_ns: rolling-shutter row period
    "I"     # frame_duration_ns
    "H"     # row0 of this window in the full sensor frame
    "H"     # col0
    "H"     # width
    "H"     # height
    "B"     # format: FORMAT_Y8 or FORMAT_JPEG
    "B"     # window_index within the frame
    "B"     # window_count: how many messages this sensor frame produced
    "B"     # reserved
    "H"     # sensor_width
    "H"     # sensor_height
    "I"     # request_id: the set_windows / full_frame request being served
    "q"     # node_send_ns: node common-scale time just before send
    "q"     # sensor_stamp_ns: raw libcamera SensorTimestamp (CLOCK_BOOTTIME)
    "q"     # clock_offset_ns: node REALTIME - BOOTTIME at capture
    "q"     # ptp_offset_ns: last offsetFromMaster reported by pmc; INT64_MIN = unknown
)
FRAME_HEADER_SIZE = FRAME_HEADER.size
PTP_UNKNOWN = -(2 ** 63)

MAX_JSON_LEN = 1 << 20
MAX_DATA_LEN = 64 << 20


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class FrameHeader:
    camera_id: str
    frame_seq: int
    stamp_ns: int
    exposure_ns: int
    line_time_ns: int
    frame_duration_ns: int
    row0: int
    col0: int
    width: int
    height: int
    format: int
    window_index: int
    window_count: int
    sensor_width: int
    sensor_height: int
    request_id: int
    node_send_ns: int
    sensor_stamp_ns: int = 0
    clock_offset_ns: int = 0
    ptp_offset_ns: int | None = None

    @property
    def format_name(self) -> str:
        return FORMAT_NAMES.get(self.format, str(self.format))

    def pack(self) -> bytes:
        ptp = PTP_UNKNOWN if self.ptp_offset_ns is None else int(self.ptp_offset_ns)
        return FRAME_HEADER.pack(
            self.camera_id.encode("ascii")[:16], self.frame_seq, self.stamp_ns,
            self.exposure_ns, self.line_time_ns, self.frame_duration_ns,
            self.row0, self.col0, self.width, self.height, self.format,
            self.window_index, self.window_count, 0,
            self.sensor_width, self.sensor_height, self.request_id,
            self.node_send_ns, self.sensor_stamp_ns, self.clock_offset_ns, ptp)

    @classmethod
    def unpack(cls, raw: bytes | memoryview) -> "FrameHeader":
        if len(raw) < FRAME_HEADER_SIZE:
            raise ProtocolError("short frame header")
        (camera_id, seq, stamp, exposure, line_time, duration, row0, col0, width,
         height, fmt, window_index, window_count, _reserved, sw, sh, request_id,
         send_ns, sensor_stamp, clock_offset, ptp) = FRAME_HEADER.unpack_from(raw)
        return cls(camera_id.rstrip(b"\0").decode("ascii"), seq, stamp, exposure,
                   line_time, duration, row0, col0, width, height, fmt,
                   window_index, window_count, sw, sh, request_id, send_ns,
                   sensor_stamp, clock_offset, None if ptp == PTP_UNKNOWN else ptp)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["format"] = self.format_name
        return d


def encode_message(msg_type: int, fixed: bytes = b"", meta: dict | None = None,
                   data: bytes | memoryview = b"") -> list:
    """Return the message as a list of buffers for ``socket.sendmsg``/writelines.

    The payload is not copied: a numpy window's ``memoryview`` goes straight
    to the socket.
    """
    meta_bytes = b"" if not meta else json.dumps(meta, separators=(",", ":")).encode()
    header = MESSAGE_HEADER.pack(MAGIC, PROTOCOL_VERSION, msg_type, len(fixed),
                                 len(meta_bytes), len(data))
    parts = [header]
    if fixed:
        parts.append(fixed)
    if meta_bytes:
        parts.append(meta_bytes)
    if len(data):
        parts.append(data)
    return parts


def encode_frame(header: FrameHeader, data: bytes | memoryview, meta: dict | None = None) -> list:
    return encode_message(MSG_FRAME, header.pack(), meta, data)


def encode_json(msg_type: int, meta: dict) -> bytes:
    return b"".join(encode_message(msg_type, b"", meta, b""))


@dataclass
class Message:
    msg_type: int
    fixed: bytes
    meta: dict
    data: bytes | memoryview

    @property
    def frame(self) -> FrameHeader:
        if self.msg_type != MSG_FRAME:
            raise ProtocolError("not a frame message")
        return FrameHeader.unpack(self.fixed)


def parse_message_header(raw: bytes | memoryview):
    """Return (msg_type, header_len, json_len, data_len) or raise ProtocolError."""
    if len(raw) < MESSAGE_HEADER_SIZE:
        raise ProtocolError("short message header")
    magic, version, msg_type, header_len, json_len, data_len = MESSAGE_HEADER.unpack_from(raw)
    if magic != MAGIC:
        raise ProtocolError(f"bad magic 0x{magic:04x}")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version {version}")
    if json_len > MAX_JSON_LEN or data_len > MAX_DATA_LEN:
        raise ProtocolError("message too large")
    return msg_type, header_len, json_len, data_len


class StreamParser:
    """Incremental parser: feed bytes in, take complete messages out.

    Keeps at most one partial message buffered; the image payload of a complete
    message is returned as a ``memoryview`` over the internal buffer slice so
    that ``numpy.frombuffer`` needs no copy.
    """

    def __init__(self):
        self._buffer = bytearray()
        self.messages_parsed = 0

    def feed(self, chunk: bytes) -> list:
        self._buffer += chunk
        out = []
        view = memoryview(self._buffer)
        offset = 0
        total = len(view)
        while total - offset >= MESSAGE_HEADER_SIZE:
            msg_type, header_len, json_len, data_len = parse_message_header(view[offset:])
            length = MESSAGE_HEADER_SIZE + header_len + json_len + data_len
            if total - offset < length:
                break
            pos = offset + MESSAGE_HEADER_SIZE
            fixed = bytes(view[pos:pos + header_len])
            pos += header_len
            meta = json.loads(bytes(view[pos:pos + json_len])) if json_len else {}
            pos += json_len
            data = bytes(view[pos:pos + data_len]) if data_len else b""
            out.append(Message(msg_type, fixed, meta, data))
            offset += length
            self.messages_parsed += 1
        view.release()
        if offset:
            del self._buffer[:offset]
        return out


def read_exact(sock, n: int) -> bytes:
    """Blocking helper for simple clients: read exactly ``n`` bytes or raise."""
    chunks = []
    remaining = n
    while remaining:
        try:
            chunk = sock.recv(min(remaining, 1 << 20))
        except TimeoutError:
            if chunks:
                # Half a message is unrecoverable: the stream would be out of sync.
                raise ConnectionError("timeout inside a message") from None
            raise
        if not chunk:
            raise ConnectionError("connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_message(sock) -> Message:
    """Read one complete message from a blocking socket."""
    head = read_exact(sock, MESSAGE_HEADER_SIZE)
    msg_type, header_len, json_len, data_len = parse_message_header(head)
    fixed = read_exact(sock, header_len) if header_len else b""
    meta = json.loads(read_exact(sock, json_len)) if json_len else {}
    data = read_exact(sock, data_len) if data_len else b""
    return Message(msg_type, fixed, meta, data)


def normalize_window(window, sensor_width: int, sensor_height: int) -> dict:
    """Validate one window request, clamping it into the sensor frame.

    Accepts ``(row0, col0, w, h[, format])`` or a dict with those keys.
    """
    if isinstance(window, dict):
        row0, col0 = int(window["row0"]), int(window["col0"])
        width, height = int(window.get("w", window.get("width"))), int(window.get("h", window.get("height")))
        fmt = window.get("format", "y8")
    else:
        row0, col0, width, height = (int(v) for v in window[:4])
        fmt = window[4] if len(window) > 4 else "y8"
    if isinstance(fmt, int):
        fmt = FORMAT_NAMES[fmt]
    if fmt not in FORMAT_CODES:
        raise ProtocolError(f"unknown window format {fmt!r}")
    if width <= 0 or height <= 0:
        raise ProtocolError("window must have positive size")
    width, height = min(width, sensor_width), min(height, sensor_height)
    row0 = min(max(row0, 0), sensor_height - height)
    col0 = min(max(col0, 0), sensor_width - width)
    # Even offsets keep windows aligned to the 2x2 Bayer cell of the binned
    # mode, which keeps the window identical to the same pixels in a full frame.
    row0 -= row0 % 2
    col0 -= col0 % 2
    return {"row0": row0, "col0": col0, "w": width, "h": height, "format": fmt}
