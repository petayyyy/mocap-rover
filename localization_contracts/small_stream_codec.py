"""The node's small stream (or its full frame) through a codec, emulated offline (replay only).

The small stream is the whole frame reduced on the camera node (e.g.
640x480 at 30 Hz) that the opponent path and SAM2 read.  ``y8`` sends the
grey plane as is; ``jpeg`` sends each frame as a grey JPEG; ``h264`` sends
one H.264 stream per camera, so every small frame of that camera goes
through the same encoder in order (inter-frame prediction is the point).

The H.264 settings follow the CM4 hardware encoder as ``pi_cam/h264_probe.py``
drives it: grey enters as the Y plane of YUV420 with the chroma at 128,
profile high, no B-frames, nothing held back (x264 ``zerolatency``), one
slice, a keyframe every ``round(rate * gop_s)`` frames, a VBR target with a
1.5x ceiling.  x264 is not the VideoCore encoder: the rate control and the
quality at a given bitrate differ; this is the closest software stand-in.

``SmallStreamCodec.roundtrip(gray)`` returns (decoded grey, encoded bytes,
decode ms); ``roundtrip_color(rgb)`` sends a real colour YUV420 frame through
the same encoder and returns the decoded Y plane and RGB.  The decoded frame is what the laptop sees; the bytes are what
goes on the wire.
"""
from __future__ import annotations

import time

import numpy as np

CODECS = ("y8", "jpeg", "h264")


class SmallStreamCodec:
    def __init__(self, codec, width, height, rate_hz, bitrate_mbit=4.0, gop_s=0.2,
                 jpeg_quality=90):
        if codec not in CODECS:
            raise ValueError(f"unknown small-stream codec {codec!r}")
        self.codec = codec
        self.width, self.height = int(width), int(height)
        self.rate_hz = float(rate_hz)
        self.bitrate = float(bitrate_mbit) * 1e6
        self.gop = max(1, int(round(self.rate_hz * float(gop_s))))
        self.jpeg_quality = int(jpeg_quality)
        self.frames = 0
        self.sizes = []               # encoded bytes per frame, in order
        self.keyframes = 0
        self._enc = self._dec = None
        self._chroma = None
        if codec == "h264":
            self._open_h264()

    # ------------------------------------------------------------------ h264
    def _open_h264(self):
        import av
        from fractions import Fraction
        rate = Fraction(self.rate_hz).limit_denominator(1000)
        enc = av.CodecContext.create("libx264", "w")
        enc.width, enc.height = self.width, self.height
        enc.pix_fmt = "yuv420p"
        enc.time_base = 1 / rate
        enc.framerate = rate
        enc.gop_size = self.gop
        enc.max_b_frames = 0
        enc.bit_rate = int(self.bitrate)
        kbit = max(1, int(self.bitrate / 1000))
        enc.options = {
            "preset": "veryfast", "tune": "zerolatency", "profile": "high",
            "x264-params": (f"keyint={self.gop}:min-keyint={self.gop}:scenecut=0:bframes=0:"
                            f"slices=1:sliced-threads=0:threads=1:rc-lookahead=0:"
                            f"vbv-maxrate={int(kbit * 1.5)}:vbv-bufsize={int(kbit * 1.5)}:"
                            f"repeat-headers=1"),
        }
        enc.open()
        self._enc = enc
        dec = av.CodecContext.create("h264", "r")
        dec.thread_count = 1
        dec.open()
        self._dec = dec
        if self.width % 2 or self.height % 2:
            raise ValueError("H.264 small stream needs even width and height")        # U then V, each (h/2)x(w/2), packed as h/2 rows of w (from_ndarray's layout).
        self._chroma = np.full((self.height // 2, self.width), 128, np.uint8)

    def _h264(self, gray):
        import av
        frame = av.VideoFrame.from_ndarray(np.vstack((gray, self._chroma)), format="yuv420p")
        frame.pts = self.frames
        packets = self._enc.encode(frame)
        size = sum(p.size for p in packets)
        self.keyframes += sum(1 for p in packets if p.is_keyframe)
        if not packets:
            raise RuntimeError("x264 held a frame: zerolatency/no B-frames not in effect")
        begin = time.perf_counter_ns()
        decoded = []
        for packet in packets:
            decoded.extend(self._dec.decode(packet))
        decode_ms = (time.perf_counter_ns() - begin) / 1e6
        if len(decoded) != 1:
            raise RuntimeError(f"H.264 decoder returned {len(decoded)} frames for one")
        out = decoded[-1].to_ndarray(format="yuv420p")[:self.height, :self.width]
        return np.ascontiguousarray(out), int(size), decode_ms

    # ------------------------------------------------------------------ jpeg
    def _jpeg(self, gray):
        import cv2
        ok, data = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        begin = time.perf_counter_ns()
        out = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        return out, int(data.size), (time.perf_counter_ns() - begin) / 1e6

    # ---------------------------------------------------------------- public
    def roundtrip(self, gray):
        """One small frame (HxW uint8 grey) -> (decoded grey, bytes, decode ms)."""
        gray = np.ascontiguousarray(gray, dtype=np.uint8)
        if gray.shape != (self.height, self.width):
            raise ValueError(f"small frame {gray.shape} != {(self.height, self.width)}")
        if self.codec == "y8":
            out, size, ms = gray, gray.size, 0.0
        elif self.codec == "jpeg":
            out, size, ms = self._jpeg(gray)
        else:
            out, size, ms = self._h264(gray)
        self.frames += 1
        self.sizes.append(size)
        return out, size, ms

    def roundtrip_rgb(self, image):
        """The same for a 3-channel grey frame (the tract's format); returns 3-channel."""
        out, size, ms = self.roundtrip(image[:, :, 0] if image.ndim == 3 else image)
        if image.ndim == 3:
            out = np.repeat(out[:, :, None], image.shape[2], axis=2)
        return out, size, ms

    def roundtrip_color(self, rgb):
        """One colour frame (HxWx3 uint8 RGB) -> (decoded Y, decoded RGB, bytes, decode ms).

        H.264 only.  The frame enters as full-range BT.601 YCbCr 4:2:0 (the
        JPEG / sYCC matrix; its Y is exactly ``cv2.COLOR_RGB2GRAY``), chroma
        reduced by a 2x2 area mean; the laptop gets the decoded Y plane (for
        the marker, which reads luminance) and the frame back in RGB, chroma
        brought up bilinearly (for the opponent and SAM2).  The decode time
        includes that conversion.
        """
        import av
        import cv2
        if self.codec != "h264":
            raise ValueError("a colour stream is H.264 only")
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        h, w = self.height, self.width
        if rgb.shape != (h, w, 3):
            raise ValueError(f"colour frame {rgb.shape} != {(h, w, 3)}")
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        half = (w // 2, h // 2)
        cr = cv2.resize(ycrcb[:, :, 1], half, interpolation=cv2.INTER_AREA)
        cb = cv2.resize(ycrcb[:, :, 2], half, interpolation=cv2.INTER_AREA)
        flat = np.concatenate((ycrcb[:, :, 0].ravel(), cb.ravel(), cr.ravel()))
        frame = av.VideoFrame.from_ndarray(flat.reshape(h * 3 // 2, w), format="yuv420p")
        frame.pts = self.frames
        packets = self._enc.encode(frame)
        size = sum(p.size for p in packets)
        self.keyframes += sum(1 for p in packets if p.is_keyframe)
        if not packets:
            raise RuntimeError("x264 held a frame: zerolatency/no B-frames not in effect")
        begin = time.perf_counter_ns()
        decoded = []
        for packet in packets:
            decoded.extend(self._dec.decode(packet))
        if len(decoded) != 1:
            raise RuntimeError(f"H.264 decoder returned {len(decoded)} frames for one")
        planes = decoded[-1].to_ndarray(format="yuv420p")
        y = np.ascontiguousarray(planes[:h, :w])
        rest = planes[h:].ravel()
        quarter = half[0] * half[1]
        cb = rest[:quarter].reshape(half[1], half[0])
        cr = rest[quarter:2 * quarter].reshape(half[1], half[0])
        up = lambda plane: cv2.resize(plane, (w, h), interpolation=cv2.INTER_LINEAR)  # noqa: E731
        out = cv2.cvtColor(cv2.merge((y, up(cr), up(cb))), cv2.COLOR_YCrCb2RGB)
        decode_ms = (time.perf_counter_ns() - begin) / 1e6
        self.frames += 1
        self.sizes.append(int(size))
        return y, out, int(size), decode_ms


def size_summary(sizes, rate_hz):
    """P50/P95 frame KB and the stream's Mbit/s (mean and from P50) for one camera."""
    if not sizes:
        return None
    array = np.asarray(sizes, dtype=float)
    return {"frames": int(array.size),
            "frame_kb_p50": round(float(np.percentile(array, 50)) / 1000, 3),
            "frame_kb_p95": round(float(np.percentile(array, 95)) / 1000, 3),
            "frame_kb_max": round(float(array.max()) / 1000, 3),
            "mbit_s_mean": round(float(array.mean()) * 8 * rate_hz / 1e6, 3),
            "mbit_s_p50": round(float(np.percentile(array, 50)) * 8 * rate_hz / 1e6, 3)}


# The codec is not tied to the small frame: the full-frame stream (every
# camera sends only its whole 1640x1232 frame as H.264) uses the same class.
StreamCodec = SmallStreamCodec


def scale_like_node(image, width, height):
    """Whole frame -> ``width`` x ``height`` as the node's ``scale_luma`` does it.

    A 2x2 area average first, then an area resize to the size asked for
    (pi_cam/camera_node.py ``scale_luma``, OpenCV branch).  Works on grey or
    3-channel grey.
    """
    import cv2
    rows, cols = (image.shape[0] // 2) * 2, (image.shape[1] // 2) * 2
    binned = cv2.resize(image[:rows, :cols], (cols // 2, rows // 2), interpolation=cv2.INTER_AREA)
    if binned.shape[1] == width and binned.shape[0] == height:
        return binned
    return cv2.resize(binned, (width, height), interpolation=cv2.INTER_AREA)
