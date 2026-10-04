"""The small stream through a codec in the replay (localization_contracts/small_stream_codec.py)."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from localization_contracts.small_stream_codec import (  # noqa: E402
    SmallStreamCodec, StreamCodec, scale_like_node, size_summary)

W, H = 160, 120


def smooth(seq=0):
    yy, xx = np.mgrid[0:H, 0:W]
    return np.clip(100 + 60 * np.sin(xx / 17.0 + seq * 0.1) * np.cos(yy / 23.0), 0, 255).astype(np.uint8)


try:
    import av  # noqa: F401
    HAVE_AV = True
except ImportError:
    HAVE_AV = False


class Y8(unittest.TestCase):
    def test_y8_returns_the_frame_untouched(self):
        codec = SmallStreamCodec("y8", W, H, 30)
        image = smooth()
        out, size, _ = codec.roundtrip(image)
        self.assertTrue(np.array_equal(out, image))
        self.assertEqual(size, W * H)
        self.assertEqual(codec.sizes, [W * H])

    def test_unknown_codec_and_wrong_size_are_refused(self):
        with self.assertRaises(ValueError):
            SmallStreamCodec("png", W, H, 30)
        with self.assertRaises(ValueError):
            SmallStreamCodec("y8", W, H, 30).roundtrip(np.zeros((H + 2, W), np.uint8))


class Jpeg(unittest.TestCase):
    def test_roundtrip_is_close_and_smaller(self):
        codec = SmallStreamCodec("jpeg", W, H, 30, jpeg_quality=90)
        image = smooth()
        out, size, _ = codec.roundtrip(image)
        self.assertEqual(out.shape, image.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertLess(np.abs(out.astype(int) - image).mean(), 2.0)
        self.assertLess(size, W * H)
        self.assertEqual(codec.sizes, [size])

    def test_three_channel_grey_stays_three_channel(self):
        codec = SmallStreamCodec("jpeg", W, H, 30)
        image = np.repeat(smooth()[:, :, None], 3, axis=2)
        out, _, _ = codec.roundtrip_rgb(image)
        self.assertEqual(out.shape, image.shape)
        self.assertTrue(np.array_equal(out[:, :, 0], out[:, :, 2]))


@unittest.skipUnless(HAVE_AV, "PyAV not installed")
class H264(unittest.TestCase):
    def test_every_frame_comes_back_at_once_and_close(self):
        codec = SmallStreamCodec("h264", W, H, 30, bitrate_mbit=2.0, gop_s=0.2)
        for seq in range(12):
            image = smooth(seq)
            out, size, _ = codec.roundtrip(image)       # nothing held: one in, one out
            self.assertEqual(out.shape, image.shape)
            self.assertEqual(out.dtype, np.uint8)
            self.assertGreater(size, 0)
        self.assertLess(np.abs(out.astype(int) - image).mean(), 3.0)
        self.assertEqual(len(codec.sizes), 12)
        self.assertEqual(codec.gop, 6)
        self.assertEqual(codec.keyframes, 2)             # frames 0 and 6

    def test_inter_frames_are_smaller_than_keyframes(self):
        codec = SmallStreamCodec("h264", W, H, 30, bitrate_mbit=2.0, gop_s=1.0)
        for seq in range(10):
            codec.roundtrip(smooth(seq))
        self.assertLess(max(codec.sizes[1:]), codec.sizes[0])

    def test_cameras_keep_separate_streams(self):
        a = SmallStreamCodec("h264", W, H, 30, bitrate_mbit=2.0)
        b = SmallStreamCodec("h264", W, H, 30, bitrate_mbit=2.0)
        dark = np.full((H, W), 30, np.uint8)
        for seq in range(4):
            out_a, _, _ = a.roundtrip(smooth(seq))
            out_b, _, _ = b.roundtrip(dark)
        self.assertLess(np.abs(out_b.astype(int) - 30).max(), 3)
        self.assertLess(np.abs(out_a.astype(int) - smooth(3)).mean(), 3.0)
        self.assertEqual((len(a.sizes), len(b.sizes)), (4, 4))
        self.assertEqual((a.frames, b.frames), (4, 4))


@unittest.skipUnless(HAVE_AV, "PyAV not installed")
class FullFrame(unittest.TestCase):
    def test_a_full_size_frame_not_a_multiple_of_16_comes_back_whole(self):
        codec = StreamCodec("h264", 1640, 1232, 50, bitrate_mbit=15.0)
        yy, xx = np.mgrid[0:1232, 0:1640]
        image = np.clip(120 + 50 * np.sin(xx / 40.0) * np.cos(yy / 60.0), 0, 255).astype(np.uint8)
        out, size, _ = codec.roundtrip(image)
        self.assertEqual(out.shape, (1232, 1640))
        self.assertLess(np.abs(out.astype(int) - image).mean(), 3.0)
        self.assertEqual(codec.gop, 10)


@unittest.skipUnless(HAVE_AV, "PyAV not installed")
class Colour(unittest.TestCase):
    def colour(self, seq=0):
        grey = smooth(seq).astype(int)
        return np.clip(np.stack((grey + 40, grey, grey - 40), axis=2), 0, 255).astype(np.uint8)

    def test_colour_comes_back_as_y_and_rgb(self):
        import cv2
        codec = SmallStreamCodec("h264", W, H, 30, bitrate_mbit=2.0, gop_s=0.2)
        for seq in range(8):
            image = self.colour(seq)
            y, rgb, size, ms = codec.roundtrip_color(image)
        self.assertEqual((y.shape, y.dtype), ((H, W), np.uint8))
        self.assertEqual((rgb.shape, rgb.dtype), ((H, W, 3), np.uint8))
        self.assertGreater(size, 0)
        self.assertGreaterEqual(ms, 0.0)
        luma = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(int)
        self.assertLess(np.abs(y.astype(int) - luma).mean(), 3.0)
        error = np.abs(rgb.astype(int) - image).reshape(-1, 3).mean(axis=0)
        self.assertLess(error.max(), 4.0)
        # Red stays above blue: the chroma made it through.
        self.assertGreater(rgb[..., 0].mean() - rgb[..., 2].mean(), 60)
        self.assertEqual((codec.frames, len(codec.sizes), codec.keyframes), (8, 8, 2))

    def test_full_size_colour_frame(self):
        codec = StreamCodec("h264", 1640, 1232, 50, bitrate_mbit=15.0)
        yy, xx = np.mgrid[0:1232, 0:1640]
        grey = np.clip(120 + 50 * np.sin(xx / 40.0) * np.cos(yy / 60.0), 0, 255)
        image = np.stack((grey, grey * 0.8, grey * 0.5), axis=2).astype(np.uint8)
        y, rgb, _, _ = codec.roundtrip_color(image)
        self.assertEqual(y.shape, (1232, 1640))
        self.assertEqual(rgb.shape, (1232, 1640, 3))
        self.assertLess(np.abs(rgb.astype(int) - image).mean(), 3.0)

    def test_colour_needs_h264_and_the_right_shape(self):
        with self.assertRaises(ValueError):
            SmallStreamCodec("jpeg", W, H, 30).roundtrip_color(self.colour())
        with self.assertRaises(ValueError):
            SmallStreamCodec("h264", W, H, 30).roundtrip_color(smooth())


class NodeScale(unittest.TestCase):
    def test_bin_then_area_like_the_node(self):
        import cv2
        image = (np.arange(1232 * 1640) % 251).astype(np.uint8).reshape(1232, 1640)
        small = scale_like_node(image, 640, 480)
        self.assertEqual(small.shape, (480, 640))
        binned = cv2.resize(image, (820, 616), interpolation=cv2.INTER_AREA)
        self.assertTrue(np.array_equal(small, cv2.resize(binned, (640, 480),
                                                         interpolation=cv2.INTER_AREA)))
        three = scale_like_node(np.repeat(image[:, :, None], 3, axis=2), 640, 480)
        self.assertTrue(np.array_equal(three[:, :, 1], small))


class Summary(unittest.TestCase):
    def test_kb_and_mbit(self):
        report = size_summary([1000] * 30, 30.0)
        self.assertEqual(report["frame_kb_p50"], 1.0)
        self.assertAlmostEqual(report["mbit_s_p50"], 0.24)
        self.assertIsNone(size_summary([], 30.0))


if __name__ == "__main__":
    unittest.main()
