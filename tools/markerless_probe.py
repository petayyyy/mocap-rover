#!/usr/bin/env python3
"""Look at markerless rover detection on real footage, frame by frame.

This is a probe, not the runtime.  It answers one question: on video from the
actual arena, does an appearance-independent detector draw a believable box
around each rover, and what does its binary mask look like?  Nothing here is
trained on a rover, so it does not care what the rovers look like.

How it works
------------
The cameras are fixed and the arena is fixed, so anything that was not there
a moment ago is a candidate.  A per-pixel background model (MOG2) learns the
empty arena continuously, which also absorbs slow lighting drift and parts
that fall off a rover and come to rest.  Its shadow class is thrown away, and
what survives is cleaned with morphology and split into connected components.

A component is reported when it is the right size and compact enough to be a
vehicle.  Those two numbers are the only thing standing between "a rover" and
"a person walking through the shot", so they are on trackbars: move them while
watching, rather than guessing them offline.

Two background models, switched with "m", because they fail differently:

``adaptive`` (MOG2)
    Learns continuously, so it shrugs off lighting drift and quietly absorbs
    debris that falls off a rover and stops.  It also absorbs a rover that
    stops, which is why a parked rover vanishes from it -- on the polygon
    footage the second rover was standing still and this model never saw it.

``reference``
    One median image of the empty arena, differenced forever.  A stopped
    rover stays visible, which is what you want on a start line, but nothing
    is forgotten either: change the lighting and the whole frame lights up.

The strongest filter is not a number though: it is the arena outline.  On the
polygon footage every false positive sat outside the floor -- people walking
behind the barrier, LED panels flickering on the wall -- and a drawn outline
removes all of them at once without touching a threshold.  Press "a" and click
the corners; the outline is saved next to the video and reloaded next time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

WINDOW = "markerless probe"
CONTROLS = "controls"


def parse_roi(text, width, height):
    """``x,y,w,h`` in pixels, or ``top``/``bottom`` for a stacked two-camera frame."""
    if not text:
        return 0, 0, width, height
    if text == "top":
        return 0, 0, width, height // 2
    if text == "bottom":
        return 0, height // 2, width, height - height // 2
    x, y, w, h = (int(v) for v in text.split(","))
    return x, y, w, h


class Detector:
    def __init__(self, history, var_threshold):
        self.reference = None
        self.mode = "adaptive"
        self.reset(history, var_threshold)

    def reset(self, history, var_threshold):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=True)
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    def set_reference(self, frames):
        """Median of several frames, so a rover crossing during capture is voted out."""
        self.reference = np.median(np.stack(frames), axis=0).astype(np.uint8)

    def __call__(self, frame, learning_rate, min_area, max_area, min_fill, blur,
                 arena=None, var_threshold=28):
        work = cv2.GaussianBlur(frame, (blur * 2 + 1, blur * 2 + 1), 0) if blur else frame
        if self.mode == "reference" and self.reference is not None:
            delta = cv2.absdiff(cv2.cvtColor(work, cv2.COLOR_BGR2GRAY),
                                cv2.cvtColor(self.reference, cv2.COLOR_BGR2GRAY))
            mask = cv2.threshold(delta, max(8, var_threshold // 2), 255,
                                 cv2.THRESH_BINARY)[1]
            # Keep feeding MOG2 so switching back does not restart its history.
            self.bg.apply(work, learningRate=learning_rate)
        else:
            raw = self.bg.apply(work, learningRate=learning_rate)
            # MOG2 marks shadows 127 and foreground 255; keeping only 255 drops
            # the shadow a rover drags behind it under the arena lights.
            mask = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
        if arena is not None:
            mask = cv2.bitwise_and(mask, arena)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        hits = []
        for i in range(1, count):
            x, y, w, h, area = stats[i]
            if not (min_area <= area <= max_area):
                continue
            # A vehicle fills most of its own box; a person, a cable or a
            # lighting artefact is long, thin or stringy.
            if area < min_fill / 100.0 * w * h:
                continue
            hits.append((x, y, w, h, area, centroids[i]))
        hits.sort(key=lambda hit: -hit[4])
        return mask, hits


def arena_path(source):
    """Keep the outline next to the footage it was drawn on."""
    return Path(str(source) + ".arena.json")


def load_arena(source):
    path = arena_path(source)
    if not path.exists():
        return None
    try:
        points = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return np.asarray(points, dtype=np.int32) if len(points) >= 3 else None


def draw_arena(frame, existing):
    """Click the corners of the floor; ENTER accepts, ESC cancels, u undoes."""
    points = [] if existing is None else [tuple(p) for p in existing.tolist()]

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        view = frame.copy()
        if points:
            cv2.polylines(view, [np.asarray(points, np.int32)], len(points) > 2,
                          (0, 255, 255), 2)
            for point in points:
                cv2.circle(view, point, 4, (0, 255, 255), -1)
        for i, line in enumerate((
                "click the corners of the arena floor",
                "ENTER accept   u undo   c clear   ESC cancel",
                f"{len(points)} points")):
            cv2.putText(view, line, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(view, line, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            cv2.setMouseCallback(WINDOW, lambda *a: None)
            return np.asarray(points, np.int32) if len(points) >= 3 else None
        if key == 27:
            cv2.setMouseCallback(WINDOW, lambda *a: None)
            return existing
        if key == ord("u") and points:
            points.pop()
        if key == ord("c"):
            points.clear()


def build_controls(args):
    cv2.namedWindow(CONTROLS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROLS, 460, 300)
    for name, value, limit in (
        ("min area", args.min_area, 20000),
        ("max area", args.max_area, 200000),
        ("min fill %", args.min_fill, 100),
        ("var threshold", args.var_threshold, 200),
        ("learn x1000", int(args.learning_rate * 1000), 100),
        ("blur", args.blur, 6),
    ):
        cv2.createTrackbar(name, CONTROLS, int(value), limit, lambda _v: None)


def read_controls():
    return dict(
        min_area=max(1, cv2.getTrackbarPos("min area", CONTROLS)),
        max_area=max(2, cv2.getTrackbarPos("max area", CONTROLS)),
        min_fill=cv2.getTrackbarPos("min fill %", CONTROLS),
        var_threshold=max(1, cv2.getTrackbarPos("var threshold", CONTROLS)),
        learning_rate=cv2.getTrackbarPos("learn x1000", CONTROLS) / 1000.0,
        blur=cv2.getTrackbarPos("blur", CONTROLS),
    )


def annotate(view, hits, info):
    for rank, (x, y, w, h, area, (cx, cy)) in enumerate(hits):
        colour = (0, 235, 0) if rank < 2 else (0, 170, 255)
        cv2.rectangle(view, (x, y), (x + w, y + h), colour, 2)
        cv2.drawMarker(view, (int(cx), int(cy)), colour, cv2.MARKER_CROSS, 14, 2)
        cv2.putText(view, f"{int(area)}px", (x, max(12, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    for i, line in enumerate(info):
        cv2.putText(view, line, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(view, line, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return view


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="video file, image file, or a camera index")
    p.add_argument("--roi", default="",
                   help="'x,y,w,h', or 'top'/'bottom' for a stacked two-camera frame")
    p.add_argument("--start", type=float, default=0.0, help="seek to this second")
    p.add_argument("--scale", type=float, default=1.0, help="resize before processing")
    p.add_argument("--min-area", type=int, default=150)
    p.add_argument("--max-area", type=int, default=40000)
    p.add_argument("--min-fill", type=int, default=25, help="percent of the box filled")
    p.add_argument("--var-threshold", type=int, default=28)
    p.add_argument("--learning-rate", type=float, default=0.004)
    p.add_argument("--blur", type=int, default=1)
    p.add_argument("--history", type=int, default=400)
    p.add_argument("--warmup", type=int, default=60,
                   help="frames fed to the background model before reporting")
    p.add_argument("--mode", choices=("adaptive", "reference"), default="adaptive",
                   help="background model to start in; 'm' switches it live")
    p.add_argument("--save-dir", default=None,
                   help="write annotated frames here instead of opening a window")
    p.add_argument("--save-every", type=int, default=30)
    p.add_argument("--max-frames", type=int, default=0)
    a = p.parse_args()

    source = int(a.source) if a.source.isdigit() else a.source
    still = (not isinstance(source, int)
             and Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"cannot open {a.source}")
    if a.start and not still:
        capture.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000.0)

    headless = a.save_dir is not None
    if headless:
        Path(a.save_dir).mkdir(parents=True, exist_ok=True)
    else:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        build_controls(a)

    reference_frames = []
    arena_points = load_arena(a.source)
    arena_mask = None
    detector = Detector(a.history, a.var_threshold)
    detector.mode = a.mode
    settings = dict(min_area=a.min_area, max_area=a.max_area, min_fill=a.min_fill,
                    learning_rate=a.learning_rate, blur=a.blur,
                    var_threshold=a.var_threshold)
    previous_var = a.var_threshold
    index = 0
    paused = False
    frame = None
    while True:
        if not paused or frame is None:
            ok, frame = capture.read()
            if not ok:
                if still or headless:
                    break
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            index += 1
        if a.max_frames and index > a.max_frames:
            break
        if not headless:
            settings = read_controls()
            if settings["var_threshold"] != previous_var:
                detector.reset(a.history, settings["var_threshold"])
                previous_var = settings["var_threshold"]
                index = 0

        working = frame
        if a.scale != 1.0:
            working = cv2.resize(working, (0, 0), fx=a.scale, fy=a.scale)
        x, y, w, h = parse_roi(a.roi, working.shape[1], working.shape[0])
        working = working[y:y + h, x:x + w]

        if arena_points is not None and (
                arena_mask is None or arena_mask.shape[:2] != working.shape[:2]):
            arena_mask = np.zeros(working.shape[:2], np.uint8)
            cv2.fillPoly(arena_mask, [arena_points], 255)
        mask, hits = detector(
            working, settings["learning_rate"], settings["min_area"],
            settings["max_area"], settings["min_fill"], settings["blur"],
            arena_mask, settings["var_threshold"])
        if index <= a.warmup:
            hits = []

        if len(reference_frames) < 25:
            reference_frames.append(working.copy())
            if len(reference_frames) == 25:
                detector.set_reference(reference_frames)
        info = [f"[{detector.mode}] frame {index}  "
                f"{'warming up' if index <= a.warmup else f'{len(hits)} blobs'}",
                f"area {settings['min_area']}-{settings['max_area']}  "
                f"fill>{settings['min_fill']}%  var {settings['var_threshold']}  "
                f"learn {settings['learning_rate']:.3f}"]
        info.append(("a: draw arena outline" if arena_points is None
                     else "arena outline active")
                    + "   m: background model   r: re-capture reference")
        view = annotate(working.copy(), hits, info)
        if arena_points is not None:
            cv2.polylines(view, [arena_points], True, (0, 220, 220), 1)
        both = np.hstack([view, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])

        if headless:
            if index % a.save_every == 0:
                cv2.imwrite(f"{a.save_dir}/f{index:05d}.jpg", both)
            continue

        cv2.imshow(WINDOW, both)
        key = cv2.waitKey(0 if still else 25) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        if key == ord("m"):
            detector.mode = ("reference" if detector.mode == "adaptive"
                             else "adaptive")
        if key == ord("r"):
            reference_frames.clear()
            detector.reference = None
            print("re-capturing the reference background")
        if key == ord("a"):
            arena_points = draw_arena(working, arena_points)
            arena_mask = None
            if arena_points is not None:
                arena_path(a.source).write_text(json.dumps(arena_points.tolist()))
                print(f"wrote {arena_path(a.source)}")
        if key == ord("s"):
            cv2.imwrite(f"probe_{index:05d}.png", both)
            print(f"wrote probe_{index:05d}.png")
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
