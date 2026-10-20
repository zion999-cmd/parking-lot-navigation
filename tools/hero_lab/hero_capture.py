"""Collect P(hero) training crops: grab frames from the live relay, drag a box, save.

Scope: this only produces a dataset. It does not know what the car is made of, does not
detect anything, and does not touch production code. The model that consumes the result
lives in train_hog.py and answers exactly one question - is this window the hero car.

    python tools/hero_lab/hero_capture.py                 # live frames from :8021
    python tools/hero_lab/hero_capture.py --every 30      # every 30th frame

Keys in the preview window:
    drag a box around the car, then
    SPACE / ENTER   keep this frame: saves the image, the bbox, the positive crop and a
                    handful of automatic negative crops from the rest of the same frame
    S               skip this frame
    Q / ESC         quit

Every saved frame also yields negatives - other cars, empty bays, road, the LED strip,
the gate - by sampling boxes from a grid that does not overlap the positive. That is what
makes colour and shape look-alikes part of the training set without hand-labelling them.

Data layout (all under tools/hero_lab/data, nothing outside this directory):

    frames/000123.jpg            the full frame, as captured
    boxes.jsonl                  one line per kept frame: frame, bbox, note
    pos/000123.jpg               the crop, resized to the detector window
    neg/000123_0.jpg ...         automatic negatives from the same frame
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FRAMES = os.path.join(DATA, "frames")
POS = os.path.join(DATA, "pos")
NEG = os.path.join(DATA, "neg")
BOXES = os.path.join(DATA, "boxes.jsonl")

# One window for the first version. Chosen from the measured hero bbox (about 90x40 px with
# the marker on the roof, so roughly 2:1); 64x64 keeps HOG's default 8x8 cells exact and
# keeps the aspect close enough that a car crop is not stretched into nonsense.
WIN_W, WIN_H = 64, 64
NEG_PER_FRAME = 4
NEG_MIN_GAP = 0.15          # IoU above which an automatic negative is dropped


def grab(stream: str, timeout: float = 15.0) -> np.ndarray:
    """One JPEG frame from an MJPEG stream."""
    handle = urllib.request.urlopen(stream, timeout=timeout)
    buffer = b""
    try:
        while True:
            buffer += handle.read(16384)
            start = buffer.find(b"\xff\xd8")
            end = buffer.find(b"\xff\xd9", start)
            if start >= 0 and end > start:
                frame = cv2.imdecode(np.frombuffer(buffer[start:end + 2], np.uint8),
                                     cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
    finally:
        handle.close()
    raise RuntimeError("no frame from %s" % stream)


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    return inter / float(aw * ah + bw * bh - inter)


def auto_negatives(frame: np.ndarray, bbox, want: int) -> list:
    """Boxes from a grid that overlap neither the positive nor each other much."""
    h, w = frame.shape[:2]
    size = int(min(WIN_W, WIN_H) * 1.6)
    out = []
    for y in range(0, h - size, size // 2):
        for x in range(0, w - size, size // 2):
            box = (x, y, size, size)
            if iou(box, bbox) > 0.0:
                continue
            if any(iou(box, other) > NEG_MIN_GAP for other in out):
                continue
            out.append(box)
    # Spread them out rather than taking the top-left corner every time.
    if len(out) > want:
        step = len(out) / float(want)
        out = [out[int(i * step)] for i in range(want)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect hero/non-hero crops by hand.")
    ap.add_argument("--stream", default="http://127.0.0.1:8021/stream")
    ap.add_argument("--every", type=int, default=20, help="keep one frame in N")
    ap.add_argument("--note", default="", help="written into boxes.jsonl for this session")
    args = ap.parse_args()

    for d in (FRAMES, POS, NEG):
        os.makedirs(d, exist_ok=True)

    index = len([f for f in os.listdir(FRAMES) if f.endswith(".jpg")])
    print("  stream : %s" % args.stream)
    print("  keeping 1 frame in %d, starting at %04d" % (args.every, index))
    print("  drag a box around the hero, SPACE to keep, S to skip, Q to quit")

    seen = 0
    kept = 0
    while True:
        frame = grab(args.stream)
        seen += 1
        if seen % args.every:
            continue

        box = cv2.selectROI("hero: drag a box, SPACE=keep  S=skip  Q=quit", frame,
                            showCrosshair=False, fromCenter=False)
        cv2.destroyWindow("hero: drag a box, SPACE=keep  S=skip  Q=quit")
        x, y, w, h = [int(v) for v in box]
        if w == 0 or h == 0:
            print("  skipped")
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            continue

        stem = "%06d" % index
        cv2.imwrite(os.path.join(FRAMES, stem + ".jpg"), frame)
        crop = frame[y:y + h, x:x + w]
        if crop.size:
            cv2.imwrite(os.path.join(POS, stem + ".jpg"),
                        cv2.resize(crop, (WIN_W, WIN_H), interpolation=cv2.INTER_AREA))
        negatives = auto_negatives(frame, (x, y, w, h), NEG_PER_FRAME)
        for k, (nx, ny, nw, nh) in enumerate(negatives):
            cv2.imwrite(os.path.join(NEG, "%s_%d.jpg" % (stem, k)),
                        cv2.resize(frame[ny:ny + nh, nx:nx + nw], (WIN_W, WIN_H),
                                   interpolation=cv2.INTER_AREA))
        with open(BOXES, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"frame": stem, "bbox": [x, y, w, h],
                                 "negatives": negatives, "note": args.note,
                                 "at": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
        print("  kept %s  bbox %dx%d at (%d,%d)  + %d negatives  (total %d)"
              % (stem, w, h, x, y, len(negatives), kept + 1))
        index += 1
        kept += 1
    print("  done: %d kept this session, %d frames on disk" % (kept, index))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
