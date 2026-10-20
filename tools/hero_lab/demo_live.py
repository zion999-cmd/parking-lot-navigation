"""One frame at a time: HOG finds the car box, the arrow inside it gives the direction.

Kept deliberately small and dependency-free so that what it prints is what happened. The
sliding window uses the SAME logic as the frame the user checked by eye: score every window
at both scales, take the SMALLEST margin (cv2.ml measures this model in reverse - the
training crops read -1.0..-1.8 for hero and +1.0..+1.6 for everything else), and then look
for the arrow inside that box with the area filter only, because a rotated arrow's bounding
box is nearly square and an aspect filter rejects it.

    python tools/hero_lab/demo_live.py                 # 5 frames, saves an annotated image
    python tools/hero_lab/demo_live.py --frames 20
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "vision", "src"))

import red_arrow                       # noqa: E402
from hero_capture import grab          # noqa: E402
from live_detect import ROI, SCALES, STRIDE, windows   # noqa: E402
from train_hog import MODEL, WIN, make_hog             # noqa: E402

DEBUG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "vision", "data", "debug")


def best_box(svm, hog, frame):
    patches, boxes = [], []
    for scale in SCALES:
        for box in windows(ROI, scale, STRIDE):
            x, y, w, h = box
            crop = frame[y:y + h, x:x + w]
            if crop.shape[0] < 8 or crop.shape[1] < 8:
                continue
            grey = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), WIN)
            patches.append(hog.compute(grey).ravel())
            boxes.append(box)
    if not patches:
        return None, None
    raw = svm.predict(np.asarray(patches, np.float32),
                      flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)[1].ravel()
    k = int(np.argmin(raw))
    return boxes[k], float(raw[k])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="http://127.0.0.1:8021/stream")
    ap.add_argument("--frames", type=int, default=5)
    args = ap.parse_args()

    svm = cv2.ml.SVM_load(MODEL)
    hog = make_hog()
    os.makedirs(DEBUG, exist_ok=True)

    for i in range(args.frames):
        frame = grab(args.stream)
        box, margin = best_box(svm, hog, frame)
        arrow = red_arrow.detect_in_box(frame, box) if box else None
        print("  frame %d: box %s  margin %+.2f   arrow %s"
              % (i, box, margin,
                 ("centre (%.0f,%.0f) heading %.1f deg" % (arrow.centre[0], arrow.centre[1],
                                                           arrow.heading)) if arrow else "not found"))
        view = frame.copy()
        if box:
            x, y, w, h = box
            cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(view, "margin %+.2f" % margin, (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if arrow:
            cx, cy = int(arrow.centre[0]), int(arrow.centre[1])
            dx = int(45 * np.cos(np.radians(arrow.heading)))
            dy = int(45 * np.sin(np.radians(arrow.heading)))
            cv2.arrowedLine(view, (cx, cy), (cx + dx, cy + dy), (0, 255, 255), 3, tipLength=0.35)
            cv2.putText(view, "%.0f deg" % arrow.heading, (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(DEBUG, "demo_live_%d.png" % i), view)
    print("  saved %s/demo_live_0..%d.png" % (DEBUG, args.frames - 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
