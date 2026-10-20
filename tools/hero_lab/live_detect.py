"""Run the trained HOG + linear SVM over live frames and report hero boxes at what cost.

    python tools/hero_lab/live_detect.py            # print per-second FPS and the best box
    python tools/hero_lab/live_detect.py --show     # also open a window with the box drawn

Deliberately simple, because the question is whether the approach is worth more work:
a multi-scale sliding window, with the scales and stride kept few, the search limited to
the lot, and the previous position tried first. No tracking, no Kalman, no direction.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hero_capture import grab          # noqa: E402
from train_hog import MODEL, WIN, make_hog

# A window counts as the car when its margin is at or below this. The training crops
# measure hero at -1.0..-1.8 and everything else at +1.0..+1.6, so -0.5 sits in the gap.
MARGIN_MAX = -0.5  # noqa: E402

# The lot, roughly. Searching the whole 800x600 for every scale is wasted work on the wall
# and the shelf, and this is the kind of bound production already uses.
ROI = (0, 80, 800, 520)
SCALES = (1.0, 1.5)          # 64 px and 96 px windows
STRIDE = 24
NEAR_HALF = 120              # when the car was just found, look around it first


def windows(roi, scale, stride):
    x0, y0, x1, y1 = roi
    w = int(WIN[0] * scale)
    h = int(WIN[1] * scale)
    for y in range(y0, y1 - h, stride):
        for x in range(x0, x1 - w, stride):
            yield x, y, w, h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="http://127.0.0.1:8021/stream")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(MODEL):
        print("no model at %s - run train_hog.py first" % MODEL)
        return 1
    svm = cv2.ml.SVM_load(MODEL)
    hog = make_hog()

    last = None
    times, hits, misses = [], 0, 0
    started = time.time()
    while time.time() - started < args.seconds:
        frame = grab(args.stream)
        t0 = time.time()

        roi = ROI
        if last is not None:
            cx, cy = last[0] + last[2] // 2, last[1] + last[3] // 2
            roi = (max(ROI[0], cx - NEAR_HALF), max(ROI[1], cy - NEAR_HALF),
                   min(ROI[2], cx + NEAR_HALF), min(ROI[3], cy + NEAR_HALF))

        best = None
        for scale in SCALES:
            boxes = []
            patches = []
            for box in windows(roi, scale, STRIDE):
                x, y, w, h = box
                crop = frame[y:y + h, x:x + w]
                if crop.shape[0] < 8 or crop.shape[1] < 8:
                    continue
                grey = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), WIN,
                                  interpolation=cv2.INTER_AREA)
                patches.append(hog.compute(grey).ravel())
                boxes.append(box)
            if not patches:
                continue
            # Raw margin, not the class label: the label is >= 0 for anything the SVM leans
            # towards, which admitted a false positive in the top-left corner. A positive
            # margin means it leans towards hero; MARGIN_MIN says how hard it must lean.
            _r, raw = svm.predict(np.asarray(patches, np.float32),
                                  flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
            # cv2.ml scores this model in reverse: the training crops measure -1.0..-1.8 for
            # hero and +1.0..+1.6 for everything else, a clean gap at zero. So the most heroic
            # window is the SMALLEST margin, not the largest, and MARGIN_MAX is the ceiling a
            # window must stay under.
            scores = raw.ravel()
            k = int(np.argmin(scores))
            if best is None or scores[k] < best[0]:
                best = (float(scores[k]), boxes[k])

        dt = time.time() - t0
        times.append(dt)
        if best is not None and best[0] <= MARGIN_MAX:
            last = best[1]
            hits += 1
        else:
            last = None
            misses += 1

        if args.show and best is not None:
            view = frame.copy()
            x, y, w, h = best[1]
            cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(view, "%.2f" % best[0], (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
            cv2.imshow("hog hero", view)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

    if not times:
        print("no frames")
        return 1
    times_sorted = sorted(times)
    print("  frames  %d   hits %d   misses %d" % (len(times), hits, misses))
    print("  frame time  avg %.1f ms  p95 %.1f ms  max %.1f ms"
          % (1000 * sum(times) / len(times), 1000 * times_sorted[int(len(times) * 0.95)],
             1000 * times_sorted[-1]))
    print("  FPS         %.1f" % (len(times) / max(sum(times), 1e-6)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
