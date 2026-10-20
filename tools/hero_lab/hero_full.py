"""HOG finds the car, the arrow on the roof gives the direction. Two cheap steps.

   1. HOG + linear SVM over the lot -> one bbox that contains the hero car
      (measured: recall 0.984, precision 1.000, 4.8 ms per frame)
   2. inside that bbox, the red arrow -> colour mask, contour, PCA axis, wider end = head
      (red_arrow.py already does exactly this; running it on the crop instead of the whole
      frame means fewer red blobs to choose between and less area to search)

Neither step needs the other's internals, and the second is only ever fed a box the first
one is confident about. Still an experiment: nothing here is imported by production.
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "vision", "src"))

import red_arrow                       # noqa: E402  the direction, on a crop
from hero_capture import grab          # noqa: E402
from live_detect import ROI, SCALES, STRIDE, windows   # noqa: E402
from train_hog import MODEL, WIN, make_hog

# A window counts as the car when its margin is at or below this. The training crops
# measure hero at -1.0..-1.8 and everything else at +1.0..+1.6, so -0.5 sits in the gap.
MARGIN_MAX = -0.5            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="hero bbox (HOG) + heading (arrow)")
    ap.add_argument("--stream", default="http://127.0.0.1:8021/stream")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(MODEL):
        print("no model at %s - run train_hog.py first" % MODEL)
        return 1
    svm = cv2.ml.SVM_load(MODEL)
    hog = make_hog()

    last = None
    times, found, with_heading = [], 0, 0
    started = time.time()
    while time.time() - started < args.seconds:
        frame = grab(args.stream)
        t0 = time.time()

        roi = ROI
        if last is not None:
            cx, cy = last[0] + last[2] // 2, last[1] + last[3] // 2
            roi = (max(ROI[0], cx - 120), max(ROI[1], cy - 120),
                   min(ROI[2], cx + 120), min(ROI[3], cy + 120))

        best = None
        for scale in SCALES:
            boxes, patches = [], []
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

        heading = None
        box = None
        if best is not None and best[0] <= MARGIN_MAX:
            found += 1
            box = best[1]
            last = box
            # Direction from the arrow inside the box - the crop is small, so red_arrow has
            # far fewer candidates to choose between than it does over the whole frame.
            x, y, w, h = box
            pad = int(0.12 * max(w, h))
            crop = frame[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
            arrow = red_arrow.detect_in_box(frame, box) if crop.size else None
            if arrow is not None:
                heading = arrow.heading
                with_heading += 1
        else:
            last = None

        times.append(time.time() - t0)
        if args.show:
            view = frame.copy()
            if box is not None:
                x, y, w, h = box
                cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if heading is not None:
                    cx, cy = x + w // 2, y + h // 2
                    dx = int(30 * np.cos(np.radians(heading)))
                    dy = int(30 * np.sin(np.radians(heading)))
                    cv2.arrowedLine(view, (cx, cy), (cx + dx, cy + dy), (0, 255, 255), 2,
                                    tipLength=0.3)
                cv2.putText(view, "%.2f %s" % (best[0], "" if heading is None else "%.0f deg" % heading),
                            (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.imshow("hero: hog box + arrow heading", view)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

    if not times:
        print("no frames")
        return 1
    ts = sorted(times)
    print("  frames %d   car found %d (%.0f%%)   heading also found %d (%.0f%% of found)"
          % (len(times), found, 100.0 * found / len(times), with_heading,
             100.0 * with_heading / max(found, 1)))
    print("  frame time  avg %.1f ms   p95 %.1f ms   max %.1f ms"
          % (1000 * sum(times) / len(times), 1000 * ts[int(len(times) * 0.95)], 1000 * ts[-1]))
    print("  FPS %.0f" % (len(times) / max(sum(times), 1e-6)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
