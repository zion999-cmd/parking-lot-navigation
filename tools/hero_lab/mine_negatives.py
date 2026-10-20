"""Hard-negative mining: feed the detector's own mistakes back as training negatives.

The first model had 98.4% recall and zero false positives on the validation crops, and then
failed live: the window it liked best sat at (600, 176) while the car was at (492, 537). The
reason is not the classifier, it is the negatives. hero_capture.py samples them from a fixed
grid, so they are whatever happened to be on that grid - not the background the sliding
window actually lands on when it is looking for a car.

This runs the model over every captured frame at the same scales and stride the live detector
uses, keeps the boxes on the hero side of the margin that are NOT the labelled car, and saves
them as negatives. Retraining on those is the standard second step, and it is cheap: no new
labelling, the frames are already on disk.

    python tools/hero_lab/mine_negatives.py            # then re-run train_hog.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from live_detect import ROI, SCALES, STRIDE, windows   # noqa: E402
from train_hog import DATA, MODEL, WIN, make_hog       # noqa: E402

HARD = os.path.join(DATA, "hard")
PER_FRAME = 6


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    return inter / float(aw * ah + bw * bh - inter)


def main() -> int:
    if not os.path.exists(MODEL):
        print("no model at %s - train first" % MODEL)
        return 1
    svm = cv2.ml.SVM_load(MODEL)
    hog = make_hog()
    os.makedirs(HARD, exist_ok=True)

    boxes_path = os.path.join(DATA, "boxes.jsonl")
    truth = {}
    for line in open(boxes_path, encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            truth[rec["frame"]] = rec["bbox"]

    total = 0
    for path in sorted(glob.glob(os.path.join(DATA, "frames", "*.jpg"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        car = truth.get(stem)
        if car is None:
            continue
        frame = cv2.imread(path)
        if frame is None:
            continue
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
            continue
        raw = svm.predict(np.asarray(patches, np.float32),
                          flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)[1].ravel()
        # Hero side of the margin (cv2.ml measures it in reverse: smaller is more heroic).
        order = np.argsort(raw)
        saved = 0
        for k in order:
            if saved >= PER_FRAME:
                break
            if iou(boxes[k], car) > 0.2:
                continue                     # that one is (partly) the car: not a false positive
            x, y, w, h = boxes[k]
            crop = frame[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(HARD, "%s_%d.jpg" % (stem, saved)),
                        cv2.resize(crop, WIN, interpolation=cv2.INTER_AREA))
            saved += 1
        total += saved
    print("  saved %d hard negatives from %d frames into %s"
          % (total, len(truth), HARD))
    print("  now: python tools/hero_lab/train_hog.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
