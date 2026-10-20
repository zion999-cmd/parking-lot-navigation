"""Train P(hero) = 1 on the crops from hero_capture.py, using HOG + a linear SVM.

One question only: is this window the hero car. No direction, no keypoints, no idea what a
car is - the bbox comes from the detector, the geometry inside it can be worked out later
exactly as red_arrow.py does today.

    python tools/hero_lab/train_hog.py            # train, report, save model

HOG parameters are the conventional ones (8x8 cells, 2x2 blocks, 8 px stride, 9 bins) and
the window is fixed at 64x64 so that the cell grid divides exactly. The classifier is
cv2.ml.SVM_LINEAR, which is what keeps inference cheap: one dot product per window.

Splitting matters more than the model. Frames are grouped by the session they were captured
in (the --note given to hero_capture.py) and the groups are split as a whole, so no two
near-identical crops from the same few seconds end up on both sides.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODEL = os.path.join(HERE, "hero_hog.yml")

WIN = (64, 64)
# cv2.HOGDescriptor's Python binding takes these positionally only, so they are spelled out
# in order rather than as keywords: conventional values (8x8 cells, 2x2 blocks, 8 px stride,
# 9 bins, L2-Hys) with a 64x64 window that the cell grid divides exactly.
HOG_ARGS = ((64, 64), (16, 16), (8, 8), (8, 8), 9, 1, -1.0, 0, 0.2, True, 64, False)


def make_hog() -> "cv2.HOGDescriptor":
    return cv2.HOGDescriptor(*HOG_ARGS)


def load_pairs():
    """(image path, label, group) for every crop on disk."""
    boxes_path = os.path.join(DATA, "boxes.jsonl")
    groups = {}
    if os.path.exists(boxes_path):
        for line in open(boxes_path, encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            groups[rec["frame"]] = rec.get("note") or "default"
    out = []
    for path in sorted(glob.glob(os.path.join(DATA, "pos", "*.jpg"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        out.append((path, 1, groups.get(stem, "default")))
    for path in sorted(glob.glob(os.path.join(DATA, "neg", "*.jpg"))):
        stem = os.path.splitext(os.path.basename(path))[0].split("_")[0]
        out.append((path, 0, groups.get(stem, "default")))
    return out


def augment(image: np.ndarray, rng: np.random.Generator):
    """Rotation, brightness, contrast, blur, JPEG. Nothing that changes the car's geometry."""
    out = image
    if rng.random() < 0.7:
        angle = rng.uniform(-15, 15)
        h, w = image.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(out, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.7:
        out = np.clip(out.astype(np.float32) * rng.uniform(0.85, 1.15)
                      + rng.uniform(-20, 20), 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (3, 3), rng.uniform(0, 1.2))
    if rng.random() < 0.4:
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY,
                                            int(rng.uniform(45, 90))])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def features(hog: cv2.HOGDescriptor, image: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    grey = cv2.resize(grey, WIN, interpolation=cv2.INTER_AREA)
    return hog.compute(grey).ravel()


def main() -> int:
    pairs = load_pairs()
    if not pairs:
        print("no crops in %s - run hero_capture.py first" % DATA)
        return 1
    groups = sorted({g for _p, _l, g in pairs})
    if len(groups) < 2:
        print("only one capture session (%s): train and validation cannot be split by "
              "session, so this would overfit. Capture again with a different --note."
              % groups[0])
        return 1

    # Whole sessions to validation, not individual crops.
    val_groups = set(groups[::3])
    train = [(item[0], item[1]) for item in pairs if item[2] not in val_groups]
    val = [(item[0], item[1]) for item in pairs if item[2] in val_groups]
    print("  positives %d  negatives %d" % (
        sum(1 for item in pairs if item[1] == 1), sum(1 for item in pairs if item[1] == 0)))
    print("  train %d  val %d   (val sessions: %s)" % (len(train), len(val), sorted(val_groups)))

    rng = np.random.default_rng(7)
    hog = make_hog()
    def build(items, augment_copies):
        xs, ys = [], []
        for path, label in items:
            image = cv2.imread(path)
            if image is None:
                continue
            copies = [image] if label == 0 else [image] + [augment(image, rng)] * augment_copies
            for copy in copies:
                xs.append(features(hog, copy))
                ys.append(label)
        return np.asarray(xs, np.float32), np.asarray(ys, np.int32)

    x_train, y_train = build(train, augment_copies=4)
    x_val, y_val = build(val, augment_copies=0)
    print("  feature length %d" % x_train.shape[1])

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(0.01)
    svm.train(x_train, cv2.ml.ROW_SAMPLE, y_train)

    for name, xs, ys in (("train", x_train, y_train), ("val", x_val, y_val)):
        _err, pred = svm.predict(xs)
        pred = pred.ravel().astype(int)
        tp = int(((pred == 1) & (ys == 1)).sum())
        fn = int(((pred == 0) & (ys == 1)).sum())
        fp = int(((pred == 1) & (ys == 0)).sum())
        tn = int(((pred == 0) & (ys == 0)).sum())
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        print("  %-5s recall %.3f  precision %.3f  (%d pos, %d neg)"
              % (name, recall, precision, tp + fn, fp + tn))

    svm.save(MODEL)
    print("  saved %s" % MODEL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
