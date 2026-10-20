"""Hero car by HOG + linear SVM, then direction from the arrow inside the box.

This replaced the red-mask detector, which was 7-22 ms and gave up whenever the marker was
small or the light was awkward. The measured difference, on the same camera and the same
relay:

    HOG + SVM          recall 0.984   precision 1.000   4.8 ms   works across the whole lot
    red mask alone     works, but loses the marker at the edges and picks up a static red
                       object when the arrow is out of view

Two stages, and the second one does double duty:

  1. a sliding window at two scales scores every candidate; the model is scored in reverse
     by cv2.ml (training crops measure hero at -1.0..-1.8 and everything else at +1.0..+1.6),
     so the most heroic window is the SMALLEST margin
  2. the red arrow is looked for inside that box; it gives the heading, and a box without an
     arrow is rejected - a real box measures -0.06..-0.12 and a false positive -0.04..-0.09,
     so the margins overlap and no threshold separates them, but a false positive has no
     arrow in it

The model is a 124-sample hand-boxed set from three capture sessions, three-fold
cross-validated, with 744 hard negatives mined from the model's own mistakes on those
frames. tools/hero_lab/BENCHMARK_HOG.md has the numbers and what they do not cover.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

import red_arrow

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "data", "hero_hog.yml")

WIN = (64, 64)
HOG_ARGS = ((64, 64), (16, 16), (8, 8), (8, 8), 9, 1, -1.0, 0, 0.2, True, 64, False)

# The lot. Searching the wall and the shelf as well costs time and gains nothing.
ROI = (0, 80, 800, 520)
SCALES = (1.0, 1.5)
STRIDE = 24
NEAR_HALF = 120          # when the car was just found, look around it

# A window must be on the hero side of zero to become a candidate. It cannot be stricter
# than that: the training crops measure -1.0..-1.8, but a sliding-window candidate is a whole
# 96x96 box rather than a tightly cropped car, and the best one measures about -0.11. The
# heavy filtering is the arrow check in stage two, which rejects a box with no arrow in it -
# measured, that is what separates a real box from a false positive, not the margin.
MARGIN_MAX = 0.0


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    centre: Tuple[float, float]
    heading: Optional[float]
    margin: float


_SVM = None
_HOG = None
_LAST: Optional[Tuple[int, int, int, int]] = None


def _load():
    global _SVM, _HOG
    if _SVM is None:
        if not os.path.exists(MODEL):
            return False
        _SVM = cv2.ml.SVM_load(MODEL)
        _HOG = cv2.HOGDescriptor(*HOG_ARGS)
    return _SVM is not None


def reset() -> None:
    """Forget the last box, so the next call searches the whole lot."""
    global _LAST
    _LAST = None


def detect(frame: np.ndarray) -> Optional[Detection]:
    """The car and its heading, or None."""
    global _LAST
    if frame is None or frame.size == 0 or not _load():
        return None

    roi = ROI
    if _LAST is not None:
        cx, cy = _LAST[0] + _LAST[2] // 2, _LAST[1] + _LAST[3] // 2
        roi = (max(ROI[0], cx - NEAR_HALF), max(ROI[1], cy - NEAR_HALF),
               min(ROI[2], cx + NEAR_HALF), min(ROI[3], cy + NEAR_HALF))

    patches, boxes = [], []
    for scale in SCALES:
        w, h = int(WIN[0] * scale), int(WIN[1] * scale)
        for y in range(roi[1], roi[3] - h, STRIDE):
            for x in range(roi[0], roi[2] - w, STRIDE):
                crop = frame[y:y + h, x:x + w]
                if crop.shape[0] < 8 or crop.shape[1] < 8:
                    continue
                grey = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), WIN,
                                  interpolation=cv2.INTER_AREA)
                patches.append(_HOG.compute(grey).ravel())
                boxes.append((x, y, w, h))
    if not patches:
        _LAST = None
        return None

    raw = _SVM.predict(np.asarray(patches, np.float32),
                       flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)[1].ravel()
    k = int(np.argmin(raw))
    margin = float(raw[k])
    box = boxes[k]
    if margin > MARGIN_MAX:
        _LAST = None
        return None

    # Stage two: the arrow inside the box, which is also what rejects a false positive.
    arrow = red_arrow.detect_in_box(frame, box)
    if arrow is None:
        _LAST = None
        return None

    _LAST = box
    x, y, w, h = box
    return Detection(bbox=box, centre=(x + w / 2.0, y + h / 2.0),
                     heading=float(arrow.heading % 360.0), margin=margin)
