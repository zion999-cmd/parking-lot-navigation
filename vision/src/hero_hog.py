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
    """Where the car is, and what that claim is anchored on.

    `centre` is the ARROW's centroid, not the box centre. The box is only what the sliding
    window proposed; its position jitters with the margin from frame to frame even when the
    car is standing still. The arrow is found by colour and shape, which is stable to about a
    pixel, so once it has been seen it is what the position is pinned to - the box stops
    mattering until the arrow is lost.
    """
    bbox: Tuple[int, int, int, int]
    centre: Tuple[float, float]
    heading: Optional[float]
    margin: float


_SVM = None
_HOG = None
_LAST: Optional[Tuple[int, int, int, int]] = None
_LAST_HEADING: Optional[float] = None   # committed direction; see the gate below
# A reading must be within this of the committed heading to be believed. The failure it guards
# against is a clean 180-degree flip, so anything well under 180 works; 30 leaves room for a
# car turning while being pushed.
HEADING_GATE = 30.0
HEADING_PERSIST = 12     # frames of contradiction before a reversal is accepted as real
_heading_disagreements = 0


def _load():
    global _SVM, _HOG
    if _SVM is None:
        if not os.path.exists(MODEL):
            return False
        _SVM = cv2.ml.SVM_load(MODEL)
        _HOG = cv2.HOGDescriptor(*HOG_ARGS)
    return _SVM is not None


def reset() -> None:
    """Forget the last box and heading."""
    global _LAST, _LAST_HEADING
    _LAST = None
    _LAST_HEADING = None


def detect(frame: np.ndarray) -> Optional[Detection]:
    """The car and its heading, or None."""
    global _LAST_HEADING, _heading_disagreements
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

    # Stage two: the arrow inside the box. It rejects a false positive (no arrow, no car) and
    # it is what the position is anchored to - the box only proposed a search area, and its
    # centre moves by a few pixels between frames even when the car does not.
    arrow = red_arrow.detect_in_box(frame, box)
    if arrow is None:
        # No arrow this frame - blurred, occluded, or the blob filter was strict. The car is
        # still there (the sliding window is confident and the box has not moved), so report
        # it with the direction already committed instead of dropping the car entirely.
        # Dropping it is what made the marker "disappear" for a frame and the page replan.
        if _LAST_HEADING is None:
            _LAST = None
            return None
        _LAST = box
        return Detection(bbox=box, centre=(box[0] + box[2] / 2.0, box[1] + box[3] / 2.0),
                         heading=float(_LAST_HEADING), margin=margin)

    # Position comes from the box, direction from the arrow, and the split is measured:
    # with the car standing still the box centre held to 0.0 px over twenty frames while the
    # arrow centroid moved 55 px, because the arrow detector can pick a red LED instead of the
    # arrow. So the box is the better position and the arrow is the only thing that can give a
    # direction - it is the one object on the table there is exactly one of.
    #
    # The arrow is required to be inside the box; an arrow found elsewhere is not this car's.
    margin_ok = (box[0] - 10 <= arrow.centre[0] <= box[0] + box[2] + 10
                 and box[1] - 10 <= arrow.centre[1] <= box[1] + box[3] + 10)
    if margin_ok:
        # Anchor the heading, as the user asked: the head/tail decision is a width comparison,
        # and on this marker it flips by exactly 180 degrees between frames that differ only
        # in blur (measured: 185, 5, 182, 4, 186, 5 ...). A car cannot change ends, so a
        # reading that contradicts the committed one by more than HEADING_GATE is ignored
        # until it keeps repeating - see _heading_disagreements.
        reading = float(arrow.heading % 360.0)
        if _LAST_HEADING is None:
            heading = reading
        else:
            diff = abs((reading - _LAST_HEADING + 180.0) % 360.0 - 180.0)
            if diff <= HEADING_GATE:
                heading = reading
            else:
                _heading_disagreements += 1
                # Twelve frames of agreement against the committed value is a real turn, not
                # a flip: measured, a genuine reversal produces a sustained run.
                if _heading_disagreements >= HEADING_PERSIST:
                    heading = reading
                    _heading_disagreements = 0
                else:
                    heading = _LAST_HEADING
        if heading is not _LAST_HEADING:
            _heading_disagreements = 0
        _LAST_HEADING = heading
    else:
        # A red blob outside the car's box is something else - a bay light, most likely - and
        # must not be allowed to set the direction. Keep the last trusted one instead.
        heading = _LAST_HEADING
    if heading is None:
        return None                      # a car with no direction yet: not usable as a pose

    _LAST = box
    return Detection(bbox=box, centre=(box[0] + box[2] / 2.0, box[1] + box[3] / 2.0),
                     heading=heading, margin=margin)
