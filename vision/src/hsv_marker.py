"""The car's position, from the red arrow on its roof - HSV mask plus Hu-moment shape.

This is the spike in tools/hero_lab/hsv_spike promoted to production, and the reasoning is
in its README. The short version:

  * The arrow is OrangeRed, H 24-46 (median 31), not the H 0-10/170-180 that "red" suggests.
    A mask on the textbook red arcs finds a different red object and reports it confidently.
  * Once the arc covers the arrow, 200 frames gave 2.17 candidates per frame and a candidate
    within 60 px of the arrow in 200/200 of them, with 1.3 px of centre jitter.
  * cv2.matchShapes against one stored arrow contour cuts the rest: <=0.415 keeps 199/200
    arrow candidates and 9/243 of everything else.
  * No model file, no training set, and it does not care where on the table the car is -
    which matters, because the HOG model stopped detecting entirely when the car was moved
    (0 hits in 40 frames) while this found it 40/40.

Position only. The heading is not in here: Hu moments are rotation invariant by construction,
and the template phase of the spike measured too weak at 800x600 to supply an angle. The
caller gets a box and a centre and asks red_arrow.detect_in_box for the direction.
"""

from __future__ import annotations

import json
import os
import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "hsv_marker")
REFERENCE_PATH = os.path.join(DATA_DIR, "reference_arrow.json")

# The measured arc, not the textbook one. See the module docstring.
H1_MIN, H1_MAX = 0, 50
H2_MIN, H2_MAX = 170, 180
S_MIN, V_MIN = 60, 40

MIN_AREA, MAX_AREA = 40, 1200
MIN_WIDTH, MIN_HEIGHT = 8, 8
MIN_ASPECT, MAX_ASPECT = 0.6, 4.0

OPEN_KERNEL = 3
CLOSE_KERNEL = 5

# Measured in the spike. 0.415 kept 199/200 arrow candidates and 9/243 of the rest, but in
# production one of those 9 landed on a bay light and threw the position 220 px across the
# frame. 0.300 costs 5 arrow frames out of 200 (recall 0.975) and drops the false positives to
# 2/243, which is the better trade for something that drives the route.
MAX_SHAPE = 0.300

# The arrow does not leave the table, so anything outside this is not it.
ROI = (0, 60, 800, 560)

_reference: Optional[np.ndarray] = None
_reference_failed = False


def _load_reference() -> Optional[np.ndarray]:
    global _reference, _reference_failed
    if _reference is not None or _reference_failed:
        return _reference
    try:
        with open(REFERENCE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _reference = np.array(data["contour"], np.int32).reshape(-1, 1, 2)
    except Exception:  # missing or unreadable reference: say so, do not crash the loop
        _reference_failed = True
        _reference = None
    return _reference


def red_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (H1_MIN, S_MIN, V_MIN), (H1_MAX, 255, 255))
    mask |= cv2.inRange(hsv, (H2_MIN, S_MIN, V_MIN), (H2_MAX, 255, 255))

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KERNEL, OPEN_KERNEL))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    x, y, w, h = ROI
    keep = np.zeros_like(mask)
    keep[y:y + h, x:x + w] = mask[y:y + h, x:x + w]
    return keep


def _contour(mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x, y, w, h = bbox
    sub = mask[y:y + h, x:x + w]
    contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


# --------------------------------------------------------------------------- candidate pool
#
# One saturation threshold cannot serve the whole table. Measured: at (240,485) the arrow is
# bright (S~115) and S>=60 finds it in 105 of 120 frames, while at (68,475) it sits in shadow
# (S 55..81) and S>=60 keeps only its most saturated core, turning it into an 18x30 diamond that
# fails the shape gate 12/12 times. S>=45 recovers the arrow there (32x17, shape 0.13) but at the
# bright position it merges the arrow with the red around it and drops to 9/120.
#
# So the threshold level becomes a *candidate generator*, not a verdict: every level produces
# blobs, duplicates are merged, and the resulting pool is scored by whoever knows the history
# (hero_belief). No single level decides anything on its own.
S_LEVELS = (60, 50, 45)


def _components(hsv: np.ndarray, s_min: int, v_min: int) -> tuple[np.ndarray, int, np.ndarray,
                                                                  np.ndarray]:
    mask = cv2.inRange(hsv, (H1_MIN, s_min, v_min), (H1_MAX, 255, 255))
    mask |= cv2.inRange(hsv, (H2_MIN, s_min, v_min), (H2_MAX, 255, 255))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KERNEL, OPEN_KERNEL))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    return mask, n, stats, cents


def candidates(frame: np.ndarray, centre: Optional[Tuple[float, float]] = None,
               radius: Optional[float] = None, s_levels: Sequence[int] = S_LEVELS,
               max_shape: float = 0.60) -> list[dict]:
    """A pool of red blobs that could be the arrow, from several saturation levels.

    `centre`/`radius` restrict the search to a window - the caller's belief layer knows roughly
    where the car is, and a local search is both cheaper and far less likely to find a
    distracting red object somewhere else on the table.

    `max_shape` is deliberately looser than the old single-shot gate: the pool is meant to be
    scored later, so rejecting here on the one number that fails at 45 degrees would just move
    the problem. The scorer sees shape_score and weighs it.
    """
    if frame is None or frame.size == 0:
        return []
    reference = _load_reference()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Search window in frame coordinates.
    if centre is not None and radius is not None:
        x0 = int(max(0, centre[0] - radius))
        y0 = int(max(0, centre[1] - radius))
        x1 = int(min(frame.shape[1], centre[0] + radius))
        y1 = int(min(frame.shape[0], centre[1] + radius))
    else:
        x0, y0 = ROI[0], ROI[1]
        x1, y1 = ROI[0] + ROI[2], ROI[1] + ROI[3]
    if x1 - x0 < 8 or y1 - y0 < 8:
        return []

    # Crop first, then threshold. Three saturation levels over the whole 800x600 frame cost
    # 15 ms; over a 180x180 window it costs a fraction of one, and the morphology kernels do not
    # care which pixels they are handed.
    hsv = hsv[y0:y1, x0:x1]

    pool: list[dict] = []
    for s_min in s_levels:
        mask, n, stats, cents = _components(hsv, s_min, V_MIN)
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < MIN_AREA or area > MAX_AREA:
                continue
            if w < MIN_WIDTH or h < MIN_HEIGHT:
                continue
            cx, cy = float(cents[i][0]) + x0, float(cents[i][1]) + y0
            aspect = w / float(h)
            if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
                continue
            contour = _contour(mask, (x, y, w, h))
            if contour is None or len(contour) < 3:
                continue
            shape = 1.0
            if reference is not None:
                shape = float(cv2.matchShapes(contour, reference, cv2.CONTOURS_MATCH_I1, 0.0))
                if shape > max_shape:
                    continue
            # Medians from this blob's own pixels, cropped - a full-frame mask per candidate
            # would cost more than the detection itself.
            sub = mask[y:y + h, x:x + w] > 0
            if not sub.any():
                continue
            crop_hsv = hsv[y:y + h, x:x + w]
            pool.append({
                "bbox": (x + x0, y + y0, w, h),
                "centre": (cx, cy),
                "area": area,
                "width": w,
                "height": h,
                "aspect": aspect,
                "shape_score": round(shape, 4),
                "s_min": s_min,
                "h": float(np.median(crop_hsv[:, :, 0][sub])),
                "s": float(np.median(crop_hsv[:, :, 1][sub])),
                "v": float(np.median(crop_hsv[:, :, 2][sub])),
            })

    # The same physical blob appears once per saturation level. Keep the best-scoring copy -
    # its geometry came from the mask that fit the arrow best.
    pool.sort(key=lambda c: c["shape_score"])
    merged: list[dict] = []
    for c in pool:
        if any(math.hypot(c["centre"][0] - m["centre"][0],
                          c["centre"][1] - m["centre"][1]) < 8.0 for m in merged):
            continue
        merged.append(c)
    return merged


def detect(frame: np.ndarray, prefer_near: Optional[Tuple[float, float]] = None,
           max_shape: float = MAX_SHAPE):
    """The arrow's box and centroid, or None.

    `prefer_near` breaks ties between several arrow-shaped candidates in favour of the one
    closest to where the arrow was last seen - two arrows on the table would otherwise swap.
    """
    if frame is None or frame.size == 0:
        return None

    reference = _load_reference()
    mask = red_mask(frame)
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    best = None
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < MIN_AREA or area > MAX_AREA:
            continue
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            continue
        aspect = w / float(h)
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            continue

        shape = -1.0
        if reference is not None:
            contour = _contour(mask, (x, y, w, h))
            if contour is None or len(contour) < 3:
                continue
            shape = float(cv2.matchShapes(contour, reference, cv2.CONTOURS_MATCH_I1, 0.0))
            if shape > max_shape:
                continue

        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        # An arrow-shaped blob is better than a shapeless one, then bigger is better, then
        # nearer to the last known position.
        distance = 0.0
        if prefer_near is not None:
            distance = ((cx - prefer_near[0]) ** 2 + (cy - prefer_near[1]) ** 2) ** 0.5
        key = (shape, distance, -area)
        if best is None or key < best[0]:
            best = (key, (x, y, w, h), cx, cy, area, aspect, shape)

    if best is None:
        return None
    _key, bbox, cx, cy, area, aspect, shape = best
    return {
        "bbox": bbox,
        "centre": (cx, cy),
        "area": area,
        "aspect": aspect,
        "shape_score": shape,
        "heading": None,   # by design: see the module docstring
    }
