"""The hero car by a white arrow on the black roof, read as a shape rather than matched.

The printed-on-white-paper arrow worked while the camera saw it the same way, and stopped
working when the car moved: the template scored 1.000 where it was cut and 0.784 three
hundred pixels away, because the marker's white square measures 60x46 there (aspect 1.30).
That is perspective, and a flat template cannot follow it.

A white arrow on the black roof is a bright shape on a dark ground, so it can be read
geometrically instead of matched:

  1. threshold, close, find bright blobs that do not touch the frame edge
  2. keep the one whose outline matches the reference arrow (Hu moments - rotation and
     scale invariant, which is the whole point)
  3. take the centroid, the principal axis with PCA, and decide which end is the head by
     comparing how wide the two ends are

Measured on a live frame: matchShapes 0.115..0.133 for the marker against 0.27+ for every
other bright blob, stable across thresholds 180-200. The synthetic sweep over 48
combinations (8 rotations x 2 perspectives x 3 blurs) had no failures and a worst heading
error of 8.3 degrees; the perspective that cost the template 0.22 of its score does not
move this at all.

One caveat, measured on the first marker: a head that is only slightly wider than the tail
(1.1:1) makes step 3 noise - the heading flipped with the threshold on a single still
frame. The reference shape here is 3.4:1.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE = os.path.join(ROOT, "data", "arrow", "design", "white_arrow2.png")

# Hu-moment distance to the reference. 0.115..0.133 for the real marker on a live frame,
# 0.27+ for the runners-up, so this sits between them.
MAX_SHAPE_DISTANCE = 0.22
MIN_AREA = 120
MAX_AREA = 3000


@dataclass
class Pose:
    centre: Tuple[float, float]
    heading: float            # 0..360, where the head points
    area: int
    shape_distance: float
    bbox: Tuple[int, int, int, int]


def _reference_contour():
    ref = cv2.imread(REFERENCE, cv2.IMREAD_GRAYSCALE)
    if ref is None:
        return None
    _t, binary = cv2.threshold(ref, 127, 255, cv2.THRESH_BINARY)
    contours, _h = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


_REF = None


def _heading(blob: np.ndarray) -> Tuple[float, float, float]:
    ys, xs = np.where(blob > 0)
    pts = np.stack([xs - xs.mean(), ys - ys.mean()], 1).astype(np.float32)
    _mean, evec = cv2.PCACompute(pts, mean=None, maxComponents=2)
    u = np.array([evec[0][0], evec[0][1]])
    axis = float(np.degrees(np.arctan2(u[1], u[0])) % 360.0)
    proj = pts @ u
    perp = np.abs(pts @ np.array([-u[1], u[0]]))
    lo, hi = float(proj.min()), float(proj.max())
    band = (hi - lo) * 0.35
    wide_lo = float(np.percentile(perp[proj < lo + band], 90))
    wide_hi = float(np.percentile(perp[proj > hi - band], 90))
    head = axis if wide_hi > wide_lo else (axis + 180.0) % 360.0
    return head, wide_lo, wide_hi


def detect(frame: np.ndarray) -> Optional[Pose]:
    """The arrow, or None. Costs a few milliseconds - no rotation sweep, no template."""
    global _REF
    if frame is None or frame.size == 0:
        return None
    if _REF is None:
        _REF = _reference_contour()
        if _REF is None:
            return None

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = grey.shape
    best = None
    for th in (180, 190, 200, 210):
        mask = (grey >= th).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        count, labels, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, count):
            x, y, bw, bh, area = stats[i]
            if area < MIN_AREA or area > MAX_AREA:
                continue
            if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
                continue                      # background that runs off the edge
            blob = (labels == i).astype(np.uint8)
            contours, _h = cv2.findContours(blob * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            d = float(cv2.matchShapes(_REF, contour, cv2.CONTOURS_MATCH_I1, 0.0))
            if d > MAX_SHAPE_DISTANCE:
                continue
            if best is None or d < best.shape_distance:
                head, _lo, _hi = _heading(blob)
                ys, xs = np.where(blob > 0)
                best = Pose(centre=(float(xs.mean()), float(ys.mean())), heading=head,
                            area=int(area), shape_distance=d,
                            bbox=(int(x), int(y), int(bw), int(bh)))
    return best
