"""The hero car by the red arrow on its white paper, found by colour and shape.

Colour alone is not enough and a template alone is not enough; together they are. Measured
on six hand-annotated positions across the table:

    colour mask alone          the largest red blob is a static red object in the room
                               (1600 px at (356, 298) in all six frames), not the car
    template matching          works at 1.000 where the template was cut and 0.5 at the
                               right edge, where the marker measures 453 in Laplacian
                               variance against 1764 in the middle - four times blurrier
    colour + size + aspect     all six positions, offset 2..5 px, including both edges

That last row is the design here. Blur destroys the fine structure a template needs, but it
does not change which pixels are red, so the blob's extent, centroid and principal axis
survive it. What colour cannot do on its own is say *which* red blob, which is why the size
and aspect filters matter - they are what excludes the static red object - and why the last
position is preferred when several candidates qualify.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# The arrow, in image pixels. Measured across six positions: 146..256 px of red,
# bbox 13x20 .. 30x16. The static red object is ~1600 px, so 60..800 separates them cleanly.
MIN_AREA = 60
MAX_AREA = 800
MIN_ASPECT = 1.2
MAX_ASPECT = 3.5

# Red, with enough saturation and value that the white paper and the black roof cannot pass.
HUE_NEAR_0 = 12
HUE_NEAR_180 = 168
MIN_SAT = 70
MIN_VAL = 50


@dataclass
class Pose:
    centre: Tuple[float, float]
    heading: float            # 0..360, the end the head points to
    area: int
    bbox: Tuple[int, int, int, int]
    aspect: float


def _mask(hsv: np.ndarray) -> np.ndarray:
    h = hsv[:, :, 0].astype(np.int16)
    s = hsv[:, :, 1].astype(np.int16)
    v = hsv[:, :, 2].astype(np.int16)
    red = ((h <= HUE_NEAR_0) | (h >= HUE_NEAR_180)) & (s >= MIN_SAT) & (v >= MIN_VAL)
    return (red.astype(np.uint8)) * 255


def _heading(blob: np.ndarray) -> float:
    ys, xs = np.where(blob > 0)
    pts = np.stack([xs - xs.mean(), ys - ys.mean()], 1).astype(np.float32)
    _mean, evec = cv2.PCACompute(pts, mean=None, maxComponents=2)
    u = np.array([evec[0][0], evec[0][1]])
    axis = float(np.degrees(np.arctan2(u[1], u[0])) % 360.0)
    proj = pts @ u
    perp = np.abs(pts @ np.array([-u[1], u[0]]))
    lo, hi = float(proj.min()), float(proj.max())
    band = max((hi - lo) * 0.30, 1.0)
    narrow = float(np.percentile(perp[proj < lo + band], 85))
    wide = float(np.percentile(perp[proj > hi - band], 85))
    # The head is the wider end. The PCA axis itself is a line with two ends, so this is the
    # only thing that makes the heading a direction rather than an axis.
    return axis if wide > narrow else (axis + 180.0) % 360.0


# The heading is measured on an upsampled ROI, not on the frame. Reason, measured over 24
# angles: at 0/90/180/270 the head/tail width margin is 2.72 px and the direction is exact, but
# at 45/135/225/315 it collapses to 0.51 px - sub-pixel - and the decision becomes a coin flip,
# 114 wrong out of 228 frames. A 32x19 px arrow has no margin left at 45 degrees; that is a
# rasterisation limit, not a threshold to tune. Upsampling the ROI 4x before the mask restores
# the margin to 10.62 px and every flip disappears, and it is cheaper than scanning the frame
# because the caller already knows where the arrow is.
ROI_HALF = 24
ROI_SCALE = 4


def heading_in_roi(frame: np.ndarray, centre: Tuple[float, float],
                   half: int = ROI_HALF, scale: int = ROI_SCALE) -> Optional[Pose]:
    """The arrow's pose, with the direction measured on a `scale`x upsampled ROI.

    `centre` is where the arrow is (hsv_marker's answer). This does not search the frame for
    candidates - that is the caller's job - it looks only inside the ROI, which is what makes
    the x4 upsample affordable.
    """
    if frame is None or frame.size == 0:
        return None
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    x0, y0 = cx - half, cy - half
    x1, y1 = cx + half, cy + half
    if x0 < 0 or y0 < 0 or x1 > frame.shape[1] or y1 > frame.shape[0]:
        return None
    crop = frame[y0:y1, x0:x1]
    up = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    hsv = cv2.cvtColor(up, cv2.COLOR_BGR2HSV)
    mask = cv2.morphologyEx(_mask(hsv), cv2.MORPH_OPEN,
                            np.ones((2 * scale, 2 * scale), np.uint8))
    _n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    pick = None
    for i in range(1, len(stats)):
        area = int(stats[i, cv2.CC_STAT_AREA])
        # The same area window as the frame-level filter, scaled with the image.
        if area < MIN_AREA * scale * scale or area > MAX_AREA * scale * scale:
            continue
        if pick is None or area > pick[1]:
            pick = (i, area)
    if pick is None:
        return None

    i, area = pick
    blob = (labels == i).astype(np.uint8)
    x, y, w, h = (int(stats[i, c]) for c in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                             cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    return Pose(centre=(x0 + float(centroids[i][0]) / scale, y0 + float(centroids[i][1]) / scale),
                heading=_heading(blob), area=int(area / (scale * scale)),
                bbox=(x0 + x // scale, y0 + y // scale,
                      max(1, w // scale), max(1, h // scale)),
                aspect=float(max(w, h) / max(min(w, h), 1)))


def detect(frame: np.ndarray, prefer_near: Optional[Tuple[float, float]] = None) -> Optional[Pose]:
    """The arrow, or None. `prefer_near` is the last position, which wins ties."""
    if frame is None or frame.size == 0:
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if frame.ndim == 3 else None
    if hsv is None:
        return None
    mask = cv2.morphologyEx(_mask(hsv), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    _n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    candidates = []
    for i in range(1, len(stats)):
        x, y, w, h, area = stats[i]
        if area < MIN_AREA or area > MAX_AREA:
            continue                      # excludes the static red object at ~1600 px
        aspect = max(w, h) / max(min(w, h), 1.0)
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            continue
        candidates.append((i, x, y, w, h, area, aspect))
    if not candidates:
        return None

    if prefer_near is not None:
        chosen = min(candidates, key=lambda c: np.hypot(centroids[c[0]][0] - prefer_near[0],
                                                        centroids[c[0]][1] - prefer_near[1]))
    else:
        chosen = max(candidates, key=lambda c: c[5])

    i, x, y, w, h, area, aspect = chosen
    blob = (labels == i).astype(np.uint8)
    return Pose(centre=(float(centroids[i][0]), float(centroids[i][1])),
                heading=_heading(blob), area=int(area),
                bbox=(int(x), int(y), int(w), int(h)), aspect=float(aspect))


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ARROW_REFERENCE = os.path.join(ROOT, "data", "arrow", "design", "white_arrow2.png")
_REF = None


def _reference_shape():
    """The arrow's outline, for matchShapes. Hu moments drop rotation and scale out."""
    global _REF
    if _REF is None:
        ref = cv2.imread(_ARROW_REFERENCE, cv2.IMREAD_GRAYSCALE)
        if ref is not None:
            _t, binary = cv2.threshold(ref, 127, 255, cv2.THRESH_BINARY_INV)
            contours, _h = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                _REF = max(contours, key=cv2.contourArea)
    return _REF


def detect_in_box(frame: np.ndarray, box, pad: float = 0.15) -> Optional[Pose]:
    """The arrow inside a known car box, where the shape filters can be relaxed.

    The aspect filter exists to choose between red blobs across a whole frame. Inside a box
    that already contains the car there is nothing to choose, and the filter actively hurts:
    an arrow turned 45 degrees has a nearly square bounding box (measured 26x28, aspect 1.08)
    and was rejected by MIN_ASPECT. Rotation cannot be filtered by an extent ratio - that is
    what the PCA axis is for - so here only the area is checked.
    """
    x, y, w, h = [int(v) for v in box]
    m = int(pad * max(w, h))
    x0, y0 = max(0, x - m), max(0, y - m)
    crop = frame[y0:y + h + m, x0:x + w + m]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.morphologyEx(_mask(hsv), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    _n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    # Which red blob, when a car box can contain the arrow AND the bay lights? Shape, not
    # proximity: picking the blob nearest the box centre let an LED win often enough that the
    # heading swung through 356 degrees over twenty frames while the car stood still.
    reference = _reference_shape()
    cx_box, cy_box = crop.shape[1] / 2.0, crop.shape[0] / 2.0
    best = None
    for i in range(1, len(stats)):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_AREA // 3 or area > MAX_AREA:
            continue
        # Three filters together, because any one of them alone let a bay light through:
        #   size      the arrow is 150..300 px here; an LED blob is smaller
        #   shape     Hu-moment distance to the printed arrow, and it must be close
        #   position  the arrow is on the roof, so near the box's centre; an LED sits at the
        #             bay's edge and lands in the corner of the box
        dist = float(np.hypot(centroids[i][0] - cx_box, centroids[i][1] - cy_box))
        if dist > 0.45 * max(crop.shape[:2]):
            continue
        blob_i = (labels == i).astype(np.uint8)
        contours, _h = cv2.findContours(blob_i * 255, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        d = float(cv2.matchShapes(reference, contour, cv2.CONTOURS_MATCH_I1, 0.0)) \
            if reference is not None else 0.0
        if d > 0.30:
            continue
        if best is None or d < best[0]:
            best = (d, i)
    if best is None:
        return None
    i = best[1]
    blob = (labels == i).astype(np.uint8)
    x1, y1, w1, h1, area = stats[i]
    return Pose(centre=(float(centroids[i][0] + x0), float(centroids[i][1] + y0)),
                heading=_heading(blob), area=int(area),
                bbox=(int(x1 + x0), int(y1 + y0), int(w1), int(h1)),
                aspect=float(max(w1, h1) / max(min(w1, h1), 1.0)))
