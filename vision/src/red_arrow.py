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
    best = None
    for i in range(1, len(stats)):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_AREA // 3 or area > MAX_AREA:
            continue
        # Prefer the blob nearest the box centre: inside a car box, that is the roof.
        d = np.hypot(centroids[i][0] - (crop.shape[1] / 2.0),
                     centroids[i][1] - (crop.shape[0] / 2.0))
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
