"""The hero car by an ArUco marker, which is what the printed arrow could not be.

Why this exists: template matching of a hand-drawn or printed arrow kept failing in ways
that were all the same thing - a flat template has to match a surface whose projection
depends on the heading. Measured on this camera: the same marker scored 0.997 facing the
way it was cut and 0.757 turned around, because the roof around it foreshortens with the
car. ArUco is built for exactly that problem: it detects a bit pattern through rotation,
scale, perspective and lighting, and it reports the four corners in the marker's own
order.

The direction comes from that order, and it is the reason this is a better marker than an
arrow:

    corner 0 -> the marker's top-left, 1 -> top-right, 2 -> bottom-right, 3 -> bottom-left

Turn the car and corner 0 moves on screen but stays the marker's top-left, so the direction
of the marker's top edge is the car's heading with no ambiguity and no averaging. There is
no 180-degree guess anywhere in this file.

Sizing, measured with the lot's 1 px ~= 1.3 mm: a DICT_4X4 marker is 6 modules across, so
35 mm prints to ~27 px (4.5 px per module - too fine) and 50 mm to ~38 px (6.4 per module,
usable). The PDF in data/arrow/design/ has both.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Whichever dictionary the marker was printed from. DICT_4X4_50 is the smallest that fits a
# roof at this resolution; the id is fixed by the PDF in data/arrow/design/.
DICT_ID = int(os.environ.get("HERO_ARUCO_DICT", cv2.aruco.DICT_4X4_50))
MARKER_ID = int(os.environ.get("HERO_ARUCO_ID", 7))


@dataclass
class Marker:
    score: float                  # 0..1: 1 when all four sides were found
    centre: Tuple[float, float]
    heading: float                # 0..360, direction the marker's top edge points
    corners: list                 # four (x, y) in image pixels, marker order


def _detector():
    dictionary = cv2.aruco.getPredefinedDictionary(DICT_ID)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    return cv2.aruco.ArucoDetector(dictionary, params)


_DETECTOR = None


def detect(frame: np.ndarray, wanted: int = MARKER_ID) -> Optional[Marker]:
    """Find the marker in a BGR frame. None when it is not there.

    A whole-frame detectMarkers() costs about 10-20 ms on 800x600, so unlike the template
    matcher this needs no region-of-interest tracking and no recovery sweep.
    """
    global _DETECTOR
    if frame is None or frame.size == 0:
        return None
    if _DETECTOR is None:
        _DETECTOR = _detector()

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    corners, ids, _rejected = _DETECTOR.detectMarkers(grey)
    if ids is None or len(ids) == 0:
        return None

    best = None
    for quad, mid in zip(corners, ids.ravel()):
        if int(mid) != wanted:
            continue
        pts = quad.reshape(-1, 2)
        if pts.shape != (4, 2):
            continue
        best = pts
        break
    if best is None:
        return None

    # top edge = corner 0 -> corner 1. Its direction is the marker's, and therefore the car's.
    top = best[1] - best[0]
    heading = float(np.degrees(np.arctan2(top[1], top[0])) % 360.0)
    cx = float(best[:, 0].mean())
    cy = float(best[:, 1].mean())
    # Side-length ratio as a crude confidence: a real quad has four similar sides.
    sides = [float(np.linalg.norm(best[(i + 1) % 4] - best[i])) for i in range(4)]
    score = 1.0 if min(sides) > 0 else 0.0
    if score and max(sides) / min(sides) > 2.0:
        score = 0.5                      # visibly not a square: probably a false positive
    return Marker(score=score, centre=(cx, cy), heading=heading,
                  corners=[(float(x), float(y)) for x, y in best])
