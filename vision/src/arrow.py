"""Find the hero car by the arrow taped to its roof.

Why this replaced the colour pipeline: the old detector looked for the yellow paint,
then guessed which end was the front from which side was dark. It needed the paint
to be visible, the lighting to behave, and the guess to be right - and it still put
the box on a bay-light column in 8 of 91 hand labels.

The arrow answers all three at once, because a person decided what it means: its
position is the car's position and its direction is the car's direction. Measured on
this board with one hand annotation as the template:

    score at the arrow      0.993        (four independent live frames)
    best wrong match        0.699        (elsewhere in the same frame)
    angle separation        0.993 at the true angle, <= 0.765 at every other

The cost is the reason for the structure here. A 24-angle sweep of the whole 800x600
frame measured 1178 ms, which is nineteen times the 66 ms budget at 15 fps. Searching
a 140x140 window around the yellow paint instead costs 43 ms, and a coarse 30 degree
sweep followed by a fine one around the winner brings it to roughly 30 ms, so it fits.

The template is a committed file (data/arrow/template.png + mask.png + meta.json),
not a live annotation: the camera's geometry does not change, and a dataset directory
is not something a running detector should depend on.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARROW_DIR = os.path.join(ROOT, "data", "arrow")

# Below this a match is not the arrow. 0.85 sits between the measured 0.993 (true) and
# 0.765 (the highest wrong angle) with room on both sides.
# Below this a match is not the arrow.
#
# It was 0.85, chosen from the angle sweep (0.993 true, 0.765 the highest wrong angle). Two
# measurements moved it: a car standing 250 px from where the template was cut scores 0.814 -
# the perspective cost - and the road-patch false positive that argued against lowering it
# sits at x=738, outside the region vehicle.py trusts (x 60..700), so it can no longer win.
MIN_SCORE = 0.78

# Search window around the paint blob, in pixels. The arrow sits on the car, and the
# paint blob is on the car, so this only has to be larger than the car.
ROI_HALF = 95

# Coarse sweep, then refine. 30 degrees is coarse enough to fit the budget and fine
# enough that the true angle is never more than 15 degrees away.
COARSE_STEP = 30.0
FINE_STEP = 5.0
FINE_SPAN = 15.0


@dataclass
class Match:
    score: float
    centre: Tuple[float, float]     # image pixels
    heading: float                  # 0..360, direction the arrow points
    corners: List[Tuple[float, float]]   # rotated template box, for drawing


class _Template:
    def __init__(self, patch: np.ndarray, mask: np.ndarray, reference_angle: float):
        self.patch = patch
        self.mask = mask
        self.reference_angle = reference_angle
        fill = float(patch[mask > 0].mean())
        h, w = patch.shape
        self.diag = int(np.ceil(np.hypot(h, w))) + 6
        self.fill = fill
        self._cache = {}

    def at(self, deg: float):
        """Rotated template and its mask, cached. Rotation by `deg` is applied to the patch."""
        key = round(float(deg) % 360.0, 3)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        d = self.diag
        canvas = np.full((d, d), self.fill, np.float32)
        mcan = np.zeros((d, d), np.uint8)
        h, w = self.patch.shape
        y0, x0 = (d - h) // 2, (d - w) // 2
        canvas[y0:y0 + h, x0:x0 + w] = self.patch
        mcan[y0:y0 + h, x0:x0 + w] = self.mask
        m = cv2.getRotationMatrix2D((d / 2.0, d / 2.0), key, 1.0)
        rot = cv2.warpAffine(canvas, m, (d, d)).astype(np.float32)
        mrot = cv2.warpAffine(mcan, m, (d, d), flags=cv2.INTER_NEAREST)
        rot[mrot == 0] = self.fill
        out = (rot, (mrot > 0).astype(np.uint8) * 255)
        self._cache[key] = out
        return out


def load(directory: str = ARROW_DIR) -> _Template:
    patch = cv2.imread(os.path.join(directory, "template.png"), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(os.path.join(directory, "mask.png"), cv2.IMREAD_GRAYSCALE)
    if patch is None or mask is None:
        raise FileNotFoundError("arrow template missing in %s" % directory)
    meta_path = os.path.join(directory, "meta.json")
    reference = 0.0
    if os.path.exists(meta_path):
        try:
            reference = float(json.load(open(meta_path, encoding="utf-8"))["reference_angle_deg"])
        except (ValueError, KeyError, TypeError):
            reference = 0.0
    return _Template(patch.astype(np.float32), mask, reference)


def find_in(grey: np.ndarray, template: _Template, origin: Tuple[int, int] = (0, 0),
            coarse: float = COARSE_STEP, fine: float = FINE_STEP,
            span: float = FINE_SPAN) -> Optional[Match]:
    """Search one (possibly cropped) grey frame. `origin` is the crop's top-left in the parent."""
    img = grey.astype(np.float32)
    best = (-2.0, 0.0, None)
    deg = 0.0
    while deg < 360.0:
        t, m = template.at(deg)
        if t.shape[0] >= img.shape[0] or t.shape[1] >= img.shape[1]:
            deg += coarse
            continue
        res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED, mask=m)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx > best[0]:
            best = (float(mx), deg, loc)
        deg += coarse
    if best[2] is None:
        return None

    # Refine around the coarse winner: rotations are not fine enough at 30 degrees.
    fine_best = best
    fine_res = None
    d = best[1] - span
    while d <= best[1] + span:
        t, m = template.at(d)
        if t.shape[0] < img.shape[0] and t.shape[1] < img.shape[1]:
            res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED, mask=m)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if mx > fine_best[0]:
                fine_best = (float(mx), d, loc)
                fine_res = res
        d += fine
    score, deg, loc = fine_best

    # Sub-degree: the fine sweep is 5 degrees apart, so the reported heading used to jump in
    # 5 degree steps and the drawn car stepped with it. The score curve near the peak is
    # smooth, so a parabola through (deg-fine, deg, deg+fine) recovers the fraction. Cheap:
    # two extra correlations.
    if fine_res is not None:
        below, _ = template.at(deg - fine)
        above, _ = template.at(deg + fine)
        if (below.shape[0] < img.shape[0] and below.shape[1] < img.shape[1]
                and above.shape[0] < img.shape[0] and above.shape[1] < img.shape[1]):
            s_below = cv2.minMaxLoc(cv2.matchTemplate(
                img, below, cv2.TM_CCOEFF_NORMED, mask=template.at(deg - fine)[1]))[1]
            s_above = cv2.minMaxLoc(cv2.matchTemplate(
                img, above, cv2.TM_CCOEFF_NORMED, mask=template.at(deg + fine)[1]))[1]
            den = s_below - 2.0 * score + s_above
            if den != 0.0:
                step = 0.5 * (s_below - s_above) / den
                if abs(step) <= 0.5:
                    deg += step * fine
    t, _m = template.at(deg)
    h, w = t.shape
    # Sub-pixel: the match lands on an integer pixel, so the car advanced in 1 px steps
    # even though the picture moved continuously. A parabola through the correlation peak
    # and its two neighbours recovers the fraction, which is what makes the marker glide.
    dx = dy = 0.0
    if fine_res is not None:
        px, py = loc
        if 0 < px < fine_res.shape[1] - 1:
            a, b, c = fine_res[py, px - 1], fine_res[py, px], fine_res[py, px + 1]
            den = a - 2.0 * b + c
            if den != 0.0:
                dx = max(-0.5, min(0.5, 0.5 * (a - c) / den))
        if 0 < py < fine_res.shape[0] - 1:
            a, b, c = fine_res[py - 1, px], fine_res[py, px], fine_res[py + 1, px]
            den = a - 2.0 * b + c
            if den != 0.0:
                dy = max(-0.5, min(0.5, 0.5 * (a - c) / den))
    cx = origin[0] + loc[0] + dx + w / 2.0
    cy = origin[1] + loc[1] + dy + h / 2.0
    # The template was cut with the arrow pointing along `reference_angle`; a match at
    # `deg` means the arrow is at reference - deg (OpenCV rotates counter-clockwise).
    heading = (template.reference_angle - deg) % 360.0
    a = np.deg2rad(-deg)
    ca, sa = np.cos(a), np.sin(a)
    corners = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        corners.append((cx + dx * ca - dy * sa, cy + dx * sa + dy * ca))
    return Match(score=score, centre=(cx, cy), heading=heading, corners=corners)


def locate(frame: np.ndarray, template: Optional[_Template] = None,
           roi_hint: Optional[Tuple[int, int]] = None,
           min_score: float = MIN_SCORE) -> Optional[Match]:
    """Find the arrow in a BGR frame, searching a window around `roi_hint` if given."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tpl = template if template is not None else load()
    if roi_hint is not None:
        cx, cy = int(roi_hint[0]), int(roi_hint[1])
        x0 = max(0, cx - ROI_HALF)
        y0 = max(0, cy - ROI_HALF)
        crop = grey[y0:y0 + 2 * ROI_HALF, x0:x0 + 2 * ROI_HALF]
        m = find_in(crop, tpl, origin=(x0, y0))
    else:
        m = find_in(grey, tpl)
    if m is None or m.score < min_score:
        return None
    return m


# ---------------------------------------------------------------------------------------------
# Perspective-tolerant matching.
#
# A flat template matches a flat marker only while the camera sees it the same way. Measured:
# the template, cut where the marker was, scored 1.000 there and 0.784 after the car moved
# 298 px across the table - and the marker's white square measured 60x46 px at the new place,
# i.e. an aspect of 1.30 instead of 1.00. That is the camera looking at the roof from a
# different angle, not a printing problem.
#
# The fix is what a fiducial detector does and what the marker's white square makes possible:
# find the square's four corners, warp it to a canonical square, and match the template inside
# that. The warp removes the perspective; the arrow inside still points where the car points.
# ---------------------------------------------------------------------------------------------

def _quad_candidates(grey, min_side=22, max_side=110):
    """Four-cornered, roughly square, bright regions: the marker's white paper."""
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _h = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_side * min_side * 0.5 or area > max_side * max_side * 2.0:
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        quad = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(quad) != 4:
            continue
        pts = quad.reshape(4, 2).astype(np.float32)
        sides = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]
        if min(sides) < min_side or max(sides) > max_side:
            continue
        if max(sides) / max(min(sides), 1e-6) > 1.6:
            continue
        hull = cv2.convexHull(pts)
        if cv2.contourArea(hull) <= 0 or area / cv2.contourArea(hull) < 0.80:
            continue
        out.append(pts)
    return out


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Corners in a stable order: sum and difference of the coordinates."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
                    np.float32)


def find_perspective(grey: np.ndarray, template: "_Template", side: int = 40,
                     min_score: float = MIN_SCORE):
    """Warp each candidate square to `side` px and match the template inside it.

    All four rotations of the corner order are tried, because contours have no idea which
    corner is the marker's top-left; the asymmetric arrow inside decides.
    """
    best = None
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], np.float32)
    for quad in _quad_candidates(grey):
        ordered = _order_quad(quad)
        for shift in range(4):
            src = np.roll(ordered, -shift, axis=0)
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(grey, M, (side, side), flags=cv2.INTER_LINEAR)
            tpl_warped = _Template(warped.astype(np.float32), np.full((side, side), 255, np.uint8), 0.0)
            # The template lives in the marker's own frame, so match it against the warp.
            m = find_in(warped.astype(np.float32), template, origin=(0, 0))
            if m is None:
                continue
            if best is None or m.score > best.score:
                # Map the match centre back to image coordinates.
                inv = np.linalg.inv(M)
                p = inv @ np.array([m.centre[0], m.centre[1], 1.0])
                p = p[:2] / p[2]
                best = Match(score=m.score, centre=(float(p[0]), float(p[1])),
                             heading=m.heading, corners=[tuple(v) for v in ordered])
    if best is None or best.score < min_score:
        return None
    return best
