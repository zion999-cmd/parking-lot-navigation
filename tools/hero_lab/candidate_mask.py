"""The candidate hero-car mask, kept OUT of the product and injected at run time.

Why this file exists rather than a patch to src/vehicle.py:

The product's mask is produced by ``vehicle.red_mask()``. The two changes explored here are
hypotheses about a better mask, and a hypothesis does not belong in the recognition chain until it
has beaten what is there now, on frames it was not tuned on. So the candidate lives here, and
``mask_provider()`` swaps it in for the duration of one experiment process:

    with mask_provider(candidate_red_mask):
        poses = vehicle.locate_all(frame)          # vehicle.py is not modified, not even at import

That is also the compatibility test. If a candidate mask can be consumed by the unmodified feature
stage, then the injection point is real and the contract downstream never has to know which mask it
got. If it needs the feature stage changed, it is not a mask candidate at all.

Run it through compare.py, which reports production against candidate on the same frames. Nothing
here imports this module; nothing here may be imported by anything under vision/.

Two changes, both with measurements behind them (see README.md):

  B1  area, not shape, separates the bay lights from the paint
  B2  the paint's dark side is found with a near-field statistic instead of a whole-ray median
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
from typing import Iterator, Optional

import cv2
import numpy as np

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vision", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import vehicle                                                        # noqa: E402

# ---------------------------------------------------------------------------------------------
# B1: the bay lights are removed by area.
#
# Measured live over 90 frames: the car's paint is the largest blob every frame and runs 421 to
# 933 px; every other yellow blob - the bay lights along the strip, the floor markings - runs 17 to
# 285 px. A gap with nothing in it, so one area threshold separates them and 300 sits inside it.
#
# Shape was tried first and measured to be the wrong handle: a "small AND thin" rule (aspect >= 3)
# removed one blob of ten, because the lights are 5x16 to 6x30 but the floor markings are 5x8 to
# 16x9 - small and square. Colour cannot do it either: lights read H 23-28 S 90-114 V 131-142 and
# the paint H 23-33 S 90-197 V 70-195, so the ranges overlap completely.
#
# It matters because vehicle.merge_near_components groups pieces within 2 * median_size of each
# other and then draws a 3 px line between the group's centroids: with the lights in the mask, the
# car and the lights land in one group and that line is the bridge. Measured on the saved
# right-hand frame, the mask became an 87x188 px column, 30 of 40 frames were refused by the border
# filter, and the rest came back with no axis at all.
MIN_PAINT_AREA = 300


def drop_small_blobs(mask: np.ndarray, min_area: int = MIN_PAINT_AREA) -> np.ndarray:
    """Keep only components plausibly big enough to be the car's paint.

    Never returns an empty mask: if nothing clears the threshold the largest component is kept.
    Turning a visible car into no car would be worse than keeping one light, and the paint shrinks
    when the car is further away or partly hidden - below a threshold that was measured on frames
    where it was not.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return mask
    keep = np.zeros_like(mask)
    for i in range(1, count):
        if int(stats[i, 4]) > min_area:
            keep[labels == i] = 255
    if not keep.any():
        keep[labels == 1 + int(np.argmax(stats[1:, 4]))] = 255
    return keep


# ---------------------------------------------------------------------------------------------
# B2: the dark side of the paint, measured close in.
#
# One median over the whole ray is what the product does, and the saved right-hand frame shows why
# it fails. That ray is 111 px long and 31 px wide and the black cabin occupies its first 25 px, so
# the median lands on the road at 131 while the cabin is at 50: a 79-level signal read as 16, below
# the 40 threshold, and the paint was left on its own. The scanline through the same car reads
# 59 53 52 52 50 49 52 54 for the cabin against 155-168 for the road.
def ray_near_level(value: np.ndarray, blob: np.ndarray, cx: float, cy: float, angle: float,
                   gap: int, reach: int, half: float, segments: int = 4) -> Optional[float]:
    """The darkest segment median along one ray out of the paint, or None."""
    step = max(reach // segments, 4)
    best = None
    for k in range(segments):
        segment = vehicle._ray_mask(value.shape, cx, cy, angle, gap + k * step, step, half)
        sel = (segment > 0) & (blob == 0)
        if int(np.count_nonzero(sel)) < 40:
            continue
        level = float(np.median(value[sel]))
        best = level if best is None else min(best, level)
    return best


def candidate_paint_side(mask: np.ndarray, value: np.ndarray, gap: int = 6, reach: int = 0,
                         min_drop: int = 40, near_field: bool = True) -> Optional[float]:
    """A copy of vehicle.paint_side() with the statistic swapped when near_field is on."""
    blob, _ = vehicle._largest_component(mask)
    ys, xs = np.nonzero(blob)
    if ys.size == 0:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    half = vehicle._paint_half_width(blob)
    reach = vehicle._reach_for(reach, int(xs.max() - xs.min()), int(ys.max() - ys.min()))

    levels = {}
    for k in range(vehicle._RAY_COUNT):
        angle = 360.0 * k / vehicle._RAY_COUNT
        if near_field:
            level = ray_near_level(value, blob, cx, cy, angle, gap, reach, half)
        else:
            ray = vehicle._ray_mask(value.shape, cx, cy, angle, gap, reach, half)
            sample = value[(ray > 0) & (blob == 0)]
            level = float(np.median(sample)) if sample.size >= 50 else None
        if level is not None:
            levels[angle] = level

    if len(levels) < vehicle._RAY_COUNT // 2:
        return None
    best = min(levels, key=lambda a: levels[a])
    others = [v for a, v in levels.items() if vehicle._ray_gap(a, best) > vehicle._RAY_CONE_DEG]
    if not others:
        return None
    if float(np.median(others)) - levels[best] < min_drop:
        return None
    return best


def candidate_attach_paint_side(mask: np.ndarray, value: np.ndarray, angle: float, gap: int = 6,
                                reach: int = 0, min_drop: int = 40,
                                near_field: bool = True) -> np.ndarray:
    """A copy of vehicle.attach_paint_side() with a tighter cut when near_field is on.

    The cut is the midpoint between what lies along the car's own direction and what the road
    reads, both measured the same way. Cutting against the paint's own brightness instead sits
    above the road on this camera - the road reads 115-138 where the paint is 216 - and swallows it.
    """
    blob, _ = vehicle._largest_component(mask)
    ys, xs = np.nonzero(blob)
    if ys.size == 0:
        return mask
    cx, cy = float(xs.mean()), float(ys.mean())
    half = vehicle._paint_half_width(blob)
    reach = vehicle._reach_for(reach, int(xs.max() - xs.min()), int(ys.max() - ys.min()))
    paint_v = float(np.median(value[blob > 0]))

    probe = vehicle._ray_mask(value.shape, cx, cy, angle, gap, reach, half * 1.5)
    here = (probe > 0) & (blob == 0)
    if not here.any():
        return mask

    if near_field:
        along_level = ray_near_level(value, blob, cx, cy, angle, gap, reach, half)
        others = []
        for k in range(vehicle._RAY_COUNT):
            other = 360.0 * k / vehicle._RAY_COUNT
            if vehicle._ray_gap(other, angle) <= vehicle._RAY_CONE_DEG:
                continue
            level = ray_near_level(value, blob, cx, cy, other, gap, reach, half)
            if level is not None:
                others.append(level)
        if along_level is None or not others:
            return mask
        cut = 0.5 * (along_level + float(np.median(others)))
    else:
        cut = 0.5 * (paint_v + float(np.median(value[here])))

    dark = here & (value <= cut)
    if not dark.any():
        return mask

    dark_ys, dark_xs = np.nonzero(dark)
    axis = math.degrees(math.atan2(float(dark_ys.mean()) - cy, float(dark_xs.mean()) - cx))
    band = vehicle._ray_mask(value.shape, cx, cy, axis, gap, reach, half * 1.15)
    keep = dark & (band > 0)
    if not keep.any():
        return mask

    near = cv2.dilate(blob, np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(keep.astype(np.uint8) * 255, 8)
    touching = None
    for i in range(1, count):
        component = labels == i
        if (component & (near > 0)).any():
            if touching is None or stats[i, 4] > stats[touching, 4]:
                touching = i
    if touching is None:
        return mask
    grown = np.zeros_like(value)
    grown[labels == touching] = 255
    return cv2.bitwise_or(mask, grown)


# ---------------------------------------------------------------------------------------------


def candidate_red_mask(hsv: np.ndarray, close_kernel: Optional[int] = None,
                       open_kernel: Optional[int] = None, *, drop_small: bool = True,
                       near_field: bool = True,
                       min_paint_area: int = MIN_PAINT_AREA) -> np.ndarray:
    """A copy of vehicle.red_mask() with B1 and B2 behind switches.

    With both off it must reproduce the product's mask exactly - compare.py checks that on every
    frame it touches, because a copy that has quietly drifted away from the original would make
    every comparison meaningless.
    """
    cfg = vehicle.vehicle_config()
    if close_kernel is None:
        close_kernel = int(cfg["close_kernel"])
    if open_kernel is None:
        open_kernel = int(cfg["open_kernel"])

    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in cfg["hue_ranges"]:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    if open_kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vehicle._ellipse(open_kernel // 2))
    if drop_small:
        mask = drop_small_blobs(mask, min_paint_area)

    paint_area = float(mask.sum())
    paint_only = mask.copy()

    if cfg.get("attach_paint_side"):
        side = candidate_paint_side(mask, hsv[:, :, 2], int(cfg.get("paint_side_gap", 6)),
                                    int(cfg.get("paint_side_reach", 0)),
                                    int(cfg.get("paint_side_min_drop", 40)), near_field)
        if side is not None:
            mask = candidate_attach_paint_side(mask, hsv[:, :, 2], side,
                                               int(cfg.get("paint_side_gap", 6)),
                                               int(cfg.get("paint_side_reach", 0)),
                                               int(cfg.get("paint_side_min_drop", 40)), near_field)

    mask = vehicle.attach_dark(mask, hsv, int(cfg.get("attach_dark_margin_v", 0)),
                               int(cfg.get("attach_dark_max_s", 160)),
                               int(cfg.get("attach_dark_max_px", 60)))
    mask = vehicle.merge_near_components(mask, gap_fraction=0.9)

    holes = mask.copy()
    cv2.floodFill(holes, np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8), (0, 0), 255)
    full = mask | cv2.bitwise_not(holes)
    if full.sum() > vehicle.ATTACH_MAX_GROWTH * paint_area:
        return paint_only
    return full


@contextlib.contextmanager
def mask_provider(fn) -> Iterator[None]:
    """Let vehicle's feature stage see a different mask, in this process only.

    locate_all() calls red_mask(hsv, close_kernel) positionally, so the replacement has to accept
    the same two arguments. The product file is never written to.
    """
    original = vehicle.red_mask
    vehicle.red_mask = fn
    try:
        yield
    finally:
        vehicle.red_mask = original
