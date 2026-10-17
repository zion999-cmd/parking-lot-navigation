"""The candidate hero-car mask, kept OUT of the product and injected at run time.

Why this file exists rather than a patch to src/vehicle.py:

The product's mask comes from ``vehicle.red_mask()``. The two changes here are hypotheses about a
better mask, and a hypothesis does not belong in the recognition chain until it has beaten what is
there, on frames it was not tuned on. So the candidate lives here and ``mask_provider()`` swaps it in
for one experiment process:

    with mask_provider(candidate_red_mask):
        poses = vehicle.locate_all(frame)          # vehicle.py is not modified, not even at import

That is also the compatibility test: if a candidate mask can be consumed by the unmodified feature
stage, the injection point is real and nothing downstream has to know which mask it got.

Two changes, both with measurements behind them (tools/hero_lab/README.md):

  B1  area, not shape, separates the bay lights from the paint
  B2  the paint's dark side is found with a near-field statistic instead of a whole-ray median

Measured against 91 hand-drawn quads: IoU 0.504 -> 0.675, centre error 17.2 -> 4.7 px, axis error
p90 68 -> 17 deg, and the left/right gap (0.548 against 0.414) closed to 0.683 against 0.656.
The cost was 195 ms/frame against the product's 72.7, which is over the frame interval and the
reason for the rewrite in ray_levels(): same statistic, computed once per direction on a crop
instead of ~120 times on the full frame.
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

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
# it fails: that ray is 111 px long and 31 px wide and the black cabin occupies its first 25 px, so
# the median lands on the road at 131 while the cabin is at 50 - a 79-level signal read as 16, below
# the 40 threshold, and the paint was left on its own. The scanline through the same car reads
# 59 53 52 52 50 49 52 54 for the cabin against 155-168 for the road.
#
# ray_levels() is that statistic, computed the cheap way. The first version called _ray_mask() for
# every segment of every direction - about 120 filled rectangles on the full 800x600 frame, each
# followed by a full-frame boolean - and cost 195 ms per frame, which is over the frame interval.
# Here the rays are generated directly in the rotated frame and the value channel is indexed once,
# on a crop around the paint, and every direction is measured in the same pass.

_CROP_STRIDE = 2          # sample every other pixel along the ray: ~4x fewer, same median
_MIN_SAMPLES = 24         # below this a segment is treated as absent


def _crop_box(cx: float, cy: float, reach: int, half: float, shape: Sequence[int]):
    pad = int(reach + 1.5 * half) + 2
    h, w = int(shape[0]), int(shape[1])
    x0 = max(0, int(cx) - pad)
    y0 = max(0, int(cy) - pad)
    x1 = min(w, int(cx) + pad + 1)
    y1 = min(h, int(cy) + pad + 1)
    return x0, y0, x1, y1


def ray_levels(value: np.ndarray, blob: np.ndarray, cx: float, cy: float,
               directions: Sequence[float], gap: int, reach: int, half: float,
               segments: int = 4) -> Dict[float, float]:
    """Per direction, the median of the darkest segment just beyond the paint.

    Same statistic as the product's whole-ray median except that it is taken *close in* and per
    segment, which is what stops the cabin being averaged into the road behind it.
    """
    x0, y0, x1, y1 = _crop_box(cx, cy, reach, half, value.shape)
    crop = value[y0:y1, x0:x1]
    in_blob = blob[y0:y1, x0:x1] > 0
    h, w = crop.shape
    step = max(reach // segments, 4)
    local_x, local_y = cx - x0, cy - y0
    out: Dict[float, float] = {}

    for angle in directions:
        rad = math.radians(angle)
        ux, uy = math.cos(rad), math.sin(rad)
        nx, ny = -uy, ux
        best = None
        for k in range(segments):
            t_lo = gap + k * step
            ts = np.arange(t_lo, t_lo + step, _CROP_STRIDE, dtype=np.float32)
            ss = np.arange(-half, half + 1e-3, _CROP_STRIDE, dtype=np.float32)
            if ts.size == 0 or ss.size == 0:
                continue
            grid_t, grid_s = np.meshgrid(ts, ss, indexing="ij")
            px = np.rint(local_x + ux * grid_t + nx * grid_s).astype(np.int32)
            py = np.rint(local_y + uy * grid_t + ny * grid_s).astype(np.int32)
            inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            if not inside.any():
                continue
            px, py = px[inside], py[inside]
            outside_paint = ~in_blob[py, px]
            if int(outside_paint.sum()) < _MIN_SAMPLES:
                continue
            level = float(np.median(crop[py, px][outside_paint]))
            best = level if best is None else min(best, level)
        if best is not None:
            out[angle] = best
    return out


def _reference_gap(levels: Dict[float, float], best: float) -> Optional[float]:
    """How much darker `best` is than the directions away from it."""
    others = [v for a, v in levels.items()
              if vehicle._ray_gap(a, best) > vehicle._RAY_CONE_DEG]
    if not others:
        return None
    return float(np.median(others)) - levels[best]


def paint_side_and_levels(mask: np.ndarray, value: np.ndarray, gap: int = 6, reach: int = 0,
                          min_drop: int = 40) -> Tuple[Optional[float], Dict[float, float],
                                                        float, float, int]:
    """Which way from the paint it is dark, plus everything the caller would else recompute.

    Returns (direction, levels, half, paint median, reach) - the last three so that the attach step
    does not measure the same rays a second time. That second pass was most of the 195 ms.
    """
    blob, _ = vehicle._largest_component(mask)
    ys, xs = np.nonzero(blob)
    if ys.size == 0:
        return None, {}, 0.0, 0.0, 0
    cx, cy = float(xs.mean()), float(ys.mean())
    half = vehicle._paint_half_width(blob)
    reach = vehicle._reach_for(reach, int(xs.max() - xs.min()), int(ys.max() - ys.min()))
    directions = [360.0 * k / vehicle._RAY_COUNT for k in range(vehicle._RAY_COUNT)]
    levels = ray_levels(value, blob, cx, cy, directions, gap, reach, half)
    if len(levels) < vehicle._RAY_COUNT // 2:
        return None, levels, half, float(np.median(value[blob > 0])), reach
    best = min(levels, key=lambda a: levels[a])
    gap_to_road = _reference_gap(levels, best)
    if gap_to_road is None or gap_to_road < min_drop:
        return None, levels, half, float(np.median(value[blob > 0])), reach
    return best, levels, half, float(np.median(value[blob > 0])), reach


def candidate_paint_side(mask: np.ndarray, value: np.ndarray, gap: int = 6, reach: int = 0,
                         min_drop: int = 40, near_field: bool = True) -> Optional[float]:
    """Compatibility wrapper: the direction only.

    With near_field=False this reproduces the product's whole-ray statistic, which the copy check in
    compare.py uses to prove the copy has not drifted.
    """
    if near_field:
        direction, _levels, _half, _median, _reach = paint_side_and_levels(
            mask, value, gap, reach, min_drop)
        return direction
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
        ray = vehicle._ray_mask(value.shape, cx, cy, angle, gap, reach, half)
        sample = value[(ray > 0) & (blob == 0)]
        if sample.size >= 50:
            levels[angle] = float(np.median(sample))
    if len(levels) < vehicle._RAY_COUNT // 2:
        return None
    best = min(levels, key=lambda a: levels[a])
    gap_to_road = _reference_gap(levels, best)
    if gap_to_road is None or gap_to_road < min_drop:
        return None
    return best


def candidate_attach_paint_side(mask: np.ndarray, value: np.ndarray, angle: float, gap: int = 6,
                                reach: int = 0, min_drop: int = 40,
                                near_field: bool = True,
                                levels: Optional[Dict[float, float]] = None) -> np.ndarray:
    """The paint plus the dark region beyond it in direction `angle`, kept to the car's own width.

    The cut is the midpoint between what lies along the car's own direction and what the road reads,
    both measured the same way, close in. Cutting against the paint's brightness instead sits above
    the road on this camera - the road reads 115-138 where the paint is 216 - and would swallow it.
    """
    blob, _ = vehicle._largest_component(mask)
    ys, xs = np.nonzero(blob)
    if ys.size == 0:
        return mask
    cx, cy = float(xs.mean()), float(ys.mean())
    half = vehicle._paint_half_width(blob)
    reach = vehicle._reach_for(reach, int(xs.max() - xs.min()), int(ys.max() - ys.min()))

    probe = vehicle._ray_mask(value.shape, cx, cy, angle, gap, reach, half * 1.5)
    here = (probe > 0) & (blob == 0)
    if not here.any():
        return mask

    if near_field:
        if levels is None:
            levels = ray_levels(value, blob, cx, cy,
                                [360.0 * k / vehicle._RAY_COUNT
                                 for k in range(vehicle._RAY_COUNT)], gap, reach, half)
        along_level = levels.get(angle)
        others = [v for a, v in levels.items()
                  if vehicle._ray_gap(a, angle) > vehicle._RAY_CONE_DEG]
        if along_level is None or not others:
            return mask
        cut = 0.5 * (along_level + float(np.median(others)))
    else:
        paint_v = float(np.median(value[blob > 0]))
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
        gap = int(cfg.get("paint_side_gap", 6))
        reach = int(cfg.get("paint_side_reach", 0))
        min_drop = int(cfg.get("paint_side_min_drop", 40))
        if near_field:
            side, levels, _half, _median, _reach = paint_side_and_levels(
                mask, hsv[:, :, 2], gap, reach, min_drop)
        else:
            # Both switches off must reproduce the product *including* how the direction is found -
            # the first version of this rewrite got that wrong and the copy check caught it: 1,229
            # differing pixels on one saved frame, because the direction still came from the near
            # field statistic while everything downstream pretended to be the product.
            side, levels = candidate_paint_side(mask, hsv[:, :, 2], gap, reach, min_drop,
                                                near_field=False), None
        if side is not None:
            mask = candidate_attach_paint_side(
                mask, hsv[:, :, 2], side, int(cfg.get("paint_side_gap", 6)),
                int(cfg.get("paint_side_reach", 0)), int(cfg.get("paint_side_min_drop", 40)),
                near_field, levels if near_field else None)

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
