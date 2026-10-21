#!/usr/bin/env python3
"""Heading spike: PCA main axis, then six ways to decide which end is the front.

The axis is the easy part - the second moment of the arrow contour gives a line through it.
The hard part is that a line has two ends, and production's current method (compare the width
of the two end halves) was measured flipping a full 180 degrees under blur. This spike runs
several discriminators on the same contours and reports which of them agree with each other,
and how steady each one is, so the choice can be made on evidence.

Nothing here is imported by production. It reads frames from the same relay and uses the
same HSV+shape candidate as production, so the contours are the ones that matter.

The measures, all reduced to "which way does the arrow point" in degrees:

  axis          the PCA line, unwrapped to [0,360) by an arbitrary sign - the baseline
  width         the wider end half is the front (what red_arrow does today)
  area          the half with more pixels is the front
  far           the farthest contour point from the centroid marks the front
  offset        the centroid's displacement along the axis - a solid arrowhead pulls it forward
  defect        the largest convexity defect - the tail notch is a concavity

For a shape that does not rotate, the useful signal is agreement: a measure that flips is
wrong even when it is prettier, and a measure that jitters by 20 degrees cannot drive a route.

Usage:
    python pca_heading.py --frames 200 --out out/heading
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPIKE = os.path.join(os.path.dirname(HERE), "hsv_spike")
sys.path.insert(0, SPIKE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "..", "vision", "src"))

from hsv_candidates import Stream, candidates, load_config      # noqa: E402
from shape_match import ARROW_X, ARROW_Y, contour_of, score_shape  # noqa: E402


def pca_axis(contour: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Unit vector along the contour's main axis, plus centroid and eigenvalues."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    cov = np.cov(centred.T)
    values, vectors = np.linalg.eigh(cov)
    axis = vectors[:, int(np.argmax(values))]
    if axis[0] < 0 or (abs(axis[0]) < 1e-9 and axis[1] < 0):
        axis = -axis                      # a consistent sign, so angles are comparable
    return math.degrees(math.atan2(-axis[1], axis[0])) % 360.0, centroid, axis


def _project(pts: np.ndarray, centroid: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return (pts - centroid) @ axis


def front_by_width(contour: np.ndarray, centroid: np.ndarray, axis: np.ndarray) -> float:
    """The wider end half is the front. This is production's current method."""
    t = _project(contour.reshape(-1, 2).astype(np.float64), centroid, axis)
    normal = np.array([-axis[1], axis[0]])
    # For each end half, the spread across the axis is its width.
    widths = []
    for side in (+1, -1):
        sel = t * side > 0
        if sel.sum() < 3:
            widths.append(0.0)
            continue
        perp = (contour.reshape(-1, 2)[sel] - centroid) @ normal
        widths.append(float(perp.max() - perp.min()))
    base = math.degrees(math.atan2(-axis[1], axis[0])) % 360.0
    return base if widths[0] >= widths[1] else (base + 180.0) % 360.0


def front_by_area(contour: np.ndarray, centroid: np.ndarray, axis: np.ndarray) -> float:
    """Whichever half of the ribbon has more pixels is the front."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    t = _project(pts, centroid, axis)
    base = math.degrees(math.atan2(-axis[1], axis[0])) % 360.0
    return base if (t > 0).sum() >= (t < 0).sum() else (base + 180.0) % 360.0


def front_by_far(contour: np.ndarray, centroid: np.ndarray, axis: np.ndarray) -> float:
    """The farthest contour point from the centroid marks the front."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    d = np.linalg.norm(pts - centroid, axis=1)
    far = pts[int(np.argmax(d))]
    v = far - centroid
    return math.degrees(math.atan2(-v[1], v[0])) % 360.0


def front_by_offset(contour: np.ndarray, centroid: np.ndarray, axis: np.ndarray) -> float:
    """Where the shape's own centre of mass sits relative to the centroid of its outline.

    For a filled arrowhead the pixel mass sits toward the head, so the vector from the
    outline's centroid to the filled centroid points at the front.
    """
    m = cv2.moments(contour)
    if m["m00"] <= 0:
        return math.degrees(math.atan2(-axis[1], axis[0])) % 360.0
    filled = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
    v = filled - centroid
    if np.linalg.norm(v) < 0.3:
        return math.degrees(math.atan2(-axis[1], axis[0])) % 360.0
    return math.degrees(math.atan2(-v[1], v[0])) % 360.0


def front_by_defect(contour: np.ndarray, centroid: np.ndarray, axis: np.ndarray) -> float:
    """The tail notch of an arrow is a concavity: point away from the largest defect."""
    hull_idx = cv2.convexHull(contour, returnPoints=False)
    base = math.degrees(math.atan2(-axis[1], axis[0])) % 360.0
    if hull_idx is None or len(hull_idx) < 4:
        return base
    try:
        defects = cv2.convexityDefects(contour, np.sort(hull_idx, axis=0))
    except cv2.error:
        return base
    if defects is None or len(defects) == 0:
        return base
    deepest = max(defects[:, 0], key=lambda d: d[3])
    far = contour[int(deepest[2])][0].astype(np.float64)
    v = centroid - far                       # away from the notch, toward the front
    if np.linalg.norm(v) < 0.3:
        return base
    return math.degrees(math.atan2(-v[1], v[0])) % 360.0


METHODS = {
    "axis": None,
    "width": front_by_width,
    "area": front_by_area,
    "far": front_by_far,
    "offset": front_by_offset,
    "defect": front_by_defect,
}


def angle_gap(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def circular_median(values: list[float]) -> float:
    if not values:
        return float("nan")
    # The arrow does not spin, so a plain median of unwrapped angles is meaningful; unwrap
    # against the first sample to avoid the 359/1 seam.
    ref = values[0]
    unwrapped = [ref + ((v - ref + 180.0) % 360.0) - 180.0 for v in values]
    return statistics.median(unwrapped) % 360.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--url", default="http://127.0.0.1:8021/stream")
    ap.add_argument("--config", default=os.path.join(SPIKE, "config.orangered.json"))
    ap.add_argument("--reference", default=os.path.join(SPIKE, "out", "phase2",
                                                        "reference_arrow.json"))
    ap.add_argument("--shape-gate", type=float, default=0.300)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "heading"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)
    with open(args.reference, encoding="utf-8") as f:
        reference = np.array(json.load(f)["contour"], np.int32).reshape(-1, 1, 2)

    rows = []
    stream = Stream(args.url)
    frame_no = 0
    try:
        while frame_no < args.frames:
            frame = stream.next()
            if frame is None:
                break
            mask, found = candidates(frame, cfg)
            picked = None
            for c in found:
                if not (abs(c["center_x"] - ARROW_X) < 60 and abs(c["center_y"] - ARROW_Y) < 60):
                    continue
                contour = contour_of(mask, c)
                if contour is None or len(contour) < 5:
                    continue
                if score_shape(contour, reference) > args.shape_gate:
                    continue
                picked = (c, contour)
                break
            if picked is not None:
                c, contour = picked
                axis_angle, centroid, axis = pca_axis(contour)
                row = {"frame": frame_no, "axis": round(axis_angle, 2),
                       "area": c["area"], "cx": c["center_x"], "cy": c["center_y"]}
                for name, fn in METHODS.items():
                    row[name] = round(axis_angle, 2) if fn is None else round(fn(contour, centroid, axis), 2)
                rows.append(row)
            frame_no += 1
    finally:
        stream.close()

    path = os.path.join(args.out, "heading.csv")
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    print("frames %d, arrow found in %d" % (frame_no, len(rows)))
    if not rows:
        print("nothing to compare - is the arrow at (%.0f, %.0f)?" % (ARROW_X, ARROW_Y))
        return 0

    print()
    print("  %-8s %8s %8s %8s %8s %8s" % ("method", "median", "spread", "vs axis", "flips", "agree"))
    axis_median = circular_median([r["axis"] for r in rows])
    for name in METHODS:
        vals = [r[name] for r in rows]
        med = circular_median(vals)
        # Spread measured away from the median so the 359/1 seam does not inflate it.
        dev = [abs(((v - med + 180.0) % 360.0) - 180.0) for v in vals]
        flips = sum(1 for d in dev if d > 90.0)
        agree = 100.0 * (1.0 - flips / len(vals))
        print("  %-8s %8.1f %8.1f %8.1f %8d %7.1f%%"
              % (name, med, max(dev), angle_gap(med, axis_median), flips, agree))

    print()
    print("  spread = worst deviation from that method's own median (0 = never moved)")
    print("  flips  = frames more than 90 deg from the median, i.e. the 180 ambiguity firing")
    print("  vs axis= how far the method's median sits from the bare PCA line")
    print("  A method is usable if spread is small AND flips is 0.")
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
