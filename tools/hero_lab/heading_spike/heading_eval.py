#!/usr/bin/env python3
"""Seven heading methods on one batch of frames, with the temporal check.

Same frames for every method, cached to disk first, so a difference in the numbers cannot be a
difference in what each one looked at. Angles are all in production's convention:
atan2(y, x) in image coordinates, clockwise-positive, as red_arrow._heading returns them.

Methods
  prod      red_arrow.detect(frame)                 - what production runs today
  axis      PCA on filled mask pixels, locked sign  - the axis, no 180 resolution
  width     outer 25% bands, mean |perp|            - the fix: filled pixels, mean not max
  area      outer 25% bands, pixel count            - the fix: filled pixels
  combined  width + area as one signed score        - the task's head_tail_score
  offset    outline centroid -> filled centroid     - the spike's only zero-flip method
  hog       hero_hog.detect(frame)                  - the fallback, for reference

The temporal check reports consecutive-frame jumps over 90 deg. A single-frame 180 error and a
real spin look the same at one frame, so this only *marks* them; it cannot hide a bad decision.

Usage:
    python heading_eval.py --frames 220 --cache out/frames --out out/heading_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # heading_spike -> hero_lab -> tools -> project
SPIKE = os.path.join(os.path.dirname(HERE), "hsv_spike")
sys.path.insert(0, SPIKE)
sys.path.insert(0, os.path.join(ROOT, "vision", "src"))

from hsv_candidates import Stream                                    # noqa: E402
import red_arrow                                                     # noqa: E402
import hero_hog                                                      # noqa: E402

BAND = 0.25          # outer fraction of the projected extent used at each end
ARROW = (240.0, 485.0)   # where hsv_marker puts the arrow; used as prefer_near so every
                         # method looks at the same blob that production looks at
AREA_WEIGHT = 0.02   # lambda in the combined score; area differences are hundreds of pixels


# ------------------------------------------------------------------ the axis, done properly


def axis_and_pixels(blob: np.ndarray):
    """PCA on the filled mask, in red_arrow's convention, plus the projected coordinates."""
    ys, xs = np.where(blob > 0)
    if len(xs) < 10:
        return None
    mx, my = xs.mean(), ys.mean()
    pts = np.stack([xs - mx, ys - my], 1).astype(np.float64)
    cov = np.cov(pts.T)
    values, vectors = np.linalg.eigh(cov)
    u = vectors[:, int(np.argmax(values))]
    # Sign does not matter for an axis, but it must be *consistent* so the halves are
    # comparable. red_arrow leaves it to LAPACK and then resolves the 180 by width; here the
    # sign is fixed by the projection instead, which is equivalent and reproducible.
    if u[0] < 0:
        u = -u
    proj = pts @ u
    perp = pts @ np.array([-u[1], u[0]])
    return u, proj, np.abs(perp), (mx, my)


def _ends(proj, perp, area_all):
    lo, hi = float(proj.min()), float(proj.max())
    span = max(hi - lo, 1e-6)
    w = max(span * BAND, 1.0)
    near_lo = proj <= lo + w
    near_hi = proj >= hi - w
    if near_lo.sum() < 5 or near_hi.sum() < 5:
        return None
    width_lo = float(perp[near_lo].mean())
    width_hi = float(perp[near_hi].mean())
    area_lo = int(near_lo.sum())
    area_hi = int(near_hi.sum())
    return width_lo, width_hi, area_lo, area_hi


def heading_width(blob):
    r = axis_and_pixels(blob)
    if r is None:
        return None, None
    u, proj, perp, _ = r
    ends = _ends(proj, perp, len(proj))
    if ends is None:
        return None, None
    width_lo, width_hi, _alo, _ahi = ends
    axis = math.degrees(math.atan2(u[1], u[0])) % 360.0
    return (axis if width_hi > width_lo else (axis + 180.0) % 360.0), width_hi - width_lo


def heading_area(blob):
    r = axis_and_pixels(blob)
    if r is None:
        return None, None
    u, proj, perp, _ = r
    ends = _ends(proj, perp, len(proj))
    if ends is None:
        return None, None
    _wlo, _whi, area_lo, area_hi = ends
    axis = math.degrees(math.atan2(u[1], u[0])) % 360.0
    return (axis if area_hi > area_lo else (axis + 180.0) % 360.0), area_hi - area_lo


def heading_combined(blob):
    r = axis_and_pixels(blob)
    if r is None:
        return None, None
    u, proj, perp, _ = r
    ends = _ends(proj, perp, len(proj))
    if ends is None:
        return None, None
    width_lo, width_hi, area_lo, area_hi = ends
    axis = math.degrees(math.atan2(u[1], u[0])) % 360.0
    score = (width_hi - width_lo) + AREA_WEIGHT * (area_hi - area_lo)
    return (axis if score > 0 else (axis + 180.0) % 360.0), score


def heading_width85(blob):
    """red_arrow's own metric, on the filled mask: 85th percentile of |perp| per end band."""
    r = axis_and_pixels(blob)
    if r is None:
        return None, None
    u, proj, perp, _ = r
    lo, hi = float(proj.min()), float(proj.max())
    w = max((hi - lo) * 0.30, 1.0)
    n_lo = perp[proj < lo + w]
    n_hi = perp[proj > hi - w]
    if len(n_lo) < 5 or len(n_hi) < 5:
        return None, None
    narrow = float(np.percentile(n_lo, 85))
    wide = float(np.percentile(n_hi, 85))
    axis = math.degrees(math.atan2(u[1], u[0])) % 360.0
    return (axis if wide > narrow else (axis + 180.0) % 360.0), wide - narrow


def heading_axis(blob):
    r = axis_and_pixels(blob)
    if r is None:
        return None, None
    u = r[0]
    return (math.degrees(math.atan2(u[1], u[0])) % 360.0), None


def heading_offset(blob):
    r = axis_and_pixels(blob)
    if r is None:
        return None, None
    u, proj, perp, (mx, my) = r
    # outline centroid vs filled centroid
    contours, _h = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None
    contour = max(contours, key=cv2.contourArea)
    m = cv2.moments(contour)
    if m["m00"] <= 0:
        return None, None
    fx, fy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    v = np.array([fx - mx, fy - my])
    if np.linalg.norm(v) < 0.3:
        return (math.degrees(math.atan2(u[1], u[0])) % 360.0), None
    return (math.degrees(math.atan2(v[1], v[0])) % 360.0), float(np.linalg.norm(v))


def blob_of(frame, prefer_near=None):
    """The same blob red_arrow would pick, so the methods see identical pixels."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.morphologyEx(red_arrow._mask(hsv), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    _n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, len(stats)):
        x, y, w, h, area = stats[i]
        if area < red_arrow.MIN_AREA or area > red_arrow.MAX_AREA:
            continue
        aspect = max(w, h) / max(min(w, h), 1.0)
        if aspect < red_arrow.MIN_ASPECT or aspect > red_arrow.MAX_ASPECT:
            continue
        if prefer_near is not None:
            d = math.hypot(centroids[i][0] - prefer_near[0], centroids[i][1] - prefer_near[1])
            key = d
        else:
            key = -area
        if best is None or key < best[0]:
            best = (key, i)
    if best is None:
        return None
    return (labels == best[1]).astype(np.uint8)


# ------------------------------------------------------------------ statistics


def circ_median(vals):
    if not vals:
        return float("nan")
    ref = vals[0]
    unwrapped = [ref + ((v - ref + 180.0) % 360.0) - 180.0 for v in vals]
    return statistics.median(unwrapped) % 360.0


def circ_dev(vals, med):
    return [abs(((v - med + 180.0) % 360.0) - 180.0) for v in vals]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=220)
    ap.add_argument("--url", default="http://127.0.0.1:8021/stream")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "heading_eval"))
    ap.add_argument("--cache", default=os.path.join(HERE, "out", "frames"),
                    help="frames are cached here so every method sees the same images")
    ap.add_argument("--capture", action="store_true", help="(re)capture the cache")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)

    # ---- frames: capture once, reuse forever
    existing = sorted(f for f in os.listdir(args.cache) if f.endswith(".jpg"))
    if args.capture or len(existing) < args.frames:
        for f in existing:
            os.unlink(os.path.join(args.cache, f))
        stream = Stream(args.url)
        got = 0
        try:
            while got < args.frames:
                frame = stream.next()
                if frame is None:
                    break
                cv2.imwrite(os.path.join(args.cache, "%04d.jpg" % got), frame)
                got += 1
        finally:
            stream.close()
        print("captured %d frames into %s" % (got, args.cache))
        files = sorted(f for f in os.listdir(args.cache) if f.endswith(".jpg"))
    else:
        files = existing[:args.frames]
        print("using %d cached frames from %s" % (len(files), args.cache))

    frames = [cv2.imread(os.path.join(args.cache, f)) for f in files]

    rows = []
    timings = {k: 0.0 for k in ("prod", "hog", "axis", "width", "area", "combined",
                                "width85", "offset")}
    for idx, frame in enumerate(frames):
        row = {"frame": idx}

        t = time.perf_counter()
        pose = red_arrow.detect(frame, prefer_near=ARROW)
        timings["prod"] += time.perf_counter() - t
        row["prod"] = round(pose.heading, 2) if pose else ""

        t = time.perf_counter()
        hog = hero_hog.detect(frame)
        timings["hog"] += time.perf_counter() - t
        row["hog"] = round(float(hog.heading) % 360.0, 2) if (hog and hog.heading is not None) else ""

        blob = blob_of(frame, prefer_near=ARROW)
        if blob is None:
            rows.append(row)
            continue
        for name, fn in (("axis", heading_axis), ("width", heading_width),
                         ("area", heading_area), ("combined", heading_combined),
                         ("width85", heading_width85),
                         ("offset", heading_offset)):
            t = time.perf_counter()
            angle, score = fn(blob)
            timings[name] += time.perf_counter() - t
            row[name] = "" if angle is None else round(angle, 2)
            row[name + "_score"] = "" if score is None else round(float(score), 3)
        rows.append(row)

    path = os.path.join(args.out, "heading_eval.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)

    methods = ["prod", "axis", "width", "area", "combined", "width85", "offset", "hog"]
    print()
    print("  %-9s %5s %8s %8s %8s %6s %8s %9s" % (
        "method", "valid", "median", "jitter", "maxdev", "flips", "temporal", "ms/frame"))
    for m in methods:
        vals = [r[m] for r in rows if r.get(m, "") != ""]
        if not vals:
            print("  %-9s %5d  (never produced a heading)" % (m, 0))
            continue
        med = circ_median(vals)
        dev = circ_dev(vals, med)
        flips = sum(1 for d in dev if d > 90.0)
        # temporal: consecutive valid frames jumping more than 90 deg
        seq = [r[m] for r in rows if r.get(m, "") != ""]
        temporal = 0
        for a, b in zip(seq, seq[1:]):
            if abs(((b - a + 180.0) % 360.0) - 180.0) > 90.0:
                temporal += 1
        # jitter: median absolute deviation, robust to the odd real jump
        mad = statistics.median(dev)
        print("  %-9s %5d %8.1f %8.1f %8.1f %6d %8d %9.2f" % (
            m, len(vals), med, mad, max(dev), flips, temporal,
            1000.0 * timings[m] / max(1, len(frames))))

    print()
    print("  jitter  = median absolute deviation from that method's own median (deg)")
    print("  maxdev  = worst deviation (deg); a 180 error shows up here as ~180")
    print("  flips   = frames more than 90 deg from the median")
    print("  temporal= consecutive-frame jumps over 90 deg (marks, does not excuse, 180 errors)")
    print("  ms/frame is the method's own cost, excluding JPEG decode")
    print("wrote %s" % path)

    # Head/tail score spread tells us how decisive the 180 decision was.
    for m in ("width", "area", "combined", "width85"):
        s = [r[m + "_score"] for r in rows if r.get(m + "_score", "") != ""]
        if s:
            print("  %-9s head_tail_score: min %.2f  median %.2f  max %.2f"
                  % (m, min(s), statistics.median(s), max(s)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
