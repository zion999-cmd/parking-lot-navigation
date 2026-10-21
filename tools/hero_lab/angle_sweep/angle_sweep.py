#!/usr/bin/env python3
"""Angle sweep: is the head/tail decision anisotropic, and did the old path do better?

The question comes from an observation that the direction is rock solid up/down/left/right and
wobbles at the diagonals. Two candidate explanations, and this tool separates them:

  1. geometry - a 32x19 px arrow rasterised at 45 degrees loses the 1-3 px head/tail margin
  2. history  - the old path had a step that is now gone

For (2) the git record is already conclusive and needs no frames: `_heading` is byte-identical
to its first commit, and no version of red_arrow.py has ever upscaled, rotated or blurred a
ROI. What did disappear is a 30 deg gate with 12-frame persistence, added in 585a327 inside
*hero_hog* - so while HOG was the heading provider every reading was pinned, and now that it is
not, a jittery-but-successful frame is accepted as-is.

For (1) this sweep. Real arrows from real frames, rotated in place by rotating a patch around
the arrow and compositing it back through a circular alpha so there is no seam. Every arm is
scored against its own reading at 0 degrees, which cancels the per-frame base angle and leaves
only the rotation effect.

Arms
  detect    red_arrow.detect            - what production runs
  in_box    red_arrow.detect_in_box     - the historical path, box around the arrow
  up4x      same mask, 4x cubic upsample - the proposed fix, measured rather than assumed

Usage:
    python angle_sweep.py --frames 60 --step 15 --out out/angle_sweep
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # angle_sweep -> hero_lab -> tools -> project
SRC = os.path.join(ROOT, "vision", "src")
FRAMES = os.path.join(ROOT, "tools", "hero_lab", "heading_spike", "out", "frames")
sys.path.insert(0, SRC)

import red_arrow      # noqa: E402
import hsv_marker     # noqa: E402

ARROW_X, ARROW_Y = 240.0, 485.0
PATCH = 64            # square patch rotated around the arrow
ALPHA_R = 28          # circular composite radius, so the paste has no square seam
UPSAMPLE = 4


def circ(a: float) -> float:
    return a % 360.0


def circ_diff(a: float, b: float) -> float:
    """Signed difference a-b in (-180, 180]."""
    return ((a - b + 180.0) % 360.0) - 180.0


def rotate_patch(frame: np.ndarray, cx: float, cy: float, angle: float) -> np.ndarray | None:
    """Rotate a patch about the arrow and paste it back through a circular alpha."""
    half = PATCH // 2
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    if x0 < 0 or y0 < 0 or x0 + PATCH > frame.shape[1] or y0 + PATCH > frame.shape[0]:
        return None
    patch = frame[y0:y0 + PATCH, x0:x0 + PATCH].copy()
    m = cv2.getRotationMatrix2D((half, half), angle, 1.0)
    rot = cv2.warpAffine(patch, m, (PATCH, PATCH), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    yy, xx = np.mgrid[0:PATCH, 0:PATCH]
    circle = (((xx - half) ** 2 + (yy - half) ** 2) <= ALPHA_R ** 2).astype(np.uint8) * 255
    circle = cv2.GaussianBlur(circle, (5, 5), 0)
    a = (circle.astype(np.float32) / 255.0)[..., None]
    out = frame.copy()
    out[y0:y0 + PATCH, x0:x0 + PATCH] = (rot * a + patch * (1 - a)).astype(np.uint8)
    return out


def arm_detect(frame: np.ndarray, cx: float, cy: float):
    pose = red_arrow.detect(frame, prefer_near=(cx, cy))
    if pose is None:
        return None, None
    return circ(pose.heading), _head_tail(frame, pose)


def _head_tail(frame: np.ndarray, pose) -> float:
    """The width margin the decision actually had, recomputed for the report."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.morphologyEx(red_arrow._mask(hsv), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    x, y, w, h = pose.bbox
    sub = mask[y:y + h, x:x + w]
    blob = (sub > 0).astype(np.uint8)
    if blob.sum() < 10:
        return None
    r = _axis(blob)
    if r is None:
        return None
    (_u, proj, perp) = r
    lo, hi = float(proj.min()), float(proj.max())
    band = max((hi - lo) * 0.30, 1.0)
    nl, nh = perp[proj < lo + band], perp[proj > hi - band]
    if len(nl) < 5 or len(nh) < 5:
        return None
    return float(np.percentile(nh, 85) - np.percentile(nl, 85))


def _axis(blob: np.ndarray):
    ys, xs = np.where(blob > 0)
    if len(xs) < 10:
        return None
    pts = np.stack([xs - xs.mean(), ys - ys.mean()], 1).astype(np.float32)
    _m, evec = cv2.PCACompute(pts, mean=None, maxComponents=2)
    u = np.array([evec[0][0], evec[0][1]])
    proj = pts @ u
    perp = np.abs(pts @ np.array([-u[1], u[0]]))
    return u, proj, perp


def arm_in_box(frame: np.ndarray, cx: float, cy: float):
    box = (int(cx) - 32, int(cy) - 16, 64, 32)
    pose = red_arrow.detect_in_box(frame, box)
    if pose is None:
        return None, None
    return circ(pose.heading), None


def arm_up4x(frame: np.ndarray, cx: float, cy: float):
    half = 24
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    if x0 < 0 or y0 < 0 or x0 + 2 * half > frame.shape[1] or y0 + 2 * half > frame.shape[0]:
        return None, None
    crop = frame[y0:y0 + 2 * half, x0:x0 + 2 * half]
    up = cv2.resize(crop, None, fx=UPSAMPLE, fy=UPSAMPLE, interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(up, cv2.COLOR_BGR2HSV)
    mask = red_arrow._mask(hsv)
    k = np.ones((2 * UPSAMPLE, 2 * UPSAMPLE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    n, labels, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < red_arrow.MIN_AREA * UPSAMPLE * UPSAMPLE:
            continue
        if best is None or area > best[1]:
            best = (i, area)
    if best is None:
        return None, None
    blob = (labels == best[0]).astype(np.uint8)
    return circ(red_arrow._heading(blob)), _head_tail_of_blob(blob)


def _head_tail_of_blob(blob: np.ndarray) -> float:
    r = _axis(blob)
    if r is None:
        return None
    (_u, proj, perp) = r
    lo, hi = float(proj.min()), float(proj.max())
    band = max((hi - lo) * 0.30, 1.0)
    nl, nh = perp[proj < lo + band], perp[proj > hi - band]
    if len(nl) < 5 or len(nh) < 5:
        return None
    return float(np.percentile(nh, 85) - np.percentile(nl, 85))


ARMS = {"detect": arm_detect, "in_box": arm_in_box, "up4x": arm_up4x}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--step", type=int, default=15)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "angle_sweep"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    names = sorted(n for n in os.listdir(FRAMES) if n.endswith(".jpg"))[:args.frames]
    frames = [cv2.imread(os.path.join(FRAMES, n)) for n in names]
    frames = [f for f in frames if f is not None]
    print("frames %d, patch %d px, upsample %dx" % (len(frames), PATCH, UPSAMPLE))

    angles = list(range(0, 360, args.step))
    rows = []

    for angle in angles:
        for arm_name, fn in ARMS.items():
            errors = []
            margins = []
            fails = 0
            for frame in frames:
                h = hsv_marker.detect(frame, prefer_near=(ARROW_X, ARROW_Y))
                if h is None:
                    continue
                cx, cy = h["centre"]
                ref, _ = fn(frame, cx, cy)          # this frame's own 0-degree reading
                if ref is None:
                    continue
                moved = rotate_patch(frame, cx, cy, angle)
                if moved is None:
                    continue
                got, margin = fn(moved, cx, cy)
                if got is None:
                    fails += 1
                    continue
                # cv2's positive angle turns the image counter-clockwise on screen, and
                # red_arrow's heading grows clockwise, so the expected reading drops by angle.
                expected = circ(ref - angle)
                errors.append(circ_diff(got, expected))
                if margin is not None:
                    margins.append(abs(margin))
            if not errors:
                rows.append({"angle": angle, "arm": arm_name, "n": 0, "fails": fails})
                continue
            flips = sum(1 for e in errors if abs(e) > 90.0)
            rows.append({
                "angle": angle, "arm": arm_name, "n": len(errors), "fails": fails,
                "success_pct": round(100.0 * len(errors) / max(1, len(errors) + fails), 1),
                "mean_err": round(statistics.fmean(errors), 1),
                "std": round(statistics.pstdev(errors), 1) if len(errors) > 1 else 0.0,
                "abs_mean": round(statistics.fmean(abs(e) for e in errors), 1),
                "flips": flips,
                "flip_pct": round(100.0 * flips / len(errors), 1),
                "margin": round(statistics.median(margins), 2) if margins else "",
            })

    path = os.path.join(args.out, "angle_sweep.csv")
    keys = ["angle", "arm", "n", "fails", "success_pct", "mean_err", "std", "abs_mean",
            "flips", "flip_pct", "margin"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

    for arm_name in ARMS:
        print()
        print("  %s" % arm_name)
        print("  %6s %7s %7s %8s %7s %7s" % ("angle", "succ%", "err std", "|err|", "flips", "margin"))
        for r in rows:
            if r["arm"] != arm_name or not r.get("n"):
                continue
            print("  %6d %7.1f %7.1f %8.1f %7d %7s"
                  % (r["angle"], r["success_pct"], r["std"], r["abs_mean"],
                     r["flips"], r.get("margin", "")))

    # Cardinal vs diagonal, which is the actual question.
    print()
    print("  cardinal (0/90/180/270) vs diagonal (45/135/225/315)")
    for arm_name in ARMS:
        card = [r for r in rows if r["arm"] == arm_name and r["angle"] % 90 == 0 and r.get("n")]
        diag = [r for r in rows if r["arm"] == arm_name and r["angle"] % 90 == 45 and r.get("n")]
        if not card or not diag:
            continue
        c_std = statistics.fmean(r["std"] for r in card)
        d_std = statistics.fmean(r["std"] for r in diag)
        c_m = [r["margin"] for r in card if r["margin"] != ""]
        d_m = [r["margin"] for r in diag if r["margin"] != ""]
        c_mi = statistics.fmean(c_m) if c_m else float("nan")
        d_mi = statistics.fmean(d_m) if d_m else float("nan")
        print("  %-8s std  cardinal %4.1f  diagonal %4.1f  (x%.2f)   "
              "margin  cardinal %4.2f  diagonal %4.2f  (x%.2f)"
              % (arm_name, c_std, d_std, d_std / c_std if c_std else 0,
                 c_mi, d_mi, c_mi / d_mi if d_mi else 0))
    print()
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
