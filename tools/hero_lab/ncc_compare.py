"""Which normalisation actually recovers the arrow inside a crop?

cv2.matchTemplate's TM_CCOEFF_NORMED with a mask is the thing under suspicion: the same
arrow scored 0.919 in a 96 px crop, 0.770 in a 190 px crop and 0.919 over the whole
frame, and TM_CCORR_NORMED read 0.995 at every angle, so it cannot tell direction.

This compares four ways on the same saved frame, at several crop sizes, and reports the
score at the true angle against the best score at any other angle - which is the number
that matters, because a detector needs both a high score and a gap.

    python tools/hero_lab/ncc_compare.py vision/data/debug/lost_case.jpg

Read-only: it loads a frame and the committed template and prints a table.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "vision", "src"))
import arrow          # noqa: E402


def score_mask_ccoeff(crop, t, m):
    return float(cv2.minMaxLoc(cv2.matchTemplate(
        crop, t, cv2.TM_CCOEFF_NORMED, mask=(m > 0).astype(np.uint8) * 255))[1])


def score_mean_removed(crop, t, m):
    """What tmplmatch would do if the normalisation were over the mask only."""
    keep = m > 0
    t2 = t.copy()
    t2[~keep] = 0.0
    t2 -= t2[keep].mean()
    c2 = crop.copy()
    c2 -= c2.mean()
    return float(cv2.minMaxLoc(cv2.matchTemplate(c2, t2, cv2.TM_CCOEFF_NORMED))[1])


def score_sqdiff(crop, t, m):
    return 1.0 - float(cv2.minMaxLoc(cv2.matchTemplate(
        crop, t, cv2.TM_SQDIFF_NORMED, mask=(m > 0).astype(np.uint8) * 255))[0])


METHODS = (("CCOEFF+mask (now)", score_mask_ccoeff),
           ("mean-removed", score_mean_removed),
           ("SQDIFF+mask", score_sqdiff))


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        arrow.ROOT, "vision", "data", "debug", "lost_case.jpg")
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit("cannot read %s" % path)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tpl = arrow.load()

    full = arrow.find_in(grey, tpl)
    cx, cy = full.centre
    print("frame %s   arrow found whole-frame at (%.0f, %.0f) angle %.1f score %.3f"
          % (os.path.basename(path), cx, cy, full.heading, full.score))
    print()
    header = "  crop        " + "".join("%-22s" % name for name, _ in METHODS)
    print(header + "true-angle vs best-other")
    for half in (48, 70, 95, 130, 190):
        x0, y0 = max(0, int(cx) - half), max(0, int(cy) - half)
        crop = grey[y0:y0 + 2 * half, x0:x0 + 2 * half]
        if crop.shape[0] < 90 or crop.shape[1] < 90:
            continue
        row = "  %3d px      " % (2 * half)
        for _name, fn in METHODS:
            at_true, best_other = None, -2.0
            degs = [float(d) for d in np.arange(0, 360, 15)]
            # The true angle is whatever the fine sweep found, which is 5-degree steps off
            # the coarse grid; mark the nearest coarse angle as "true" and the rest as rivals.
            true_deg = min(degs, key=lambda d: abs((full.heading - d + 180.0) % 360.0 - 180.0))
            for deg in degs:
                t, m = tpl.at(deg)
                if t.shape[0] >= crop.shape[0] or t.shape[1] >= crop.shape[1]:
                    continue
                s = fn(crop, t, m)
                if deg == true_deg:
                    at_true = s
                else:
                    best_other = max(best_other, s)
            if at_true is None:
                row += "%-22s" % "n/a"
            else:
                row += "%-22s" % ("%.3f / %.3f" % (at_true, best_other))
        print(row)
    print()
    print("  Each cell: score at the true angle / the best score at any other angle.")
    print("  A usable method keeps the first above ~0.90 and the second well below it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
