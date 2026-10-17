"""Grab a candidate reference frame, without touching the production one.

The production reference is vision/data/reference/reference.jpg and it is written by the `r` key in
main.py, atomically. If that window is not open, this does the same job into
vision/data/reference/reference_new.jpg and refuses when the frame is not empty.

Refusing matters more than capturing. main.py's own notes record the evening lost to a reference
captured with something standing in the scene: whatever is in the picture becomes part of the
reference and every difference test afterwards is wrong in a way that looks like a detection bug.
So the two checks here are the ones that would have caught it:

  * no car-sized blob of paint in the frame   - the paint is 400-1000 px on this camera, and a
    handful of bay-light slivers is not the same thing;
  * the frame differs from the old reference by a plausible amount, reported either way.

    python tools/hero_lab/capture_reference.py              # save if the frame looks empty
    python tools/hero_lab/capture_reference.py --force      # save anyway (it will be rejected later)

Then compare with the 139 labelled frames before promoting it:

    python tools/hero_lab/diff_reference.py --reference vision/data/reference/reference_new.jpg
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "vision", "src"))

import vehicle                                                        # noqa: E402

REF_DIR = os.path.join(ROOT, "vision", "data", "reference")
OLD = os.path.join(REF_DIR, "reference.jpg")
NEW = os.path.join(REF_DIR, "reference_new.jpg")

CAR_PAINT_MIN = 350      # px: the paint reads 421-933 live; a light sliver is 17-285


def paint_area(frame: np.ndarray) -> int:
    """The largest blob of car paint in the frame, in px."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(frame.shape[:2], np.uint8)
    for lo, hi in vehicle.vehicle_config()["hue_ranges"]:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    count, _l, stats, _c = cv2.connectedComponentsWithStats(mask, 8)
    return 0 if count <= 1 else int(np.max(stats[1:, 4]))


def sharpness(frame: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="http://192.168.3.118/stream")
    ap.add_argument("--frames", type=int, default=30, help="frames to look at before choosing")
    ap.add_argument("--out", default=NEW)
    ap.add_argument("--force", action="store_true", help="save even if the frame is not empty")
    args = ap.parse_args()

    from camera import open_camera
    cam = open_camera(args.source)
    best, best_sharp, seen = None, -1.0, 0
    worst_paint = 0
    started = time.time()
    while seen < args.frames and time.time() - started < 20:
        frame = cam.read()
        if frame is None:
            continue
        seen += 1
        worst_paint = max(worst_paint, paint_area(frame))
        sharp = sharpness(frame)
        if sharp > best_sharp:                     # the sharpest of the batch: less motion blur
            best, best_sharp = frame, sharp
    cam.close()
    if best is None:
        print("  no frame from the camera")
        return 1

    paint = paint_area(best)
    print(f"  frames {seen}   sharpest {best_sharp:.0f}   largest paint blob {paint} px "
          f"(worst over the batch {worst_paint})")
    old = cv2.imread(OLD)
    if old is not None and old.shape == best.shape:
        diff = cv2.absdiff(cv2.cvtColor(best, cv2.COLOR_BGR2GRAY),
                           cv2.cvtColor(old, cv2.COLOR_BGR2GRAY))
        changed = float(np.count_nonzero(diff > 18)) / diff.size * 100.0
        print(f"  differs from the old reference in {changed:.1f}% of pixels "
              f"(a whole-scene change looks like 15%+, the car alone is under 1%)")

    if paint >= CAR_PAINT_MIN and not args.force:
        print(f"  REFUSED: a blob of {paint} px of car paint is in the frame. The car reads "
              f"{CAR_PAINT_MIN}+ px, so something yellow and car-sized is standing in the scene -")
        print("           and whatever is in the reference becomes part of every difference test.")
        print("           Move the car out of the picture and run this again, or pass --force.")
        return 2

    os.makedirs(REF_DIR, exist_ok=True)
    tmp = args.out + ".tmp.jpg"
    if not cv2.imwrite(tmp, best):
        print("  could not write")
        return 1
    os.replace(tmp, args.out)                      # atomic, like main.py's own capture
    print(f"  saved {args.out}")
    print("  the production reference is untouched; promote it only after diff_reference.py agrees")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
