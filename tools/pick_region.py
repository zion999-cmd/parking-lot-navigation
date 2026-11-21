#!/usr/bin/env python3
"""Draw the parking-lot outline: click four corners on the reference frame.

The hero detector searches the whole frame for red blobs, and the sandbox has plenty of red that is
not a car - measured 13 blobs passing the size and aspect filters in a single frame, which is why the
tracked car hopped between them. This tool writes one extra field into the calibration file:

    "lot_region": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

and from then on hero detection only considers candidates whose centre is inside it. It does not touch
the bays, the roads or anything else in that file.

    tools/pick_region.py                        # reference frame, click 4 corners
    tools/pick_region.py --frame some.jpg       # a different still
    tools/pick_region.py --show                 # just draw the current outline and exit

In the window: click the four corners in order (any order, as long as the quad is not self-crossing),
then Enter or s to save, r to start over, q or Esc to leave without writing. Keys work the same way as
the bay editor; the points snap to whole pixels.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "vision", "src"))

DEFAULT_FRAME = os.path.join(ROOT, "vision", "data", "reference", "reference.jpg")
DEFAULT_CONFIG = os.path.join(ROOT, "vision", "config", "parking_spots.json")
WINDOW = "parking lot outline - click 4 corners, Enter saves, r resets, q quits"
MAX_W = 1100


def current_region(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle).get("lot_region")
    except (OSError, ValueError):
        return None


def write_region(path, points):
    """Add or replace lot_region, leaving every other key exactly as it was."""
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    if points is None:
        document.pop("lot_region", None)
    else:
        document["lot_region"] = [[int(round(x)), int(round(y))] for x, y in points]
    tmp = path + ".writing"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=1)
        handle.write("\n")
    os.replace(tmp, path)
    return document.get("lot_region")


def draw(base, points, region):
    canvas = base.copy()
    if region:
        pts = np.array(region, np.int32).reshape(-1, 1, 2)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], (60, 60, 60))
        canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
        cv2.polylines(canvas, [pts], True, (0, 220, 255), 2)
    for index, (x, y) in enumerate(points):
        cv2.circle(canvas, (int(x), int(y)), 6, (0, 255, 0), -1)
        cv2.putText(canvas, str(index + 1), (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if len(points) > 1:
        cv2.polylines(canvas, [np.array(points, np.int32).reshape(-1, 1, 2)], len(points) == 4,
                      (0, 255, 0), 1)
    hint = f"{len(points)}/4 points"
    cv2.putText(canvas, hint, (14, canvas.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
    return canvas


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", default=DEFAULT_FRAME)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--show", action="store_true", help="draw the current outline and exit")
    args = ap.parse_args(argv)

    frame = cv2.imread(args.frame, cv2.IMREAD_COLOR)
    if frame is None:
        print(f"error: could not read {args.frame}", file=sys.stderr)
        return 2

    region = current_region(args.config)
    if args.show:
        if not region:
            print(f"  no lot_region in {args.config}")
            return 1
        print("  lot_region:", region)
        cv2.imshow(WINDOW, draw(frame, [], region))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 0

    scale = min(1.0, MAX_W / frame.shape[1])
    view = frame if scale >= 1.0 else cv2.resize(frame, None, fx=scale, fy=scale,
                                                 interpolation=cv2.INTER_AREA)
    points = []                       # in frame coordinates, not view coordinates

    def on_mouse(event, x, y, flags, _user):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x / scale, y / scale))

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print("  click the four corners of the parking lot, then Enter to save (r resets, q quits)")
    if region:
        print(f"  existing outline: {region}")

    while True:
        shown = draw(view, [(x * scale, y * scale) for x, y in points], 
                     [(x * scale, y * scale) for x, y in region] if region and not points else None)
        cv2.imshow(WINDOW, shown)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10, ord("s")) and len(points) == 4:
            saved = write_region(args.config, points)
            print(f"  wrote lot_region {saved} -> {args.config}")
            print("  hero detection will only consider candidates inside this outline from the next")
            print("  main.py start (restart it: ./start.sh stop --all; ./start.sh)")
            break
        if key == ord("r"):
            points = []
            region = None
        if key in (27, ord("q")):
            print("  left without writing")
            break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
