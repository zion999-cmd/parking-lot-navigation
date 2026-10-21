#!/usr/bin/env python3
"""Phase 2: do the candidates' shapes separate the arrow from the red clutter?

Phase 1 left 1-6 candidates per frame: the arrow, plus 1-4 blobs at the frame edges where the
bay lights and the LED strip are. Those edge blobs are redder than the arrow, so no threshold
will remove them. The question here is whether shape will.

Method: take one arrow contour as the reference (picked from a frame at the position Phase 1
measured, i.e. the arrow itself - not a training set), then score every candidate against it
with cv2.matchShapes. Its Hu-moment metrics are invariant to position, scale and rotation,
which is what we want for a candidate filter; the cost is that the scores say nothing about
which way the arrow points, so heading still needs the template phase.

Usage:
    python shape_match.py --frames 200 --out out/phase2
    python shape_match.py --frames 200 --reference reference_arrow.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from hsv_candidates import Stream, candidates, load_config, overlay  # noqa: E402

ARROW_X, ARROW_Y = 240.0, 485.0   # measured in Phase 1, the candidate 200/200 frames shared


def contour_of(mask: np.ndarray, c: dict) -> np.ndarray:
    """The outer contour of exactly this candidate, cut out of the mask."""
    x, y, w, h = c["bbox"]
    sub = mask[y:y + h, x:x + w]
    contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 1, 2), np.int32)
    return max(contours, key=cv2.contourArea)


def score_shape(contour: np.ndarray, reference: np.ndarray) -> float:
    """Lower is more similar. 0.0 would be identical."""
    if contour is None or len(contour) < 3 or reference is None or len(reference) < 3:
        return float("nan")
    return float(cv2.matchShapes(contour, reference, cv2.CONTOURS_MATCH_I1, 0.0))


def pick_reference(frames_wanted: int, cfg: dict, url: str) -> tuple[np.ndarray, dict]:
    """Grab frames until one has a clean candidate at the arrow's known position."""
    stream = Stream(url)
    try:
        for _ in range(60):
            frame = stream.next()
            if frame is None:
                break
            mask, found = candidates(frame, cfg)
            near = [c for c in found
                    if abs(c["center_x"] - ARROW_X) < 40 and abs(c["center_y"] - ARROW_Y) < 40]
            if near:
                c = max(near, key=lambda q: q["area"])
                contour = contour_of(mask, c)
                if len(contour) >= 5:
                    return contour, c
    finally:
        stream.close()
    raise SystemExit("could not find a candidate at the arrow position to use as reference")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--url", default="http://127.0.0.1:8021/stream")
    ap.add_argument("--config", default=os.path.join(HERE, "config.orangered.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "out", "phase2"))
    ap.add_argument("--reference", help="reuse a saved reference contour instead of picking one")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)

    if args.reference and os.path.exists(args.reference):
        data = json.load(open(args.reference, encoding="utf-8"))
        reference = np.array(data["contour"], np.int32).reshape(-1, 1, 2)
        print("reference: %s (%d points)" % (args.reference, len(reference)))
    else:
        reference, c = pick_reference(40, cfg, args.url)
        path = os.path.join(args.out, "reference_arrow.json")
        json.dump({"contour": reference.reshape(-1, 2).tolist(),
                   "picked_from": {"center_x": c["center_x"], "center_y": c["center_y"],
                                   "area": c["area"], "bbox": c["bbox"]}},
                  open(path, "w", encoding="utf-8"), indent=1)
        print("reference: picked at (%.1f, %.1f) area %d, %d points -> %s"
              % (c["center_x"], c["center_y"], c["area"], len(reference), path))

    rows = []
    stream = Stream(args.url)
    frame_no = 0
    try:
        while frame_no < args.frames:
            frame = stream.next()
            if frame is None:
                break
            mask, found = candidates(frame, cfg)
            for i, c in enumerate(found):
                contour = contour_of(mask, c)
                s = score_shape(contour, reference)
                near = (abs(c["center_x"] - ARROW_X) < 60 and abs(c["center_y"] - ARROW_Y) < 60)
                rows.append({"frame": frame_no, "index": i, "near_arrow": int(near),
                             "score": s, "area": c["area"],
                             "cx": c["center_x"], "cy": c["center_y"],
                             "w": c["width"], "h": c["height"]})
            if frame_no % 40 == 0:
                vis = overlay(frame, mask, found)
                for r in [r for r in rows if r["frame"] == frame_no]:
                    x, y = int(r["cx"]), int(r["cy"])
                    txt = "%.2f" % r["score"] if r["score"] == r["score"] else "nan"
                    cv2.putText(vis, txt, (x + 4, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 255), 1, cv2.LINE_AA)
                cv2.imwrite(os.path.join(args.out, "overlay_%04d.png" % frame_no), vis)
            frame_no += 1
    finally:
        stream.close()

    path = os.path.join(args.out, "shape_scores.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "index", "near_arrow", "score",
                                          "area", "cx", "cy", "w", "h"])
        w.writeheader()
        w.writerows(rows)

    near = [r["score"] for r in rows if r["near_arrow"] and r["score"] == r["score"]]
    far = [r["score"] for r in rows if not r["near_arrow"] and r["score"] == r["score"]]
    print()
    print("frames %d, candidates %d (%d near the arrow, %d elsewhere)"
          % (frame_no, len(rows), len(near), len(far)))
    if near:
        print("  near arrow   n=%3d  score min %.3f  median %.3f  max %.3f"
              % (len(near), min(near), statistics.median(near), max(near)))
    if far:
        print("  elsewhere    n=%3d  score min %.3f  median %.3f  max %.3f"
              % (len(far), min(far), statistics.median(far), max(far)))
    if near and far:
        print("  separation   arrow median %.3f vs others median %.3f  (gap %.3f)"
              % (statistics.median(near), statistics.median(far),
                 statistics.median(far) - statistics.median(near)))
        best = min(near)
        print("  if the cut is the arrow's worst (%.3f): keeps %d/%d arrow candidates, %d/%d others"
              % (max(near), sum(1 for s in near if s <= max(near)), len(near),
                 sum(1 for s in far if s <= max(near)), len(far)))
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
