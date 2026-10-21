#!/usr/bin/env python3
"""Phase 3: the angle. Shape matching cannot give a heading, a rotated template can.

One template is cut from the arrow itself (at the position Phase 1 measured), then rotated
through 24 angles. For each candidate - and only inside that candidate's ROI, never the full
frame - every rotation is scored with cv2.matchTemplate and the best pair is kept.

That gives best score and best angle per candidate. The score says "this blob is the arrow",
the angle says which way it points. The question this answers is whether the angle is steady
enough to be a heading, and what it costs.

Usage:
    python template_match.py --frames 200 --out out/phase3
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

from hsv_candidates import Stream, candidates, load_config                 # noqa: E402
from shape_match import ARROW_X, ARROW_Y, contour_of, score_shape         # noqa: E402

TEMPLATE_SIZE = 40        # square crop, big enough to hold the arrow plus a little context
ANGLE_STEP = 15           # 24 templates
ROI_PAD = 8               # extra pixels around a candidate's bbox before matching


def square_crop(frame: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray | None:
    half = size // 2
    x0, y0 = int(cx) - half, int(cy) - half
    if x0 < 0 or y0 < 0 or x0 + size > frame.shape[1] or y0 + size > frame.shape[0]:
        return None
    return frame[y0:y0 + size, x0:x0 + size].copy()


def rotations(tpl: np.ndarray, step: int = ANGLE_STEP) -> list[tuple[float, np.ndarray]]:
    """(angle, image) for every rotation. Angle is the rotation applied, in degrees."""
    h, w = tpl.shape[:2]
    centre = (w / 2.0, h / 2.0)
    out = []
    for angle in range(0, 360, step):
        m = cv2.getRotationMatrix2D(centre, angle, 1.0)
        rot = cv2.warpAffine(tpl, m, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        out.append((float(angle), rot))
    return out


def match_roi(frame: np.ndarray, c: dict, rots: list[tuple[float, np.ndarray]]
              ) -> tuple[float, float] | tuple[None, None]:
    """Best (score, angle) inside this candidate's ROI only."""
    x, y, w, h = c["bbox"]
    th, tw = rots[0][1].shape[:2]
    x0 = max(0, x - ROI_PAD)
    y0 = max(0, y - ROI_PAD)
    x1 = min(frame.shape[1], x + w + ROI_PAD)
    y1 = min(frame.shape[0], y + h + ROI_PAD)
    roi = frame[y0:y1, x0:x1]

    # A ROI smaller than the template cannot be matched; pad it rather than skip the candidate.
    if roi.shape[0] < th or roi.shape[1] < tw:
        roi = cv2.copyMakeBorder(roi, 0, max(0, th - roi.shape[0]),
                                 0, max(0, tw - roi.shape[1]),
                                 cv2.BORDER_REPLICATE)
    if roi.shape[0] < th or roi.shape[1] < tw:
        return None, None

    best_score, best_angle = -2.0, None
    for angle, rot in rots:
        res = cv2.matchTemplate(roi, rot, cv2.TM_CCOEFF_NORMED)
        _, mx, _, _ = cv2.minMaxLoc(res)
        if mx > best_score:
            best_score, best_angle = float(mx), angle
    return best_score, best_angle


def pick_template(cfg: dict, url: str) -> tuple[np.ndarray, dict]:
    stream = Stream(url)
    try:
        for _ in range(60):
            frame = stream.next()
            if frame is None:
                break
            _mask, found = candidates(frame, cfg)
            near = [c for c in found
                    if abs(c["center_x"] - ARROW_X) < 40 and abs(c["center_y"] - ARROW_Y) < 40]
            if not near:
                continue
            c = max(near, key=lambda q: q["area"])
            tpl = square_crop(frame, c["center_x"], c["center_y"], TEMPLATE_SIZE)
            if tpl is not None:
                return tpl, c
    finally:
        stream.close()
    raise SystemExit("could not cut a template at the arrow position")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--url", default="http://127.0.0.1:8021/stream")
    ap.add_argument("--config", default=os.path.join(HERE, "config.orangered.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "out", "phase3"))
    ap.add_argument("--shape-gate", type=float, default=0.415,
                    help="only score by template when phase 2 shape score is under this")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)

    tpl, c = pick_template(cfg, args.url)
    cv2.imwrite(os.path.join(args.out, "template.png"), tpl)
    print("template: %dx%d cut at (%.1f, %.1f), area %d"
          % (tpl.shape[1], tpl.shape[0], c["center_x"], c["center_y"], c["area"]))
    rots = rotations(tpl)
    sheet = np.hstack([r for _, r in rots])
    cv2.imwrite(os.path.join(args.out, "template_rotations.png"), sheet)
    print("rotations: %d (step %d deg) -> template_rotations.png" % (len(rots), ANGLE_STEP))

    # The reference contour for the phase 2 gate, from the same template region.
    ref_path = os.path.join(HERE, "out", "phase2", "reference_arrow.json")
    reference = None
    if os.path.exists(ref_path):
        reference = np.array(json.load(open(ref_path, encoding="utf-8"))["contour"],
                             np.int32).reshape(-1, 1, 2)

    rows = []
    stream = Stream(args.url)
    frame_no = 0
    t_total = 0.0
    try:
        while frame_no < args.frames:
            frame = stream.next()
            if frame is None:
                break
            mask, found = candidates(frame, cfg)
            for i, cand in enumerate(found):
                t0 = cv2.getTickCount()
                score, angle = match_roi(frame, cand, rots)
                t_total += (cv2.getTickCount() - t0) / cv2.getTickFrequency()

                shape = float("nan")
                if reference is not None:
                    shape = score_shape(contour_of(mask, cand), reference)
                near = (abs(cand["center_x"] - ARROW_X) < 60
                        and abs(cand["center_y"] - ARROW_Y) < 60)
                rows.append({"frame": frame_no, "index": i, "near_arrow": int(near),
                             "shape": shape,
                             "tmpl_score": "" if score is None else round(score, 4),
                             "tmpl_angle": "" if angle is None else angle,
                             "cx": cand["center_x"], "cy": cand["center_y"],
                             "area": cand["area"]})
            if frame_no % 40 == 0:
                vis = frame.copy()
                for r in [r for r in rows if r["frame"] == frame_no]:
                    if r["tmpl_score"] == "":
                        continue
                    x, y = int(r["cx"]), int(r["cy"])
                    ok = float(r["tmpl_score"]) > 0.5 and r["shape"] <= args.shape_gate
                    colour = (0, 255, 0) if ok else (0, 0, 255)
                    cv2.circle(vis, (x, y), 3, colour, -1)
                    cv2.putText(vis, "%.2f@%d" % (float(r["tmpl_score"]), r["tmpl_angle"]),
                                (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1,
                                cv2.LINE_AA)
                    if ok:
                        a = np.deg2rad(r["tmpl_angle"])
                        cv2.arrowedLine(vis, (x, y), (x + int(30 * np.cos(a)),
                                                      y - int(30 * np.sin(a))),
                                        colour, 2, tipLength=0.3)
                cv2.imwrite(os.path.join(args.out, "overlay_%04d.png" % frame_no), vis)
            frame_no += 1
    finally:
        stream.close()

    path = os.path.join(args.out, "template_scores.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "index", "near_arrow", "shape",
                                          "tmpl_score", "tmpl_angle", "cx", "cy", "area"])
        w.writeheader()
        w.writerows(rows)

    def nums(key, group):
        out = []
        for r in rows:
            if r["near_arrow"] != group or r[key] == "":
                continue
            out.append(float(r[key]))
        return out

    print()
    print("frames %d, candidates %d" % (frame_no, len(rows)))
    for group, label in ((1, "near the arrow"), (0, "elsewhere       ")):
        s = nums("tmpl_score", group)
        a = nums("tmpl_angle", group)
        if s:
            print("  %s n=%3d  score median %.3f  |  angle median %s"
                  % (label, len(s), statistics.median(s),
                     ("%.0f deg" % statistics.median(a)) if a else "-"))
    arrow_scores = nums("tmpl_score", 1)
    arrow_angles = nums("tmpl_angle", 1)
    if arrow_scores and arrow_angles:
        print()
        print("  arrow angle: %d distinct values over %d frames"
              % (len(set(arrow_angles)), len(arrow_angles)))
        hist = {}
        for a in arrow_angles:
            hist[int(a)] = hist.get(int(a), 0) + 1
        print("  angle histogram: " + "  ".join("%d:%d" % (k, hist[k]) for k in sorted(hist)))

    gated = [r for r in rows
             if r["tmpl_score"] != "" and float(r["tmpl_score"]) > 0.5
             and (r["shape"] != r["shape"] or r["shape"] <= args.shape_gate)]
    print("  after shape<=%.3f AND template>0.5: %d candidates from %d frames"
          % (args.shape_gate, len(gated), len({r["frame"] for r in gated})))
    print("  template matching cost: %.2f ms per candidate (24 rotations, ROI only)"
          % (1000.0 * t_total / max(1, len(rows))))
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
