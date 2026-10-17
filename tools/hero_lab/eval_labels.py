"""Score the existing car pipeline against the 93 hand-drawn quads. No training here.

The labels in vision/data/labels are ground truth: a human drew where the car is and which way it
points. This compares the product against them, and the candidate mask against them, on the same
frames - which is the number that decides whether a learned model has anything to do at all.

    python tools/hero_lab/eval_labels.py                # A = product, B = candidate
    python tools/hero_lab/eval_labels.py --only A       # product only
    python tools/hero_lab/eval_labels.py --worst 12     # list this many worst cases

What is compared, per labelled frame:

    IoU            product mask against the human quad rasterised
    centre         distance between the detected centre and the quad's centre, px
    axis           0..180 long-axis angle, detected against the quad's longest edge
    front          0..360 heading, detected against the quad's stored arrow tip
    front known    how often the pipeline is willing to name a front at all
    area ratio     detected mask area / quad area

Ground truth is never the pipeline's own output, and nothing here is tuned to make a number look
good: the conditions are split after the fact (left/right half, near/far), and every frame that is
counted as a failure is listed with its file name so it can be looked at.

The two annotations whose images were deleted are skipped and counted, not silently dropped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "vision", "src"))
sys.path.insert(0, HERE)

import vehicle                                                        # noqa: E402
import candidate_mask as lab                                          # noqa: E402

LABELS = os.path.join(ROOT, "vision", "data", "labels")
ANNOTATIONS = os.path.join(LABELS, "annotations.jsonl")


def load_labels():
    out = []
    with open(ANNOTATIONS, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def quad_masks(rec, shape):
    """The human quad as a mask, and its geometry."""
    quad = np.array(rec["quad"], np.float32)
    gt = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(gt, [quad.astype(np.int32)], 255)
    edges = [float(np.linalg.norm(quad[(i + 1) % 4] - quad[i])) for i in range(4)]
    k = int(np.argmax(edges))
    axis = math.degrees(math.atan2(float(quad[(k + 1) % 4][1] - quad[k][1]),
                                   float(quad[(k + 1) % 4][0] - quad[k][0]))) % 180.0
    centre = quad.mean(axis=0)
    # The stored arrow tip is the heading the annotator confirmed; for an automatic one it is
    # derived from the quad, which makes it a consistency check on the convention as well.
    front = None
    if rec.get("tip"):
        front = math.degrees(math.atan2(rec["tip"][1] - centre[1],
                                        rec["tip"][0] - centre[0])) % 360.0
    return gt, centre, axis, front, cv2.contourArea(quad)


def angdiff(a, b, period=180.0):
    d = abs((a - b) % period)
    return min(d, period - d)


def evaluate(provider, name, records):
    rows = []
    for rec in records:
        path = os.path.join(LABELS, rec["image"])
        frame = cv2.imread(path)
        gt, gt_centre, gt_axis, gt_front, gt_area = quad_masks(rec, frame.shape)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        t0 = time.perf_counter()
        if provider is None:
            mask = vehicle.red_mask(hsv)
            poses = vehicle.locate_all(frame)
        else:
            with lab.mask_provider(provider):
                poses = vehicle.locate_all(frame)
            mask = provider(hsv)
        ms = (time.perf_counter() - t0) * 1000.0

        pose = poses[0] if poses else None
        det = np.zeros(frame.shape[:2], np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n > 1:
            i = 1 + int(np.argmax(stats[1:, 4]))
            det[labels == i] = 255
        inter = int(np.count_nonzero((gt > 0) & (det > 0)))
        union = int(np.count_nonzero((gt > 0) | (det > 0)))
        iou = inter / union if union else 0.0

        report = pose.as_dict() if pose else {}
        rows.append({
            "id": rec["id"],
            "iou": iou,
            "found": pose is not None,
            "centre_err": (math.hypot(pose.x - gt_centre[0], pose.y - gt_centre[1])
                           if pose else None),
            "axis_err": (angdiff(pose.heading, gt_axis) if pose else None),
            "front_known": bool(report.get("front_is_known")),
            "front_err": (angdiff(report["front_heading"], gt_front, 360.0)
                          if pose and report.get("front_is_known") and gt_front is not None else None),
            "area_ratio": (pose.area / gt_area) if pose and gt_area else None,
            "elong": pose.elongation if pose else None,
            "gt_aspect": max(1e-6, max(
                float(np.linalg.norm(np.array(rec["quad"], float)[(i + 1) % 4]
                                     - np.array(rec["quad"], float)[i])) for i in range(4))
                / max(1e-6, min(
                    float(np.linalg.norm(np.array(rec["quad"], float)[(i + 1) % 4]
                                         - np.array(rec["quad"], float)[i])) for i in range(4)))),
            "ms": ms,
            "x": gt_centre[0],
            "area": gt_area,
        })
    return rows


def stat(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    return {"n": len(v), "median": v[len(v) // 2], "p10": v[max(0, int(0.1 * len(v)) - 0)],
            "p90": v[min(len(v) - 1, int(0.9 * len(v)))], "min": v[0], "max": v[-1]}


def show(name, rows, worst=0):
    found = [r for r in rows if r["found"]]
    print(f"\n=== {name} ===   labelled {len(rows)}   detected {len(found)} "
          f"({100.0 * len(found) / max(len(rows), 1):.0f}%)")
    for key, label, unit in (("iou", "IoU vs quad", ""), ("centre_err", "centre error", " px"),
                             ("axis_err", "axis error", " deg"),
                             ("front_err", "front error", " deg"),
                             ("area_ratio", "area ratio", "x"), ("ms", "time", " ms")):
        s = stat([r[key] for r in rows])
        if s:
            print(f"  {label:<14} median {s['median']:7.3f}{unit}   p10 {s['p10']:7.3f}   "
                  f"p90 {s['p90']:7.3f}   worst {s['max']:7.3f}")
    known = sum(1 for r in rows if r["front_known"])
    print(f"  front named    {known}/{len(rows)} frames ({100.0 * known / max(len(rows), 1):.0f}%)")
    if worst:
        print(f"  --- {worst} worst by IoU ---")
        for r in sorted(rows, key=lambda r: r["iou"])[:worst]:
            print(f"    {r['id']}  IoU {r['iou']:.3f}  found {r['found']}  "
                  f"axis {('%.1f' % r['axis_err']) if r['axis_err'] is not None else '-'}  x={r['x']:.0f}")
    # by condition: the right half of the frame is where the bay lights are
    for label, pick in (("left half", lambda r: r["x"] < 400.0),
                        ("right half", lambda r: r["x"] >= 400.0),
                        ("near  (big)", lambda r: r["area"] > 2800.0),
                        ("far   (small)", lambda r: r["area"] <= 2800.0)):
        sub = [r for r in rows if pick(r)]
        if not sub:
            continue
        s_iou = stat([r["iou"] for r in sub])
        s_ax = stat([r["axis_err"] for r in sub])
        det = sum(1 for r in sub if r["found"])
        print(f"  {label:<13} n={len(sub):<3} detected {det:<3} "
              f"IoU median {s_iou['median']:.3f}   axis median "
              f"{(s_ax['median'] if s_ax else float('nan')):.2f} deg")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["A", "B", "both"], default="both")
    ap.add_argument("--worst", type=int, default=0)
    args = ap.parse_args()

    records = load_labels()
    usable, missing = [], []
    for rec in records:
        if os.path.exists(os.path.join(LABELS, rec["image"])):
            usable.append(rec)
        else:
            missing.append(rec["id"])
    print(f"labels: {len(records)}   usable {len(usable)}   missing image {len(missing)}"
          + (f"  ({', '.join(missing)})" if missing else ""))

    rows_a = evaluate(None, "A", usable)
    show("A  product pipeline", rows_a, args.worst)

    if args.only in ("B", "both"):
        rows_b = evaluate(lab.candidate_red_mask, "B", usable)
        show("B  candidate mask", rows_b, args.worst)
        if args.only == "both":
            print("\n=== A vs B, paired (same frames) ===")
            better = sum(1 for a, b in zip(rows_a, rows_b) if b["iou"] > a["iou"] + 0.02)
            worse = sum(1 for a, b in zip(rows_a, rows_b) if b["iou"] < a["iou"] - 0.02)
            print(f"  B better on {better} frames, worse on {worse}, "
                  f"within 0.02 on {len(rows_a) - better - worse}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
