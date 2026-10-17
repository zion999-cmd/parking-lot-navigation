"""Three-way comparison on the labelled frames: product, candidate mask, reference differencing.

    python tools/hero_lab/diff_reference.py                    # uses reference_new.jpg if present
    python tools/hero_lab/diff_reference.py --reference vision/data/reference/reference.jpg
    python tools/hero_lab/diff_reference.py --skip AB          # reference differencing only

Why differencing is worth measuring at all: the camera and the carpark are static, so the car is the
only thing that changes between frames. A difference against an empty reference sees the whole car
body rather than only its yellow paint, and it cannot be fooled by the bay lights - they are in the
reference too and cancel out. It costs a few milliseconds. The two failures the 139 labels exposed -
a mask covering a quarter of the car in near-end-on views, and the light column at the right edge
being detected instead of the car - are both exactly the failures differencing should not have.

It is measured the same way as the others: the difference mask is fed to the *unmodified* feature
stage through mask_provider, so the quad-derived ground truth, the IoU, the centre, the axis and the
front all come from the same code for all three.

A stale reference makes this worthless in a way that looks like a bug - the first run against the
10 October reference differed over 70,000 px of a frame where the car is 2,400, i.e. the whole scene
had changed and no amount of alignment fixed it (phase correlation response ~0.0). Hence
capture_reference.py and the freshness line printed here before any number is trusted.
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
import eval_labels as E                                               # noqa: E402

REF_DIR = os.path.join(ROOT, "vision", "data", "reference")
OLD = os.path.join(REF_DIR, "reference.jpg")
NEW = os.path.join(REF_DIR, "reference_new.jpg")


def load_reference(path: str):
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"  cannot read {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _align_light(value: np.ndarray, ref_v: np.ndarray) -> np.ndarray:
    """Put the frame's brightness on the reference's scale, robustly.

    Measured why this is needed: a reference captured at 11:53 differs from frames captured between
    09:11 and 11:2x in 40% of their pixels, because the light moved - not because the car did. Raw
    differencing then reports the whole room as foreground and finds the car in 0 of 139 frames.
    Median and the 10-90 percentile spread are used rather than mean and standard deviation because
    the car itself is a few thousand brighter or darker pixels among 480,000 and must not drag the
    fit.
    """
    now = value.astype(np.float32)
    ref = ref_v.astype(np.float32)
    m_now, s_now = float(np.median(now)), float(np.percentile(now, 90) - np.percentile(now, 10))
    m_ref, s_ref = float(np.median(ref)), float(np.percentile(ref, 90) - np.percentile(ref, 10))
    scale = s_ref / s_now if s_now > 1e-6 else 1.0
    return np.clip((now - m_now) * scale + m_ref, 0, 255)


def diff_mask(hsv: np.ndarray, ref_v: np.ndarray, threshold: int = 18,
              min_area: int = 150, normalise: bool = False) -> np.ndarray:
    """The car as the one thing that differs from the empty reference, in the largest piece."""
    value = hsv[:, :, 2]
    if normalise:
        value = _align_light(value, ref_v)
    if value.shape != ref_v.shape:
        return np.zeros(value.shape, np.uint8)
    # float32 on both sides: the normalised path returns floats while the reference is uint8
    diff = cv2.absdiff(np.asarray(value, np.float32), np.asarray(ref_v, np.float32))
    mask = (diff > threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return np.zeros(value.shape, np.uint8)
    i = 1 + int(np.argmax(stats[1:, 4]))
    if int(stats[i, 4]) < min_area:
        return np.zeros(value.shape, np.uint8)
    return np.where(labels == i, 255, 0).astype(np.uint8)


def provider_for(ref_v: np.ndarray, normalise: bool = False):
    def provider(hsv, close_kernel=None, open_kernel=None):
        return diff_mask(hsv, ref_v, normalise=normalise)
    return provider


def rows_for(name, provider, records):
    rows = []
    for rec in records:
        frame = cv2.imread(os.path.join(E.LABELS, rec["image"]))
        gt, gt_centre, gt_axis, gt_front, gt_area = E.quad_masks(rec, frame.shape)
        t0 = time.perf_counter()
        if provider is None:
            poses = vehicle.locate_all(frame)
            mask = vehicle.red_mask(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
        else:
            with lab.mask_provider(provider):
                poses = vehicle.locate_all(frame)
            mask = provider(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
        ms = (time.perf_counter() - t0) * 1000.0
        det = np.zeros(frame.shape[:2], np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if count > 1:
            i = 1 + int(np.argmax(stats[1:, 4]))
            det[labels == i] = 255
        inter = int(np.count_nonzero((gt > 0) & (det > 0)))
        union = int(np.count_nonzero((gt > 0) | (det > 0)))
        pose = poses[0] if poses else None
        report = pose.as_dict() if pose else {}
        rows.append({
            "id": rec["id"], "iou": inter / union if union else 0.0,
            "found": pose is not None,
            "centre_err": (math.hypot(pose.x - gt_centre[0], pose.y - gt_centre[1])
                           if pose else None),
            "axis_err": (E.angdiff(pose.heading, gt_axis) if pose else None),
            "front_err": (E.angdiff(report["front_heading"], gt_front, 360.0)
                          if pose and report.get("front_is_known") and gt_front is not None else None),
            "front_known": bool(report.get("front_is_known")),
            "area_ratio": (pose.area / gt_area) if pose and gt_area else None,
            "ms": ms,
        })
    return rows


def summarise(name, rows):
    ious = sorted(r["iou"] for r in rows)
    def med(key):
        v = sorted(x for x in (r[key] for r in rows) if x is not None)
        return v[len(v) // 2] if v else float("nan")
    def p90(key):
        v = sorted(x for x in (r[key] for r in rows) if x is not None)
        return v[min(len(v) - 1, int(0.9 * len(v)))] if v else float("nan")
    bad = sum(1 for r in rows if r["iou"] < 0.30)
    good = sum(1 for r in rows if r["iou"] > 0.70)
    det = sum(1 for r in rows if r["found"])
    print(f"  {name:<22} IoU med {ious[len(ious)//2]:.3f}  ({bad} frames <0.30, {good} >0.70)   "
          f"centre {med('centre_err'):5.1f}px  axis {med('axis_err'):4.1f}deg (p90 {p90('axis_err'):4.1f})  "
          f"front {med('front_err'):4.1f}deg  area x{med('area_ratio'):.2f}  "
          f"det {det}/{len(rows)}  {med('ms'):5.1f} ms")
    return {"iou": ious[len(ious) // 2], "bad": bad, "good": good}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=NEW if os.path.exists(NEW) else OLD)
    ap.add_argument("--skip", default="", help="letters of groups to skip, e.g. AB")
    ap.add_argument("--normalise", action="store_true",
                    help="put the frame's brightness on the reference's scale before differencing")
    args = ap.parse_args()

    ref_v = load_reference(args.reference)
    print(f"reference: {os.path.relpath(args.reference, ROOT)}   ({time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(args.reference)))})")

    records = [r for r in E.load_labels()
               if os.path.exists(os.path.join(E.LABELS, r["image"]))]
    print(f"labelled frames: {len(records)}")

    # Freshness first: a reference that no longer matches the scene makes every number below a lie.
    fresh = [cv2.imread(os.path.join(E.LABELS, r["image"])) for r in records[:20]]
    changed = []
    for frame in fresh:
        if frame is None:
            continue
        d = cv2.absdiff(np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), np.float32),
                        np.asarray(ref_v, np.float32))
        changed.append(float(np.count_nonzero(d > 18)) / d.size * 100.0)
    if changed:
        print(f"reference freshness: frames differ from it in {np.median(changed):.1f}% of pixels "
              f"(the car alone is under 1%; 15%+ means the scene changed)")

    results = {}
    results["A"] = summarise("A  product", rows_for("A", None, records))
    if "B" not in args.skip:
        results["B"] = summarise("B  candidate mask", rows_for("B", lab.candidate_red_mask, records))
    if "C" not in args.skip:
        results["C"] = summarise("C  reference diff" + (" + norm" if args.normalise else ""),
                                 rows_for("C", provider_for(ref_v, args.normalise), records))

    if "A" in results and "C" in results:
        print(f"\n  C against A: {results['C']['good'] - results['A']['good']:+d} more frames above "
              f"IoU 0.70, {results['C']['bad'] - results['A']['bad']:+d} more below 0.30")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
