#!/usr/bin/env python3
"""Do the arrow fragments regroup into something the shape test recognises?

The HSV mask breaks the arrow into pieces - head, shaft, sometimes a sliver at the frame edge -
and the current pipeline hands each piece to matchShapes on its own, so a two-fragment arrow
scores like two wrong shapes instead of one right one. This runs the fix offline and measures it,
before anything in the live path changes:

    HSV (loose, three saturation levels)
      -> connected components        (fragments, not candidates)
      -> dilate ONLY to decide grouping
      -> take the group's REAL pixels from the undilated mask
      -> merged contour, geometry, physical gates
      -> shape against a bank of REAL sampled references, leave-one-out
      -> tiers: <=0.30 strong, <=0.60 weak

Three metrics, kept apart because they fail for different reasons:

    A recall     the true arrow is in the candidate pool at all
    B selection  the best candidate in the pool is the true arrow
    C stability  needs consecutive frames; reported as pool size here, measured separately

Usage:
    python fragment_grouping.py --calib vision/data/hsv_marker/calibration.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "vision", "src"))

import hsv_marker       # noqa: E402

S_LEVELS = (60, 50, 45)
GROUP_KERNEL = 9        # px; the gap that still counts as "the same object"
MIN_FRAGMENT = 12       # px; anything smaller is noise, not evidence

# Physical gates for a group. Deliberately loose - the point of this stage is recall, and the
# belief layer keeps the identity decision - but a red wall must not become a candidate.
GATE_AREA = (80, 900)
GATE_SIDE = (8, 60)
GATE_ASPECT = (0.30, 3.2)
GATE_FILL = 0.22
GATE_SPAN = 130         # a group stretching further than this in either axis is not an arrow

STRONG = 0.30
WEAK = 0.60


# ----------------------------------------------------------------- geometry helpers


def fragments(mask: np.ndarray) -> list[dict]:
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < MIN_FRAGMENT:
            continue
        out.append({"id": i, "bbox": (x, y, w, h), "area": area,
                    "centre": (float(cents[i][0]), float(cents[i][1]))})
    return out


def group_fragments(mask: np.ndarray, frags: list[dict], k: int = GROUP_KERNEL) -> list[dict]:
    """Dilate only to decide who belongs with whom; pixels always come from `mask`.

    k <= 1 disables grouping: every fragment is its own group, which is the current pipeline.
    """
    if not frags:
        return []
    if k <= 1:
        return [{"group": f["id"], "frags": [f]} for f in frags]
    dil = cv2.dilate(mask, np.ones((k, k), np.uint8))
    gn, glabels = cv2.connectedComponents(dil)

    buckets: dict[int, list[dict]] = {}
    for f in frags:
        x, y, w, h = f["bbox"]
        # Majority vote over the fragment's own pixels: one pixel is noise, a label is a group.
        votes = glabels[y:y + h, x:x + w][mask[y:y + h, x:x + w] > 0]
        if votes.size == 0:
            continue
        gid = int(np.bincount(votes).argmax())
        if gid == 0:
            continue
        buckets.setdefault(gid, []).append(f)
    return [{"group": gid, "frags": fs} for gid, fs in buckets.items()]


def build_candidate(mask: np.ndarray, group: dict, hsv: np.ndarray) -> dict | None:
    """Real pixels of every fragment in the group, one merged contour."""
    fs = group["frags"]
    x0 = min(f["bbox"][0] for f in fs)
    y0 = min(f["bbox"][1] for f in fs)
    x1 = max(f["bbox"][0] + f["bbox"][2] for f in fs)
    y1 = max(f["bbox"][1] + f["bbox"][3] for f in fs)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None

    # The group's real red pixels: the original mask, restricted to the group's fragments.
    keep = np.zeros((h, w), np.uint8)
    for f in fs:
        fx, fy, fw, fh = f["bbox"]
        sub = mask[fy:fy + fh, fx:fx + fw]
        keep[fy - y0:fy - y0 + fh, fx - x0:fx - x0 + fw] |= (sub > 0).astype(np.uint8)

    area = int(keep.sum())
    contours, _ = cv2.findContours(keep * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    if len(contours) == 1:
        contour = contours[0]
    else:
        # Several fragments that dilation says belong together: their union outline is the
        # arrow, and a hull is the least inventive way to say that.
        pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.int32).reshape(-1, 1, 2)
        contour = cv2.convexHull(pts)

    aspect = w / float(h)
    fill = area / float(w * h)
    sub_hsv = hsv[y0:y1, x0:x1]
    inside = keep > 0
    return {
        "bbox": (x0, y0, w, h),
        "centre": (x0 + float(np.argwhere(inside)[1].mean()) if inside.any() else x0 + w / 2.0,
                   y0 + float(np.argwhere(inside)[0].mean()) if inside.any() else y0 + h / 2.0),
        "area": area, "width": w, "height": h, "aspect": aspect, "fill": fill,
        "fragments": len(fs),
        "contour": contour,
        "h": float(np.median(sub_hsv[:, :, 0][inside])) if inside.any() else 0.0,
        "s": float(np.median(sub_hsv[:, :, 1][inside])) if inside.any() else 0.0,
        "v": float(np.median(sub_hsv[:, :, 2][inside])) if inside.any() else 0.0,
    }


def passes_gates(c: dict) -> bool:
    if not (GATE_AREA[0] <= c["area"] <= GATE_AREA[1]):
        return False
    if not (GATE_SIDE[0] <= c["width"] <= GATE_SIDE[1]):
        return False
    if not (GATE_SIDE[0] <= c["height"] <= GATE_SIDE[1]):
        return False
    if not (GATE_ASPECT[0] <= c["aspect"] <= GATE_ASPECT[1]):
        return False
    if c["fill"] < GATE_FILL:
        return False
    if c["width"] > GATE_SPAN or c["height"] > GATE_SPAN:
        return False
    return True


# ----------------------------------------------------------------- reference bank


def rotated(contour: np.ndarray, step: int = 15) -> list[np.ndarray]:
    out = []
    for a in range(0, 360, step):
        m = cv2.getRotationMatrix2D((0, 0), a, 1.0)
        pts = contour.reshape(-1, 2).astype(np.float64) @ m[:, :2].T
        out.append(pts.reshape(-1, 1, 2).astype(np.int32))
    return out


def shape_score(contour: np.ndarray, refs: list[np.ndarray]) -> tuple[float, int]:
    """Best score over the bank, and which entry produced it."""
    best, which = 9.9, -1
    for i, ref in enumerate(refs):
        v = float(cv2.matchShapes(contour, ref, cv2.CONTOURS_MATCH_I1, 0.0))
        if v < best:
            best, which = v, i
    return best, which


def tier(score: float) -> str:
    if score <= STRONG:
        return "strong"
    if score <= WEAK:
        return "weak"
    return "reject"


# ----------------------------------------------------------------- the run


def label_of(batch: str) -> tuple[str, int]:
    m = re.match(r"^(\D*?)[\s_-]*(\d{1,3})\s*°?$", batch.strip())
    if m:
        return m.group(1), int(m.group(2))
    return batch.strip(), -1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calib", default=os.path.join(ROOT, "vision", "data", "hsv_marker",
                                                    "calibration.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "out", "fragment_grouping"))
    ap.add_argument("--tol", type=float, default=14.0, help="px; a candidate this close is the arrow")
    ap.add_argument("--kernel", type=int, default=GROUP_KERNEL,
                    help="grouping dilation; <=1 disables grouping (each fragment stands alone)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "overlay"), exist_ok=True)

    rows = []
    with open(args.calib, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("no calibration samples")
        return 1

    # Load every sample once, with its masks and groups.
    samples = []
    for r in rows:
        pos, ang = label_of(r.get("batch") or "")
        path = os.path.join(ROOT, "vision", "data", "hsv_marker", "frames", r.get("frame") or "")
        frame = cv2.imread(path) if r.get("frame") else None
        if frame is None:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        truth = (float(r["centre"][0]), float(r["centre"][1]))
        levels = {}
        for s_min in S_LEVELS:
            mask, _n, _st, _ce = hsv_marker._components(hsv, s_min, hsv_marker.V_MIN)
            levels[s_min] = mask
        samples.append({"rec": r, "pos": pos, "ang": ang, "frame": frame, "hsv": hsv,
                        "truth": truth, "levels": levels})

    print("samples %d" % len(samples))

    # A reference bank from REAL sampled contours: one per orientation, spread across positions.
    # Leave-one-out is enforced below, so no sample ever matches its own contour.
    bank_src = []
    for s in samples:
        mask = s["levels"].get(s["rec"].get("s_min_used") or 60)
        if mask is None:
            continue
        x, y, w, h = s["rec"]["bbox"]
        sub = (mask[y:y + h, x:x + w] > 0).astype(np.uint8)
        cs, _ = cv2.findContours(sub * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cs:
            bank_src.append({"key": s["rec"]["frame"], "pos": s["pos"], "ang": s["ang"],
                             "contour": max(cs, key=cv2.contourArea)})
    print("reference bank candidates: %d real contours" % len(bank_src))

    out_rows = []
    for s in samples:
        rec = s["rec"]
        single_ok = None       # baseline: the recorded component, single reference
        group_ok = None        # grouping, single reference
        multi_ok = None        # grouping, real bank
        pool_sizes = {}

        # ---- baseline: the component the operator clicked, one reference
        ref0 = hsv_marker._load_reference()
        mask = s["levels"].get(rec.get("s_min_used") or 60)
        base_near = False
        if mask is not None:
            x, y, w, h = rec["bbox"]
            sub = (mask[y:y + h, x:x + w] > 0).astype(np.uint8)
            cs, _ = cv2.findContours(sub * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if cs and ref0 is not None:
                sc0 = float(cv2.matchShapes(max(cs, key=cv2.contourArea), ref0,
                                            cv2.CONTOURS_MATCH_I1, 0.0))
                single_ok = tier(sc0) != "reject"
            else:
                sc0 = 9.9
        else:
            sc0 = 9.9

        # ---- grouping with one reference, then with the real bank
        for mode in ("single", "multi"):
            refs = [ref0] if mode == "single" else \
                   [b["contour"] for b in bank_src if b["key"] != rec["frame"]]
            if mode == "multi":
                refs = refs + rotated(ref0)[:8]     # a little programmatic coverage as backup
            pool = []
            for s_min, m in s["levels"].items():
                frags = fragments(m)
                for g in group_fragments(m, frags, args.kernel):
                    c = build_candidate(m, g, s["hsv"])
                    if c is None or not passes_gates(c):
                        continue
                    sc, which = shape_score(c["contour"], refs)
                    if sc > WEAK:
                        continue
                    c["shape_score"] = sc
                    c["ref_index"] = which
                    c["s_min"] = s_min
                    pool.append(c)
            # de-duplicate across saturation levels
            pool.sort(key=lambda c: c["shape_score"])
            merged = []
            for c in pool:
                if any(abs(c["centre"][0] - m2["centre"][0]) < 10
                       and abs(c["centre"][1] - m2["centre"][1]) < 10 for m2 in merged):
                    continue
                merged.append(c)
            pool_sizes[mode] = len(merged)
            has_true = any(abs(c["centre"][0] - s["truth"][0]) <= args.tol
                           and abs(c["centre"][1] - s["truth"][1]) <= args.tol for c in merged)
            best_is_true = bool(merged) and (
                abs(merged[0]["centre"][0] - s["truth"][0]) <= args.tol
                and abs(merged[0]["centre"][1] - s["truth"][1]) <= args.tol)
            if mode == "single":
                group_ok = (has_true, best_is_true)
            else:
                multi_ok = (has_true, best_is_true)

        out_rows.append({
            "batch": rec.get("batch"), "pos": s["pos"], "ang": s["ang"],
            "recorded_shape": rec.get("shape_score"), "baseline_shape": round(sc0, 4),
            "baseline_in_pool": int(bool(single_ok)),
            "group_single_recall": int(group_ok[0]), "group_single_select": int(group_ok[1]),
            "group_multi_recall": int(multi_ok[0]), "group_multi_select": int(multi_ok[1]),
            "pool_single": pool_sizes.get("single", 0),
            "pool_multi": pool_sizes.get("multi", 0),
        })

        # debug overlay: fragments, group boxes, merged contours
        vis = s["frame"].copy()
        for s_min, m in s["levels"].items():
            for f in fragments(m):
                x, y, w, h = f["bbox"]
                cv2.rectangle(vis, (x, y), (x + w, y + h), (90, 90, 90), 1)
        refs = [ref0] + [b["contour"] for b in bank_src if b["key"] != rec["frame"]]
        for s_min, m in s["levels"].items():
            for g in group_fragments(m, fragments(m), args.kernel):
                c = build_candidate(m, g, s["hsv"])
                if c is None:
                    continue
                x, y, w, h = c["bbox"]
                ok = passes_gates(c)
                cv2.rectangle(vis, (x, y), (x + w, y + h),
                              (0, 200, 0) if ok else (0, 0, 120), 2)
                if ok:
                    cv2.putText(vis, "%d frag" % c["fragments"], (x, max(10, y - 3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1, cv2.LINE_AA)
        cv2.circle(vis, (int(s["truth"][0]), int(s["truth"][1])), 5, (0, 255, 255), -1)
        cv2.putText(vis, "%s  base %.2f  grp %d/%d  multi %d/%d"
                    % (rec.get("batch"), sc0, group_ok[0], group_ok[1], multi_ok[0], multi_ok[1]),
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        name = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(rec.get("batch")))
        cv2.imwrite(os.path.join(args.out, "overlay", "%s.png" % name), vis)

    path = os.path.join(args.out, "fragment_grouping.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    n = len(out_rows)
    def pct(k):
        return 100.0 * sum(r[k] for r in out_rows) / max(1, n)
    print()
    print("  %-24s %8s %8s %8s" % ("", "recall", "select", "avg pool"))
    print("  %-24s %7.1f%% %8s %8s" % ("baseline (single comp)",
                                       pct("baseline_in_pool"), "-", "-"))
    print("  %-24s %7.1f%% %7.1f%% %8.1f" % ("A  + grouping",
                                             pct("group_single_recall"),
                                             pct("group_single_select"),
                                             sum(r["pool_single"] for r in out_rows) / n))
    print("  %-24s %7.1f%% %7.1f%% %8.1f" % ("B  + real multi-reference",
                                             pct("group_multi_recall"),
                                             pct("group_multi_select"),
                                             sum(r["pool_multi"] for r in out_rows) / n))
    print()
    print("  Per sample (pos/heading):")
    for r in out_rows:
        print("    %-8s %3s  recorded %.3f  base %.3f%s  grp %d/%d  multi %d/%d  pool %d/%d"
              % (r["pos"], r["ang"], r["recorded_shape"], r["baseline_shape"],
                 "" if r["baseline_in_pool"] else " x",
                 r["group_single_recall"], r["group_single_select"],
                 r["group_multi_recall"], r["group_multi_select"],
                 r["pool_single"], r["pool_multi"]))
    print()
    print("  wrote %s and %d overlays in %s" % (path, n, os.path.join(args.out, "overlay")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
