#!/usr/bin/env python3
"""What the HERO calibration says, per position.

Reads vision/data/hsv_marker/calibration.jsonl - samples produced by `./start.sh label` in its
"HSV 标定" mode, where a human clicks the arrow and the server records the blob under it at three
saturation levels - and reports, per batch, what that arrow actually looked like there.

The question this answers: the shape gate compares every candidate against one reference contour
taken from one frame at one position, so the same physical arrow scores 0.14 in bright light and
1.14 in the dark corner. Per-position numbers turn "some places are wrong" into a curve.

    python tools/hero_lab/hsv_calib_report.py
    python tools/hero_lab/hsv_calib_report.py --batch 左下角
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CALIB = os.path.join(ROOT, "vision", "data", "hsv_marker", "calibration.jsonl")


def load(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", default=CALIB)
    ap.add_argument("--batch", help="only this batch")
    args = ap.parse_args()

    rows = load(args.file)
    if args.batch:
        rows = [r for r in rows if r.get("batch") == args.batch]
    if not rows:
        print("no samples in %s" % args.file)
        print()
        print("collect them with:  ./start.sh label")
        print("  → 模式选「HSV 标定（点一下箭头）」")
        print("  → 备注写位置名（如 左下角 / 中间 / 亮区）")
        print("  → 在箭头上点一下，按 S 保存；换位置再来一次")
        return 1

    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r.get("batch") or "(unnamed)", []).append(r)

    print("samples %d in %d batch(es)" % (len(rows), len(groups)))
    print()
    print("  %-12s %4s %6s %6s %6s %6s %6s %8s %9s  %s"
          % ("batch", "n", "H", "S", "V", "area", "shape", "x..y", "best S", "verdict"))
    for name, rs in sorted(groups.items()):
        def med(key):
            vals = [r[key] for r in rs if isinstance(r.get(key), (int, float))]
            return statistics.median(vals) if vals else float("nan")
        best_by_level = {}
        for r in rs:
            for lvl in r.get("levels", []):
                best_by_level.setdefault(lvl["s_min"], []).append(lvl["shape"])
        best_level = None
        if best_by_level:
            best_level = min(best_by_level, key=lambda s: statistics.median(best_by_level[s]))
        xs = [r["centre"][0] for r in rs if r.get("centre")]
        ys = [r["centre"][1] for r in rs if r.get("centre")]
        span = ("%.0f..%.0f" % (min(xs), max(xs))) if xs else "-"
        shape_med = med("shape_score")
        if shape_med <= 0.30:
            verdict = "passes the 0.30 gate"
        elif shape_med <= 0.60:
            verdict = "borderline"
        else:
            verdict = "REJECTED by the gate"
        print("  %-12s %4d %6.0f %6.0f %6.0f %6d %6.2f %8s %9s  %s"
              % (name, len(rs), med("h"), med("s"), med("v"), med("area"),
                 shape_med, span, ("S>=%d" % best_level) if best_level else "-", verdict))

    print()
    print("  Per-level detail (median shape score; lower is a better fit):")
    levels = sorted({lvl["s_min"] for r in rows for lvl in r.get("levels", [])})
    if levels:
        print("  %-12s %s" % ("batch", "  ".join("S>=%-4d" % s for s in levels)))
        for name, rs in sorted(groups.items()):
            cells = []
            for s in levels:
                vals = [lvl["shape"] for r in rs for lvl in r.get("levels", [])
                        if lvl["s_min"] == s]
                cells.append("%-8s" % ("%.2f" % statistics.median(vals)) if vals else "%-8s" % "-")
            print("  %-12s %s" % (name, "  ".join(cells)))

    print()
    rejected = [n for n, rs in groups.items()
                if statistics.median([r["shape_score"] for r in rs]) > 0.30]
    if rejected:
        print("  Positions the current shape gate rejects: %s" % ", ".join(rejected))
        print("  These are the ones the reference contour does not describe - the arrow is the")
        print("  same object, the mask it produces there is simply a different shape.")
    else:
        print("  Every sampled position passes the current shape gate.")
    print()
    print("  Not yet measured: positions with no sample. The gate has never seen them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
