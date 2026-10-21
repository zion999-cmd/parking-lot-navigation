"""Measure the hero detector over a fixed window and keep the per-frame record.

    python tools/bench/hero_bench.py --seconds 60

Writes tools/bench/out/hero_track.csv (one row per frame: time, found, x, y, heading,
margin, processing ms) and prints the summary a paper table needs.

What this can and cannot say: the car is standing still, so what it measures is the
detection rate and the stability of the position and heading, plus the cost per frame. It
does not measure position accuracy in centimetres - that needs a ground-truth pose, which is
a separate measurement.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "vision", "src"))
sys.path.insert(0, os.path.join(ROOT, "tools", "hero_lab"))

import hero_hog                                   # noqa: E402
from hero_capture import grab                     # noqa: E402

OUT = os.path.join(HERE, "out")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="http://192.168.3.118/stream")
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args()

    hero_hog.reset()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    started = time.time()
    while time.time() - started < args.seconds:
        frame = grab(args.stream)
        t0 = time.perf_counter()
        det = hero_hog.detect(frame)
        ms = (time.perf_counter() - t0) * 1000.0
        rows.append({"t": round(time.time() - started, 3), "found": 1 if det else 0,
                     "x": round(det.centre[0], 2) if det else "",
                     "y": round(det.centre[1], 2) if det else "",
                     "heading": round(det.heading, 1) if det else "",
                     "margin": round(det.margin, 3) if det else "",
                     "ms": round(ms, 3)})

    path = os.path.join(OUT, "hero_track.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    found = [r for r in rows if r["found"]]
    ms_all = sorted(r["ms"] for r in rows)
    print("  frames %d over %.1f s" % (len(rows), rows[-1]["t"]))
    print("  detection rate    %.3f  (%d/%d)" % (len(found) / len(rows), len(found), len(rows)))
    if found:
        xs = [r["x"] for r in found]
        ys = [r["y"] for r in found]
        hs = [r["heading"] for r in found]
        spread = lambda a: max(a) - min(a)
        sd = lambda a: statistics.pstdev(a) if len(a) > 1 else 0.0
        # Circular spread for the heading: a plain min/max is meaningless across 0/360.
        circ = [((h - hs[0] + 180.0) % 360.0) - 180.0 for h in hs]
        print("  position x        mean %.1f  spread %.1f px  sd %.2f px"
              % (statistics.mean(xs), spread(xs), sd(xs)))
        print("  position y        mean %.1f  spread %.1f px  sd %.2f px"
              % (statistics.mean(ys), spread(ys), sd(ys)))
        print("  heading           mean %.1f deg  spread %.1f deg  sd %.2f deg (circular)"
              % (statistics.mean(hs), max(circ) - min(circ), statistics.pstdev(circ)))
    print("  processing ms     mean %.1f  p50 %.1f  p95 %.1f  max %.1f"
          % (statistics.mean(ms_all), statistics.median(ms_all),
             ms_all[int(len(ms_all) * 0.95)], ms_all[-1]))
    print("  csv               %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
