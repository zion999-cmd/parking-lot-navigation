"""Compare the occupancy detector against a human verdict, and print the confusion matrix.

The detector's job is a ratio per bay; the truth is a person looking at the same frame. This
grabs a frame from the camera itself, annotates every bay with the ratio the detector
computed, and writes that image out - so the human verdict can be given by reading one
picture instead of walking the table.

    python tools/bench/occupancy_bench.py --snapshot           # just capture + annotate
    python tools/bench/occupancy_bench.py --truth truth.csv    # score against a verdict

truth.csv is one line per bay that has a car in it (any order), either
    9,OCCUPIED
or just
    9
Everything not listed counts as EMPTY. Unknown is the system's own state and is reported
separately, because "I cannot see it" is a different failure from "I saw it and was wrong".
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "vision", "src"))

import occupancy                                   # noqa: E402
from spot_config import load_spots                  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "tools", "hero_lab"))
from hero_capture import grab                 # noqa: E402


MAP_PATH = os.path.join(ROOT, "vision", "out", "parking_map.json")
REFERENCE = os.path.join(ROOT, "vision", "data", "reference", "reference.jpg")
OUT_DIR = os.path.join(HERE, "out")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="http://127.0.0.1:8021/stream")
    ap.add_argument("--truth", default="", help="CSV file, or comma separated bay ids")
    ap.add_argument("--round", default="", help="label for this run, e.g. r1; appends a summary row")
    ap.add_argument("--quiet", action="store_true", help="do not print the per-bay ratios")
    ap.add_argument("--from-state", action="store_true",
                    help="score parking_state.json (what production decided) instead of "
                         "re-detecting on a freshly grabbed frame")
    ap.add_argument("--threshold", type=float, default=occupancy.OccupancyParams().changed_ratio_threshold,
                    help="default: the production value, so this measures what production does")
    args = ap.parse_args()

    map_doc = json.load(open(MAP_PATH, encoding="utf-8"))
    reference = cv2.imread(REFERENCE)
    if reference is None:
        print("no reference at %s - press r in the vision window with every bay empty" % REFERENCE)
        return 1
    # The relay answers immediately with the frame it already has, so a single grab can
    # return a stale picture - two rounds in a row measured identically (frame diff 0.0)
    # because of it. Drain a few and keep the last: the detector must see what is on the
    # table now, not what was there when the previous round ran.
    for _ in range(3):
        frame = grab(args.stream)
    if frame.shape[:2] != reference.shape[:2]:
        print("reference is %s but the camera gives %s - they must match"
              % (reference.shape[:2], frame.shape[:2]))
        return 1

    if args.from_state:
        # What production actually decided. Re-detecting here would measure this script's
        # own frame grab rather than the system, and the grab turned out to be able to return
        # a stale picture - two rounds measured identically while the strip showed red LEDs.
        doc = json.load(open(os.path.join(ROOT, "vision", "out", "parking_state.json"),
                             encoding="utf-8"))
        bays = {str(k): v for k, v in (doc.get("spots") or {}).items()}
        ratios = {k: float(v.get("changed_ratio") or 0.0) for k, v in bays.items()}
        states = {k: v.get("state") or "UNKNOWN" for k, v in bays.items()}
        frame = reference if False else None
        results = None
    else:
        params = occupancy.OccupancyParams(changed_ratio_threshold=args.threshold)
        detector = occupancy.OccupancyDetector(params)
        spots = load_spots(os.path.join(ROOT, "vision", "config", "parking_spots.json"),
                           reference.shape[:2])
        detector.set_reference(reference, spots)
        results = detector.evaluate(frame, spots)
    # evaluate() returns {spot_id: SpotResult}
    if args.from_state:
        pass
    elif isinstance(results, dict):
        ratios = {str(k): float(v.changed_ratio) for k, v in results.items()}
        states = {str(k): v.state for k, v in results.items()}
    else:
        ratios = {str(r.spot_id): float(r.changed_ratio) for r in results}
        states = {str(r.spot_id): r.state for r in results}

    if not args.truth:
        # Annotate every bay with its ratio, so a person can give the verdict by eye.
        view = frame.copy()
        for spot in map_doc.get("spots") or []:
            sid = str(spot["id"])
            poly = np.array(spot["polygon"], np.int32)
            r = ratios.get(sid, 0.0)
            over = r >= args.threshold
            cv2.polylines(view, [poly], True, (0, 0, 255) if over else (0, 200, 0), 1)
            cx, cy = int(poly[:, 0].mean()), int(poly[:, 1].mean())
            cv2.putText(view, "%s %.2f" % (sid, r), (cx - 16, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        os.makedirs(OUT_DIR, exist_ok=True)
        out = os.path.join(OUT_DIR, "occupancy_annotated.jpg")
        cv2.imwrite(out, view)
        flagged = sorted((sid, r) for sid, r in ratios.items() if r >= args.threshold)
        print("  threshold %.2f  ->  %d bays over" % (args.threshold, len(flagged)))
        if not args.quiet:
            print("  %s" % ", ".join("%s %.2f" % (s, r) for s, r in flagged))
        print("  annotated image: %s" % out)
        print("  give the verdict as truth.csv (bays that HAVE a car), then re-run with --truth")
        return 0

    truth = set()
    if os.path.exists(args.truth):
        with open(args.truth, encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if row and row[0].strip():
                    truth.add(row[0].strip())
    else:
        for part in args.truth.replace("，", ",").split(","):
            if part.strip():
                truth.add(part.strip())
    bays = [str(s["id"]) for s in map_doc.get("spots") or []]
    m = {"TP": 0, "FN": 0, "FP": 0, "TN": 0, "UNKNOWN": 0}
    misses, false_pos = [], []
    for sid in bays:
        verdict = sid in truth
        state = states.get(sid, "UNKNOWN")
        if state == "UNKNOWN":
            m["UNKNOWN"] += 1
            continue
        said = state == "OCCUPIED"
        if verdict and said:
            m["TP"] += 1
        elif verdict and not said:
            m["FN"] += 1
            misses.append(sid)
        elif not verdict and said:
            m["FP"] += 1
            false_pos.append(sid)
        else:
            m["TN"] += 1
    decided = m["TP"] + m["FN"] + m["FP"] + m["TN"]
    print("  confusion matrix (48 bays, threshold %.2f)" % args.threshold)
    print("                   system EMPTY   system OCCUPIED   system UNKNOWN")
    print("  truth EMPTY    %12d   %15d   %14d" % (m["TN"], m["FP"], m["UNKNOWN"]))
    print("  truth OCCUP.   %12d   %15d   %14d" % (m["FN"], m["TP"], 0))
    print("  accuracy on decided bays  %.3f  (%d of 48)" % ((m["TP"] + m["TN"]) / max(decided, 1), decided))
    print("  false occupied  %d %s" % (m["FP"], false_pos))
    print("  false empty     %d %s" % (m["FN"], misses))
    print("  unknown rate    %.3f" % (m["UNKNOWN"] / 48.0))

    summary = os.path.join(OUT_DIR, "occupancy_rounds.csv")
    new = not os.path.exists(summary)
    with open(summary, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["round", "threshold", "truth_occupied", "tp", "fn", "fp", "tn",
                        "unknown", "accuracy", "false_occupied", "false_empty", "missed"])
        w.writerow([args.round or "-", args.threshold, len(truth), m["TP"], m["FN"], m["FP"],
                    m["TN"], m["UNKNOWN"],
                    round((m["TP"] + m["TN"]) / max(decided, 1), 4), m["FP"], m["FN"],
                    " ".join(misses)])
    print("  appended to %s" % summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
