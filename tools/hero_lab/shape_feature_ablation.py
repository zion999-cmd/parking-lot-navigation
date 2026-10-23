#!/usr/bin/env python3
"""Is shape a gate or a feature? Three policies, same 32 calibration frames.

Measured previously: with shape used as a hard pool gate, 90.6% of the sampled arrows enter the
pool and 84.4% end up selected. That is not good enough, and the reason is structural - shape is
deciding *whether a candidate exists* when what it can honestly report is *how arrow-like it
looks*. This compares three policies:

  1 prod         shape gate at 0.60 against the single stored reference   (today's pipeline)
  2 bank_gate    shape gate at 0.60 against a 40-entry bank of real contours
  3 bank_feature no shape gate in TRACKING; shape is one feature among six, scored by HeroBelief

Stability is measured on eight short sessions built from the same frames: for each position the
four headings in order, which is a car rotating in place - the position must not move while the
direction does. That is a surrogate for a video, not a video; it has no motion blur and only four
frames per session.

    python shape_feature_ablation.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "vision", "src"))
sys.path.insert(0, HERE)

import hero_belief                          # noqa: E402
import hsv_marker                           # noqa: E402
import fragment_grouping as FG              # noqa: E402

S_LEVELS = (60, 50, 45)
TOL = 14.0                  # px, "this candidate is the arrow" (overridden by --tol)
JUMP_PX = 40.0


def label_of(batch: str) -> tuple[str, int]:
    m = re.match(r"^(\D*?)[\s_-]*(\d{1,3})\s*°?$", batch.strip())
    return (m.group(1), int(m.group(2))) if m else (batch.strip(), -1)


def load_samples(calib: str) -> list[dict]:
    samples = []
    with open(calib, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            path = os.path.join(ROOT, "vision", "data", "hsv_marker", "frames", r.get("frame") or "")
            frame = cv2.imread(path)
            if frame is None:
                continue
            pos, ang = label_of(r.get("batch") or "")
            samples.append({"rec": r, "pos": pos, "ang": ang, "frame": frame,
                            "hsv": cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
                            "truth": (float(r["centre"][0]), float(r["centre"][1])),
                            "masks": {s: hsv_marker._components(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
                                                                s, hsv_marker.V_MIN)[0]
                                      for s in S_LEVELS}})
    return samples


def build_bank(samples: list[dict], ref0) -> dict[str, list]:
    """Real sampled contours, keyed by the frame they came from, plus rotated fallbacks."""
    bank = {}
    for s in samples:
        mask = s["masks"].get(s["rec"].get("s_min_used") or 60)
        x, y, w, h = s["rec"]["bbox"]
        sub = (mask[y:y + h, x:x + w] > 0).astype(np.uint8)
        cs, _ = cv2.findContours(sub * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cs:
            bank[s["rec"]["frame"]] = max(cs, key=cv2.contourArea)
    rotated = FG.rotated(ref0)[:8]
    return {"by_frame": bank, "rotated": rotated}


def pool_for(sample: dict, bank: dict, policy: str, ref0) -> tuple[list[dict], float]:
    """Candidates for one frame. `policy` decides whether shape gates or merely scores."""
    t0 = time.perf_counter()
    refs = [ref0] if policy == "prod" else \
           [c for k, c in bank["by_frame"].items() if k != sample["rec"]["frame"]] + bank["rotated"]
    out = []
    for s_min, mask in sample["masks"].items():
        for f in FG.fragments(mask):
            for g in FG.group_fragments(mask, [f], 0):     # no grouping: measured to hurt
                c = FG.build_candidate(mask, g, sample["hsv"])
                if c is None or not FG.passes_gates(c):
                    continue
                sc, which = FG.shape_score(c["contour"], refs)
                c["shape_score"] = sc
                c["ref_index"] = which
                c["s_min"] = s_min
                # policy decides what shape is allowed to do
                if policy in ("prod", "bank_gate") and sc > FG.WEAK:
                    continue
                out.append(c)
    out.sort(key=lambda c: c["shape_score"])
    merged = []
    for c in out:
        if any(abs(c["centre"][0] - m["centre"][0]) < 10
               and abs(c["centre"][1] - m["centre"][1]) < 10 for m in merged):
            continue
        merged.append(c)
    return merged, (time.perf_counter() - t0) * 1000.0


def near(a, b, tol=None) -> bool:
    # Default arguments are bound at definition time, so a module-level TOL set later from the
    # command line would never be seen. Read it at call time instead.
    t = TOL if tol is None else tol
    return abs(a[0] - b[0]) <= t and abs(a[1] - b[1]) <= t


def main() -> int:
    global TOL              # --tol overrides the module default; must precede any use
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calib", default=os.path.join(ROOT, "vision", "data", "hsv_marker",
                                                    "calibration.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "out", "shape_feature_ablation"))
    ap.add_argument("--tol", type=float, default=TOL,
                    help="px tolerance against the recorded centre; the recorded blob is\n                          sometimes a partially masked or merged version of the arrow")
    args = ap.parse_args()
    TOL = args.tol
    os.makedirs(args.out, exist_ok=True)

    samples = load_samples(args.calib)
    if not samples:
        print("no calibration frames")
        return 1
    ref0 = hsv_marker._load_reference()
    bank = build_bank(samples, ref0)
    print("samples %d, bank %d real + %d rotated"
          % (len(samples), len(bank["by_frame"]), len(bank["rotated"])))

    results = {}
    for policy in ("prod", "bank_gate", "bank_feature"):
        # ---- per-frame: candidate recall and selection accuracy
        recall = 0
        select = 0
        pools = 0
        ms = []
        detail = {}
        for s in samples:
            pool, cost = pool_for(s, bank, policy, ref0)
            ms.append(cost)
            pools += len(pool)
            has = any(near(c["centre"], s["truth"]) for c in pool)
            recall += int(has)
            if policy == "bank_feature":
                # The belief decides, not the shape order. Seed it with a plausible history so
                # this measures "the car is where we think it is and this frame must not lose it".
                b = hero_belief.HeroBelief()
                b.update([{"bbox": (int(s["truth"][0]) - 16, int(s["truth"][1]) - 10, 32, 20),
                           "centre": s["truth"], "area": s["rec"]["area"],
                           "width": s["rec"]["width"], "height": s["rec"]["height"],
                           "aspect": s["rec"]["aspect"], "shape_score": 0.2,
                           "s": s["rec"]["s"], "v": s["rec"]["v"]}], 0.1)
                v = b.update(pool, 0.1)
                picked = v["position"]
            else:
                picked = pool[0]["centre"] if pool else None
            ok = picked is not None and near(picked, s["truth"])
            select += int(ok)
            detail[s["rec"]["batch"]] = {"recall": int(has), "select": int(ok), "pool": len(pool),
                                         "shape_min": round(min((c["shape_score"] for c in pool),
                                                                default=9.9), 3)}

        # ---- stability: eight sessions, four headings each, position must not move
        sessions = {}
        for s in samples:
            sessions.setdefault(s["pos"], []).append(s)
        lost = jumped = frames = 0
        for pos, ss in sessions.items():
            ss.sort(key=lambda x: x["ang"])
            b = hero_belief.HeroBelief(use_motion_gate=True, use_commitment=True)
            prev = None
            for s in ss:
                frames += 1
                full, _ = pool_for(s, bank, policy, ref0)
                if b.position is None or b.should_search_globally():
                    pool = full
                else:
                    r = b.search_radius()
                    pool = [c for c in full
                            if abs(c["centre"][0] - b.position[0]) <= r
                            and abs(c["centre"][1] - b.position[1]) <= r]
                v = b.update(pool, 0.1)
                if v["state"] == hero_belief.LOST:
                    lost += 1
                p = v["position"]
                if p is not None and prev is not None:
                    if ((p[0] - prev[0]) ** 2 + (p[1] - prev[1]) ** 2) ** 0.5 > JUMP_PX:
                        jumped += 1
                prev = p if p is not None else prev

        results[policy] = {
            "recall": recall, "select": select, "n": len(samples),
            "pool": pools / len(samples), "ms": statistics.median(ms),
            "lost": lost, "jumped": jumped, "frames": frames, "detail": detail,
        }

    print()
    print("  %-14s %9s %9s %10s %8s %7s %7s" % ("policy", "recall", "select",
                                                "cand/frame", "ms/frame", "lost", "jumps"))
    for policy, r in results.items():
        print("  %-14s %5d/%-3d %5d/%-3d %10.1f %8.1f %7d %7d"
              % (policy, r["recall"], r["n"], r["select"], r["n"], r["pool"], r["ms"],
                 r["lost"], r["jumped"]))

    print()
    print("  The two samples the task asks about specifically:")
    for name in ("左上90", "左下90"):
        print("    %-8s" % name, end="")
        for policy, r in results.items():
            d = r["detail"].get(name)
            if d:
                print("  %s: recall %d select %d shape %.2f pool %d"
                      % (policy, d["recall"], d["select"], d["shape_min"], d["pool"]), end="")
        print()

    path = os.path.join(args.out, "ablation.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "detail"} | {"detail": v["detail"]}
                   for k, v in results.items()}, fh, ensure_ascii=False, indent=1)
    print()
    print("  wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
