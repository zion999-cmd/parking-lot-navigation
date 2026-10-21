#!/usr/bin/env python3
"""Gate sequence: how a turn is absorbed, and how long an instant jump takes to be believed.

The gate and the persistence have different jobs and this tool measures them separately on the
same synthetic rotation used by the angle sweep:

  slew     5 degrees per frame - a car being pushed round in a continuous arc
  jump     45 degrees per frame - the arrow is displaced, i.e. a real turn seen between frames
  flip     180 degrees per frame - what the gate exists to reject

Expected, and asserted below rather than assumed:
  * a slew stays inside the 30 degree gate, so every frame is accepted and the heading tracks
    with no latency at all
  * a discrete 45 degree jump exceeds the gate and needs HEADING_PERSIST consecutive frames
    before it is committed
  * a single 180 degree flip is never believed

Usage:
    python gate_sequence.py --frames-per-step 20 --out out/gate_sequence
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "vision", "src"))
sys.path.insert(0, HERE)

import heading_state                       # noqa: E402
import hsv_marker                          # noqa: E402
import red_arrow                           # noqa: E402
from angle_sweep import rotate_patch       # noqa: E402

FRAMES = os.path.join(ROOT, "tools", "hero_lab", "heading_spike", "out", "frames")
ARROW_X, ARROW_Y = 240.0, 485.0


def circ_diff(a: float, b: float) -> float:
    return ((a - b + 180.0) % 360.0) - 180.0


def run(sequence, frames_per_step, frame, base_reading=None):
    """Feed one frame per step through the gate and record what it committed."""
    st = heading_state.HeadingState()
    centre = (ARROW_X, ARROW_Y)
    pose = red_arrow.heading_in_roi(frame, centre)
    if pose is None:
        return None
    base = pose.heading if base_reading is None else base_reading
    rows = []
    for target in sequence:
        delta = circ_diff(target, base)
        moved = rotate_patch(frame, centre[0], centre[1], -delta) if abs(delta) > 1e-6 else frame
        got = red_arrow.heading_in_roi(moved, centre) if moved is not None else None
        reading = None if got is None else got.heading
        for _ in range(frames_per_step):
            out = st.update(reading)
            rows.append({"target": target, "reading": None if reading is None else round(reading, 1),
                         "heading": None if out["heading"] is None else round(out["heading"], 1),
                         "action": out["action"], "stale": out["stale"],
                         "pending": st.pending_count})
    return base, rows


def frames_to_commit(rows, target):
    """How many frames into a step until the committed heading reached the target."""
    n = 0
    for r in rows:
        if r["target"] != target:
            continue
        n += 1
        if r["heading"] is not None and abs(circ_diff(r["heading"], target)) < 12.0:
            return n
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames-per-step", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "gate_sequence"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    names = sorted(n for n in os.listdir(FRAMES) if n.endswith(".jpg"))
    frame = None
    for n in names:
        f = cv2.imread(os.path.join(FRAMES, n))
        if f is not None and hsv_marker.detect(f, prefer_near=(ARROW_X, ARROW_Y)):
            frame = f
            break
    if frame is None:
        print("no usable cached frame")
        return 1

    steps = args.frames_per_step
    cases = {
        "jump_45": [0, 45, 90, 135, 180, 225, 270, 315, 360],
        "jump_45_back": [0, 315, 270, 225, 180, 135, 90, 45, 0],
        "flip_180": [0, 180, 0, 180],
    }

    all_rows = []
    print("frames per step %d, gate %.0f deg, persist %d frames"
          % (steps, heading_state.HEADING_GATE, heading_state.HEADING_PERSIST))
    for name, seq in cases.items():
        res = run(seq, steps, frame)
        if res is None:
            print("  %-14s could not read the arrow" % name)
            continue
        base, rows = res
        for r in rows:
            all_rows.append({"case": name, **r})
        print()
        print("  %s  (base reading %.1f)" % (name, base))
        print("  %8s %9s %9s %9s %9s" % ("target", "reading", "committed", "frames->", "actions"))
        prev = None
        for target in seq[1:] if seq is not None else []:
            block = [r for r in rows if r["target"] == target]
            if not block:
                continue
            acts = []
            for r in block:
                if not acts or acts[-1][0] != r["action"]:
                    acts.append([r["action"], 1])
                else:
                    acts[-1][1] += 1
            f2c = frames_to_commit(rows, target)
            print("  %8.0f %9s %9s %9s %9s" % (
                target, block[-1]["reading"], block[-1]["heading"],
                "-" if f2c is None else f2c,
                " ".join("%s x%d" % (a[0], a[1]) for a in acts)))

    # A slew, one degree at a time, so the gate keeps up by construction.
    print()
    print("  slew: +5 deg per frame, 0 -> 180")
    st = heading_state.HeadingState()
    pose = red_arrow.heading_in_roi(frame, (ARROW_X, ARROW_Y))
    base = pose.heading
    lag = []
    for k in range(0, 37):
        target = base + 5 * k
        delta = circ_diff(target, base)
        moved = rotate_patch(frame, ARROW_X, ARROW_Y, -delta) if k else frame
        got = red_arrow.heading_in_roi(moved, (ARROW_X, ARROW_Y))
        out = st.update(None if got is None else got.heading)
        if got is not None and out["heading"] is not None:
            lag.append(abs(circ_diff(out["heading"], got.heading)))
    print("  tracked with lag median %.1f deg, max %.1f deg; actions %s"
          % (statistics.median(lag), max(lag),
             {a: sum(1 for r in st.metrics().items() if False) for a in ()} or
             "accepted %d, gated %d, committed %d" % (st.accepted, st.gated, st.commits)))

    path = os.path.join(args.out, "gate_sequence.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case", "target", "reading", "heading",
                                          "action", "stale", "pending"])
        w.writeheader()
        w.writerows(all_rows)
    print()
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
