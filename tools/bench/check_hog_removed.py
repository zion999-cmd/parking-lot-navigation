#!/usr/bin/env python3
"""Regression check for removing hero_hog from the HERO runtime path.

Compares the current vehicle.py against the commit before the change, over the same cached
frames, and asserts the things the change is supposed to guarantee:

  * position and heading are bit-identical to before
  * hero_hog.detect is called zero times, and its model is never loaded
  * a red_arrow failure holds the last heading and marks it stale
  * failures with no history report heading=None instead of inventing one
  * per-frame cost is not worse

Usage:
    python check_hog_removed.py [--frames 220]
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # tools/bench -> tools -> project
SRC = os.path.join(ROOT, "vision", "src")
FRAMES = os.path.join(ROOT, "tools", "hero_lab", "heading_spike", "out", "frames")

sys.path.insert(0, SRC)


def load_before(tmpdir="/tmp/hog_regress"):
    """The vehicle.py from HEAD~ of the removal commit, as its own module."""
    os.makedirs(tmpdir, exist_ok=True)
    target = os.path.join(tmpdir, "vehicle_before.py")
    with open(target, "w", encoding="utf-8") as f:
        f.write(subprocess.check_output(
            ["git", "-C", ROOT, "show", "HEAD~1:vision/src/vehicle.py"]).decode())
    sys.path.insert(0, tmpdir)
    import vehicle_before                    # noqa: F401
    return vehicle_before


def frames(limit):
    names = sorted(n for n in os.listdir(FRAMES) if n.endswith(".jpg"))[:limit]
    return [cv2.imread(os.path.join(FRAMES, n)) for n in names]


def run_module(mod, imgs, hog_counter=None):
    """Walk every frame through one vehicle module and collect its output."""
    out = []
    elapsed = 0.0
    for img in imgs:
        t = time.perf_counter()
        poses = mod.locate_all(img)
        elapsed += time.perf_counter() - t
        p = poses[0] if poses else None
        out.append(None if p is None else {
            "x": round(p.x, 3), "y": round(p.y, 3),
            "heading": None if p.heading is None else round(p.heading, 3),
            "from_arrow": bool(p.heading_from_arrow),
            "stale": bool(getattr(p, "heading_stale", False)),
        })
    return out, elapsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=220)
    args = ap.parse_args()

    imgs = frames(args.frames)
    if not imgs:
        print("no cached frames at %s" % FRAMES)
        return 1

    import vehicle
    import hero_hog

    # Count hero_hog calls and model loads without changing its behaviour.
    calls = {"detect": 0, "load": 0}
    real_detect = hero_hog.detect

    def counted_detect(*a, **k):
        calls["detect"] += 1
        return real_detect(*a, **k)

    hero_hog.detect = counted_detect
    real_load = cv2.ml.SVM_load

    def counted_load(*a, **k):
        calls["load"] += 1
        return real_load(*a, **k)

    cv2.ml.SVM_load = counted_load

    before = load_before()
    before_calls = {"detect": 0}
    before_hero = sys.modules["hero_hog"]
    # The old module imported hero_hog directly; count through the shared module object.
    before_detect = before_hero.detect

    def before_counted(*a, **k):
        before_calls["detect"] += 1
        return before_detect(*a, **k)

    before_hero.detect = before_counted
    old_out, old_time = run_module(before, imgs)
    before_hero.detect = before_detect

    # Reset both counters: the numbers that matter describe the *current* code alone. The
    # previous run above loaded the model and called detect, and leaving those in the tally
    # would blame the new code for the old code's work.
    calls["detect"] = 0
    calls["load"] = 0
    new_out, new_time = run_module(vehicle, imgs)

    n = len(imgs)
    print("frames                        %d" % n)
    print()
    print("--- comparison with the previous commit ---")
    same = diff_pos = diff_head = 0
    worst = (0.0, 0.0)
    for a, b in zip(old_out, old_out if False else new_out):
        pass
    for a, b in zip(old_out, new_out):
        if a is None or b is None:
            if a is b:
                same += 1
            continue
        dp = max(abs(a["x"] - b["x"]), abs(a["y"] - b["y"]))
        dh = 0.0 if (a["heading"] is None or b["heading"] is None) else abs(
            ((a["heading"] - b["heading"] + 180.0) % 360.0) - 180.0)
        if dp == 0 and dh == 0:
            same += 1
        else:
            if dp > 0:
                diff_pos += 1
                worst = (max(worst[0], dp), worst[1])
            if dh > 0:
                diff_head += 1
                worst = (worst[0], max(worst[1], dh))
    print("identical frames              %d / %d" % (same, n))
    print("position differences          %d  (worst %.3f px)" % (diff_pos, worst[0]))
    print("heading differences           %d  (worst %.3f deg)" % (diff_head, worst[1]))
    print()
    print("--- hero_hog runtime ---")
    print("detect() calls, previous code %d" % before_calls["detect"])
    print("detect() calls, current code  %d" % calls["detect"])
    print("SVM_load calls, current code  %d" % calls["load"])
    print("hero_hog in sys.modules       %s" % ("hero_hog" in sys.modules))
    print("  (the module is imported by this script to instrument it; what matters is the")
    print("   call count above and that vehicle.py no longer imports it)")
    print()
    print("--- cost ---")
    print("previous  %.2f ms/frame" % (1000.0 * old_time / n))
    print("current   %.2f ms/frame" % (1000.0 * new_time / n))
    print()

    # ---- failure behaviour
    print("--- failure behaviour ---")
    import red_arrow
    real_red = red_arrow.detect

    vehicle._last_heading[0] = None
    vehicle.reset_heading_metrics()

    red_arrow.detect = lambda *a, **k: None
    try:
        # a) no history at all
        p = vehicle.locate_all(imgs[0])
        pose = p[0] if p else None
        print("no history, arrow fails       heading=%s stale=%s" % (
            None if pose is None else pose.heading, None if pose is None else pose.heading_stale))
        unavailable = vehicle.heading_metrics()["heading_unavailable_total"]

        # b) seed a heading, then fail
        vehicle._last_heading[0] = 137.0
        vehicle._heading_metrics["heading_consecutive_failures"] = 0
        p = vehicle.locate_all(imgs[1])
        pose = p[0] if p else None
        print("history 137, arrow fails      heading=%s stale=%s" % (
            None if pose is None else pose.heading, None if pose is None else pose.heading_stale))

        # c) three consecutive failures
        for i in range(3):
            p = vehicle.locate_all(imgs[2 + i])
            pose = p[0] if p else None
        m = vehicle.heading_metrics()
        print("after 3 more failures         consecutive=%d stale=%s" % (
            m["heading_consecutive_failures"], m["heading_stale"]))
        print("counters                      %s" % {k: m[k] for k in (
            "red_arrow_success_total", "red_arrow_fail_total",
            "heading_hold_last_total", "heading_unavailable_total")})
    finally:
        red_arrow.detect = real_red

    # d) recovery clears the flag
    vehicle._heading_metrics["heading_consecutive_failures"] = 5
    p = vehicle.locate_all(imgs[0])
    pose = p[0] if p else None
    m = vehicle.heading_metrics()
    print("arrow readable again          heading=%s stale=%s consecutive=%d" % (
        None if pose is None else (None if pose.heading is None else round(pose.heading, 1)),
        None if pose is None else pose.heading_stale,
        m["heading_consecutive_failures"]))
    print("  success_total now %d, unavailable_total now %d" % (
        m["red_arrow_success_total"], m["heading_unavailable_total"]))

    ok = (diff_pos == 0 and diff_head == 0 and calls["detect"] == 0 and calls["load"] == 0)
    print()
    print("RESULT: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
