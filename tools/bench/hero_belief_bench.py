#!/usr/bin/env python3
"""Ablation for the HERO belief layer: which part actually stops the jumping.

Runs the same frames through five configurations and reports, for each, how often it reported the
car where the car actually was, how far the reported position moved between frames, and what it
cost. Truth is a single manually-checked point - the arrow does not move in these clips - so
"error" is distance from that point, not a ground-truth track.

    python hero_belief_bench.py --frames tools/.../*.jpg --truth 66,473
    python hero_belief_bench.py --capture 60 --truth 66,473
"""

from __future__ import annotations

import argparse
import glob
import os
import socket
import statistics
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "vision", "src"))

import hero_belief      # noqa: E402
import hsv_marker       # noqa: E402
import red_arrow        # noqa: E402


def grab_stream(n: int, url="http://127.0.0.1:8021/stream"):
    from urllib.parse import urlparse
    u = urlparse(url)
    s = socket.create_connection((u.hostname, u.port or 80), timeout=30)
    s.sendall(("GET %s HTTP/1.0\r\nHost: %s\r\n\r\n" % (u.path, u.hostname)).encode())
    s.settimeout(30)
    buf, out = b"", []
    while len(out) < n:
        buf += s.recv(65536)
        a = buf.find(b"\xff\xd8")
        b = buf.find(b"\xff\xd9", a)
        if a >= 0 and b > a:
            out.append(cv2.imdecode(np.frombuffer(buf[a:b + 2], np.uint8), cv2.IMREAD_COLOR))
            buf = buf[b + 2:]
    s.close()
    return out


def baseline(frame):
    """Pre-belief behaviour: single threshold, shape gate, then the whole-frame fallback."""
    h = hsv_marker.detect(frame)
    if h is not None:
        return h["centre"]
    r = red_arrow.detect(frame)
    return None if r is None else r.centre


def variant(name, frame, belief, dt):
    if belief is None:
        return baseline(frame)
    if belief.should_search_globally():
        pool = hsv_marker.candidates(frame)
    else:
        pool = hsv_marker.candidates(frame, centre=belief.position,
                                     radius=belief.search_radius())
    v = belief.update(pool, dt)
    return v["position"] if v["state"] != hero_belief.LOST else None


def run(frames, truth, name, belief):
    xs, errors, jumps, lost, ms = [], [], [], 0, []
    prev = None
    for f in frames:
        t = time.perf_counter()
        p = variant(name, f, belief, 0.1)
        ms.append((time.perf_counter() - t) * 1000)
        if p is None:
            lost += 1
            continue
        if prev is not None:
            d = ((p[0] - prev[0]) ** 2 + (p[1] - prev[1]) ** 2) ** 0.5
            jumps.append(d)
        prev = p
        xs.append(p[0])
        errors.append(((p[0] - truth[0]) ** 2 + (p[1] - truth[1]) ** 2) ** 0.5)
    on_target = sum(1 for e in errors if e < 20.0)
    big = [j for j in jumps if j > 40.0]
    return {
        "name": name, "frames": len(frames), "on_target": on_target,
        "lost": lost, "jumps": len(big),
        "max_jump": max(jumps) if jumps else 0.0,
        "median_err": statistics.median(errors) if errors else float("nan"),
        "ms": statistics.median(ms),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="*", help="jpg paths")
    ap.add_argument("--capture", type=int, help="grab this many frames from the live relay")
    ap.add_argument("--truth", help="x,y of the arrow, manual")
    ap.add_argument("--auto-truth", action="store_true",
                    help="use the full model's median position as the reference and print it")
    args = ap.parse_args()
    if args.truth:
        tx, ty = (float(v) for v in args.truth.split(","))
    else:
        tx = ty = float("nan")

    if args.capture:
        frames = grab_stream(args.capture)
    else:
        paths = []
        for pat in args.frames or []:
            paths.extend(sorted(glob.glob(pat)))
        frames = [cv2.imread(p) for p in paths]
        frames = [f for f in frames if f is not None]

    variants = [
        ("baseline", None),
        ("+local/multi", hero_belief.HeroBelief(use_motion_gate=False,
                                                use_appearance=False, use_commitment=False)),
        ("+motion", hero_belief.HeroBelief(use_appearance=False, use_commitment=False)),
        ("+appearance", hero_belief.HeroBelief(use_commitment=False)),
        ("+commitment (full)", hero_belief.HeroBelief()),
    ]
    if args.auto_truth:
        probe = hero_belief.HeroBelief()
        ps = []
        for f in frames:
            pool = (hsv_marker.candidates(f) if probe.should_search_globally()
                    else hsv_marker.candidates(f, centre=probe.position,
                                               radius=probe.search_radius()))
            v = probe.update(pool, 0.1)
            if v["position"] is not None:
                ps.append(v["position"])
        if not ps:
            print("no positions; cannot infer truth")
            return 1
        tx = statistics.median(p[0] for p in ps)
        ty = statistics.median(p[1] for p in ps)
        print("auto truth from the full model's median: (%.0f, %.0f)" % (tx, ty))
        print("  (the arrow does not move in this clip; if this is not where the car is, the")
        print("   model locked onto the wrong blob and every row below is measured against that)")
    print("frames %d, truth (%.0f, %.0f)" % (len(frames), tx, ty))
    print()
    print("  %-20s %8s %9s %5s %6s %9s %9s %7s"
          % ("variant", "on-target", "lost", "jumps", "maxjump", "med err", "med ms", ""))
    for name, belief in variants:
        r = run(frames, (tx, ty), name, belief)
        print("  %-20s %6d/%-3d %9d %5d %6.0f %9.1f %9.2f"
              % (name, r["on_target"], r["frames"], r["lost"], r["jumps"],
                 r["max_jump"], r["median_err"], r["ms"]))
    print()
    print("  on-target = frames reporting within 20 px of the manual truth")
    print("  jumps     = frame-to-frame movement over 40 px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
