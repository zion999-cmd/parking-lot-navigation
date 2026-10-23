#!/usr/bin/env python3
"""Record a real push-around, then replay it through old and new implementations.

Static frames cannot test a motion gate. This records the car actually being pushed - frames and
a per-frame timeline - so the same recording can be replayed through two implementations and
compared, which is the only fair way to answer "is the new pipeline better" without two cars.

    python tools/bench/hero_dynamic.py record --seconds 60 --label A_left_to_right
    python tools/bench/hero_dynamic.py summarise tools/bench/out/dyn/A_left_to_right.csv
    python tools/bench/hero_dynamic.py replay   tools/bench/out/dyn/A_left_to_right --impl head
    python tools/bench/hero_dynamic.py compare  <old.csv> <new.csv>

Timeline columns are chosen so a wrong jump can be explained afterwards: every frame records the
five evidence scores of the candidate that won, plus how many candidates there were.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from urllib.parse import urlparse

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "vision", "src")
RELAY = "http://127.0.0.1:8021/stream"
OUT = os.path.join(HERE, "out", "dyn")

COLUMNS = ["frame", "t", "x", "y", "heading", "confidence", "state", "stale",
           "position_stale", "candidate_count", "best_shape", "position_score",
           "motion_score", "size_score", "colour_score", "shape_score", "ms"]


def stream(url=RELAY, timeout=30):
    u = urlparse(url)
    s = socket.create_connection((u.hostname, u.port or 80), timeout=timeout)
    s.sendall(("GET %s HTTP/1.0\r\nHost: %s\r\n\r\n" % (u.path, u.hostname)).encode())
    s.settimeout(timeout)
    buf = b""
    while True:
        buf += s.recv(65536)
        a = buf.find(b"\xff\xd8")
        b = buf.find(b"\xff\xd9", a)
        if a >= 0 and b > a:
            yield cv2.imdecode(np.frombuffer(buf[a:b + 2], np.uint8), cv2.IMREAD_COLOR)
            buf = buf[b + 2:]


def load_impl(which: str):
    """'current' or a git ref, which is checked out to /tmp as its own module."""
    if which == "current":
        sys.path.insert(0, SRC)
        import vehicle
        return vehicle
    tmp = "/tmp/dyn_impl"
    os.makedirs(tmp, exist_ok=True)
    path = os.path.join(tmp, "vehicle_%s.py" % which.replace("/", "_"))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(subprocess.check_output(
            ["git", "-C", ROOT, "show", "%s:vision/src/vehicle.py" % which]).decode())
    sys.path.insert(0, tmp)
    sys.path.insert(0, SRC)
    mod = __import__("vehicle_%s" % which.replace("/", "_"))
    return mod


def row_for(mod, frame, idx, t, ms):
    pose = mod.locate_all(frame)
    p = pose[0] if pose else None
    belief = getattr(mod, "_belief", None)
    m = belief.metrics() if belief is not None else {}
    sc = getattr(belief, "last_score", None) if belief is not None else None
    return {
        "frame": idx, "t": round(t, 3),
        "x": None if p is None else round(p.x, 1),
        "y": None if p is None else round(p.y, 1),
        "heading": None if (p is None or p.heading is None) else round(p.heading, 1),
        "confidence": m.get("hero_confidence"),
        "state": m.get("hero_state"),
        "stale": int(bool(getattr(p, "position_stale", False))),
        "position_stale": int(bool(getattr(p, "position_stale", False))),
        "candidate_count": belief.last_candidates if belief is not None else None,
        "best_shape": None if not sc else round(sc["candidate"]["shape_score"], 3),
        "position_score": None if not sc else round(sc["position_score"], 3),
        "motion_score": None if not sc else round(sc["motion_score"], 3),
        "size_score": None if not sc else round(sc["size_score"], 3),
        "colour_score": None if not sc else round(sc["colour_score"], 3),
        "shape_score": None if not sc else round(sc["shape_score"], 3),
        "ms": round(ms, 2),
    }


def summarise(path: str) -> dict:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        return {}
    def f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    xs = [(f(r["x"]), f(r["y"])) for r in rows]
    valid = [p for p in xs if p[0] is not None]
    steps = [((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
             for a, b in zip(valid[1:], valid[:-1])]
    jumps = [s for s in steps if s > 40]
    ms = [f(r["ms"]) for r in rows if f(r["ms"]) is not None]
    states = {}
    for r in rows:
        states[r.get("state") or "?"] = states.get(r.get("state") or "?", 0) + 1
    out = {
        "frames": len(rows),
        "position_valid": len(valid),
        "tracked_pct": 100.0 * len(valid) / len(rows),
        "lost_frames": states.get("LOST", 0),
        "uncertain_frames": states.get("UNCERTAIN", 0),
        "stale_frames": sum(int(f(r["stale"]) or 0) for r in rows),
        "false_jumps": len(jumps),
        "max_jump_px": round(max(jumps), 1) if jumps else 0.0,
        "position_step_median_px": round(statistics.median(steps), 1) if steps else 0.0,
        "ms_median": round(statistics.median(ms), 2) if ms else 0.0,
        "ms_p95": round(sorted(ms)[int(len(ms) * 0.95)], 2) if ms else 0.0,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["record", "summarise", "replay", "compare"])
    ap.add_argument("target", nargs="?")
    ap.add_argument("second", nargs="?")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--label", default="run")
    ap.add_argument("--impl", default="current", help="current, or a git ref like HEAD~1")
    ap.add_argument("--save-frames", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)

    if args.cmd == "record":
        mod = load_impl("current")
        if hasattr(mod, "reset_hero_metrics"):
            mod.reset_hero_metrics()
        stem = os.path.join(OUT, args.label)
        fdir = stem + "_frames"
        os.makedirs(fdir, exist_ok=True)
        rows = []
        t0 = time.time()
        for i, frame in enumerate(stream()):
            if time.time() - t0 > args.seconds:
                break
            t = time.perf_counter()
            r = row_for(mod, frame, i, time.time() - t0, 0.0)
            r["ms"] = round((time.perf_counter() - t) * 1000, 2)
            rows.append(r)
            if args.save_frames:
                cv2.imwrite(os.path.join(fdir, "%05d.jpg" % i), frame)
            if i % 30 == 0:
                print("  frame %4d  state %s  pos %s" % (i, r["state"], (r["x"], r["y"])))
        path = stem + ".csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        print()
        print(json.dumps(summarise(path), indent=1))
        print("wrote %s and frames in %s" % (path, fdir))
        return 0

    if args.cmd == "summarise":
        print(json.dumps(summarise(args.target), indent=1))
        return 0

    if args.cmd == "replay":
        fdir = args.target
        files = sorted(glob.glob(os.path.join(fdir, "*.jpg")))
        if not files:
            print("no frames in %s" % fdir)
            return 1
        mod = load_impl(args.impl)
        if hasattr(mod, "reset_hero_metrics"):
            mod.reset_hero_metrics()
        rows = []
        for i, path in enumerate(files):
            frame = cv2.imread(path)
            t = time.perf_counter()
            r = row_for(mod, frame, i, i * 0.1, 0.0)
            r["ms"] = round((time.perf_counter() - t) * 1000, 2)
            rows.append(r)
        out = os.path.join(OUT, "replay_%s.csv" % args.impl.replace("/", "_"))
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        print(json.dumps(summarise(out), indent=1))
        print("wrote %s" % out)
        return 0

    if args.cmd == "compare":
        a, b = summarise(args.target), summarise(args.second)
        keys = [("tracked_pct", "tracked %"), ("lost_frames", "lost frames"),
                ("false_jumps", "false jumps"), ("max_jump_px", "max jump px"),
                ("ms_median", "median ms"), ("ms_p95", "p95 ms"),
                ("position_step_median_px", "median step px")]
        print("  %-16s %14s %14s" % ("metric", "old", "new"))
        for k, label in keys:
            print("  %-16s %14s %14s" % (label, a.get(k), b.get(k)))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
