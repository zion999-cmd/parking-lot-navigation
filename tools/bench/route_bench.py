"""Batch the route planner and time it. Offline, no camera, no HTTP, no production change.

    python tools/bench/route_bench.py                  # 8 starts x 48 targets
    python tools/bench/route_bench.py --starts 12

The planner is a pure function of two documents - the map and a state snapshot - so this
calls route.plan() directly, which is what makes the result reproducible: no network, no
detector, no timing noise from the camera. All bays are set EMPTY because the planner
refuses a target that is occupied or unknown, and the question here is only whether a route
exists and how long the search takes.

Writes tools/bench/out/route_bench.csv (one row per case) and prints the summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "vision", "src"))

import route                                   # noqa: E402

MAP_PATH = os.path.join(ROOT, "vision", "out", "parking_map.json")
OUT_DIR = os.path.join(HERE, "out")


def pick_starts(map_doc, count):
    """Evenly spaced road nodes, so the starts spread over the lot rather than clustering."""
    nodes = [n for n in (map_doc.get("road_network", {}).get("nodes") or [])
             if "x" in n and "y" in n and "id" in n]
    if not nodes:
        return []
    step = max(1, len(nodes) // count)
    return nodes[::step][:count]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=8)
    args = ap.parse_args()

    map_doc = json.load(open(MAP_PATH, encoding="utf-8"))
    spots = [str(s["id"]) for s in map_doc.get("spots") or []]
    starts = pick_starts(map_doc, args.starts)
    if not spots or not starts:
        print("map has no spots or no road nodes")
        return 1
    print("  targets %d   starts %d   -> %d cases" % (len(spots), len(starts), len(spots) * len(starts)))

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    ok_count = 0
    for start in starts:
        state = {
            "timestamp": 0,
            "vehicle": {"id": "hero", "visible": True,
                        "x": float(start["x"]), "y": float(start["y"]),
                        "heading": 0.0, "front_is_known": True},
            "spots": {sid: {"state": "EMPTY", "changed_ratio": 0.0} for sid in spots},
        }
        for target in spots:
            t0 = time.perf_counter()
            result = route.plan(map_doc, state, target)
            ms = (time.perf_counter() - t0) * 1000.0
            ok_count += 1 if result.ok else 0
            rows.append({"start": start["id"], "target": target, "ok": int(result.ok),
                         "nodes": len(result.nodes), "length_px": round(float(result.distance), 1),
                         "planning_ms": round(ms, 3), "reason": result.reason or ""})

    path = os.path.join(OUT_DIR, "route_bench.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    times = sorted(r["planning_ms"] for r in rows)
    lengths = [r["length_px"] for r in rows if r["ok"]]
    def pct(p):
        return times[min(len(times) - 1, int(len(times) * p))]
    print("  success      %d / %d = %.1f%%" % (ok_count, len(rows), 100.0 * ok_count / len(rows)))
    if lengths:
        print("  route length mean %.0f px  min %.0f  max %.0f" % (
            statistics.mean(lengths), min(lengths), max(lengths)))
    print("  planning ms  mean %.2f   p50 %.2f   p95 %.2f   max %.2f" % (
        statistics.mean(times), statistics.median(times), pct(0.95), times[-1]))
    failed = [r for r in rows if not r["ok"]]
    if failed:
        reasons = {}
        for r in failed:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        print("  failures: %s" % reasons)
    print("  csv          %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
