"""Turn the recorded acceptance cases into an occupancy threshold.

The threshold is supposed to come from measurements, not from taste. Record each
case with:

    python src/main.py --headless --duration 20 --csv data/cases/case1_all_empty.csv

then describe what was physically true while each file was recorded, either in a
labels JSON (see --write-template) or on the command line:

    python tests/fit_threshold.py --cases data/cases/*.csv --labels data/cases/labels.json

It prints the changed_ratio distributions, the overlap (if any), and the
threshold that separates EMPTY from OCCUPIED with the largest margin.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Dict, List, Tuple


def read_case(path: str) -> Dict[str, List[float]]:
    """{spot_id: [changed_ratio, ...]} for one recording."""
    per_spot: Dict[str, List[float]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            per_spot.setdefault(row["spot_id"], []).append(float(row["changed_ratio"]))
    return per_spot


def write_template(cases: List[str], out: str) -> None:
    doc = {}
    for path in cases:
        stem = os.path.splitext(os.path.basename(path))[0]
        doc[stem] = {sid: "EMPTY" for sid in sorted(read_case(path))}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    print(f"wrote label template to {out}")
    print("edit it so each spot says EMPTY or OCCUPIED as it physically was, then re-run.")


def collect(cases: List[str], labels: Dict[str, Dict[str, str]]):
    empty: List[Tuple[str, float]] = []
    occupied: List[Tuple[str, float]] = []
    unknown = []
    for path in cases:
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem not in labels:
            unknown.append(stem)
            continue
        for sid, values in read_case(path).items():
            want = labels[stem].get(sid)
            if want is None:
                continue
            want = want.strip().upper()
            if want not in ("EMPTY", "OCCUPIED"):
                continue
            bucket = occupied if want == "OCCUPIED" else empty
            bucket.extend((f"{stem}/{sid}", v) for v in values)
    return empty, occupied, unknown


def stats(values: List[float]) -> str:
    if not values:
        return "no samples"
    vs = sorted(values)
    n = len(vs)
    mean = sum(vs) / n
    p95 = vs[min(int(0.95 * n), n - 1)]
    p05 = vs[min(int(0.05 * n), n - 1)]
    return (f"n={n:5d}  min={vs[0]:.4f}  p05={p05:.4f}  mean={mean:.4f}  "
            f"p95={p95:.4f}  max={vs[-1]:.4f}")


def best_threshold(empty: List[float], occupied: List[float]) -> Tuple[float, float, int, int]:
    """Largest gap between the empty and occupied distributions.

    Returns (threshold, margin, false_positives, false_negatives).
    """
    if not empty or not occupied:
        return 0.0, 0.0, 0, 0
    cands = sorted(set(empty + occupied))
    mids = [cands[0] - 1e-3] + [(a + b) / 2 for a, b in zip(cands, cands[1:])] + [cands[-1] + 1e-3]
    best = None
    for t in mids:
        fp = sum(1 for v in empty if v >= t)             # said occupied, was empty
        fn = sum(1 for v in occupied if v < t)           # said empty, was occupied
        margin = min(occupied) - max(empty)
        key = (-(fp + fn), margin)
        if best is None or key > best[0]:
            best = (key, t, margin, fp, fn)
    _, t, margin, fp, fn = best
    return t, margin, fp, fn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", required=True, help="recorded CSV files (globs ok)")
    ap.add_argument("--labels", help="JSON: {case_stem: {spot_id: EMPTY|OCCUPIED}}")
    ap.add_argument("--write-template", metavar="JSON",
                    help="write a labels template for the given cases and exit")
    args = ap.parse_args()

    cases: List[str] = []
    for pattern in args.cases:
        cases.extend(sorted(glob.glob(pattern)))
    cases = sorted(set(cases))
    if not cases:
        print("error: no CSV files matched", file=sys.stderr)
        return 2

    if args.write_template:
        write_template(cases, args.write_template)
        return 0

    if not args.labels:
        print("error: --labels is required (or use --write-template first)", file=sys.stderr)
        return 2
    with open(args.labels, encoding="utf-8") as fh:
        labels = json.load(fh)

    empty, occupied, unknown = collect(cases, labels)
    if unknown:
        print(f"warning: no labels for {', '.join(unknown)} - skipped")
    if not empty or not occupied:
        print("error: need both EMPTY and OCCUPIED samples to fit a threshold",
              file=sys.stderr)
        return 2

    e = [v for _, v in empty]
    o = [v for _, v in occupied]

    print(f"{len(cases)} recording(s), {len(labels)} label entr(ies)\n")
    print("changed_ratio distributions")
    print(f"  EMPTY    {stats(e)}")
    print(f"  OCCUPIED {stats(o)}")

    margin = min(o) - max(e)
    print(f"\nseparation: min(OCCUPIED)={min(o):.4f}  max(EMPTY)={max(e):.4f}  "
          f"margin={margin:+.4f}")
    if margin <= 0:
        print("  ⚠ the distributions OVERLAP - a single threshold cannot separate them.")
        print(f"  {sum(1 for v in e if v >= min(o))} EMPTY sample(s) sit at or above the "
              f"lowest OCCUPIED value.")

    t, _, fp, fn = best_threshold(e, o)
    print(f"\nrecommended changed_ratio_threshold = {t:.3f}")
    print(f"  false OCCUPIED (empty read as occupied): {fp}/{len(e)}")
    print(f"  false EMPTY    (occupied read as empty) : {fn}/{len(o)}")

    # Per-spot view: sometimes one global value is fine but one bay misbehaves.
    per_spot: Dict[str, Dict[str, List[float]]] = {}
    for tag, v in empty:
        per_spot.setdefault(tag.split("/")[1], {"E": [], "O": []})["E"].append(v)
    for tag, v in occupied:
        per_spot.setdefault(tag.split("/")[1], {"E": [], "O": []})["O"].append(v)
    print("\nper spot")
    for sid in sorted(per_spot, key=lambda s: (len(s), s)):
        d = per_spot[sid]
        if d["E"] and d["O"]:
            m = min(d["O"]) - max(d["E"])
            print(f"  spot {sid}: max(EMPTY)={max(d['E']):.4f} min(OCCUPIED)={min(d['O']):.4f} "
                  f"margin={m:+.4f}")
        elif d["E"]:
            print(f"  spot {sid}: only EMPTY samples (max={max(d['E']):.4f})")
        else:
            print(f"  spot {sid}: only OCCUPIED samples (min={min(d['O']):.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
