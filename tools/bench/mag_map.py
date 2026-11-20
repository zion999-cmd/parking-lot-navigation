#!/usr/bin/env python3
"""Find out which TCA9548A channel is physically under which bay - by measurement.

P0005 says not to assume B1=CH0, B2=CH1, ...  and that is right: the wiring was rebuilt
several times while these sensors were being brought up, and a wrong mapping produces a
detector that is confidently wrong on two bays at once. This tool asks the operator to put
one car on one bay at a time, measures which channel moves, and only writes the mapping when
the assignment is unambiguous.

    # discover and rewrite vision/config/mag_bays.json
    tools/bench/mag_map.py --host 192.168.3.118 --bays B1 B2 B3 B4 --write

    # look without writing
    tools/bench/mag_map.py --host 192.168.3.118 --bays B1 B2 B3 B4

The phases are: empty baseline, then one bay at a time. Every sample is deduplicated on the
board's own read counter - the HTTP poll rate and the sensor sample rate are unrelated, so two
polls can legitimately return the same reading, and counting those twice would understate the
noise.
"""

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG = os.path.join(ROOT, "vision", "config", "mag_bays.json")

# A channel is only attributed to a bay if it moves clearly more than the next best one;
# anything closer than this is a crosstalk case and must be looked at, not guessed.
MIN_MARGIN = 1.6


def read_mag(host, timeout=2.0):
    url = f"http://{host}/api/mag"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:                                        # noqa: BLE001
        return None


def sample(host, seconds, rate_hz, label, log):
    """Return {(ch, axis): [counts]} plus the number of duplicate reads that were dropped."""
    interval = 1.0 / max(rate_hz, 0.5)
    deadline = time.time() + seconds
    values = {}
    scales = {}
    last_reads = {}
    duplicates = 0
    misses = 0
    fresh = 0
    while time.time() < deadline:
        doc = read_mag(host)
        if doc is None:
            misses += 1
        else:
            for reading in doc.get("sensors") or []:
                channel = reading.get("ch")
                if channel is None or not reading.get("ok", True):
                    continue
                reads = reading.get("reads")
                # Same counter as last time means the chip has not sampled again: not new data.
                if reads is not None and last_reads.get(channel) == reads:
                    duplicates += 1
                    continue
                last_reads[channel] = reads
                fresh += 1
                scales[channel] = float(reading.get("ut_per_lsb") or 0.0)
                for axis in ("x", "y", "z"):
                    values.setdefault((channel, axis), []).append(float(reading.get(axis, 0)))
        if log and sys.stdout.isatty():
            sys.stdout.write(f"\r  {label}: {fresh:4d} fresh, {time.time() - deadline:+.1f} s   ")
            sys.stdout.flush()
        time.sleep(interval)
    if log and sys.stdout.isatty():
        sys.stdout.write("\n")
    return values, scales, duplicates, misses, fresh


def stats(values):
    out = {}
    for key, series in values.items():
        if not series:
            continue
        out[key] = (statistics.fmean(series),
                    statistics.stdev(series) if len(series) > 1 else 0.0)
    return out


def delta(empty_stats, bay_stats):
    """Per-channel change, in microtesla, for each axis and for the vector length."""
    channels = sorted({ch for ch, _ in bay_stats})
    result = {}
    for channel in channels:
        axes = {}
        for axis in ("x", "y", "z"):
            key = (channel, axis)
            if key in empty_stats and key in bay_stats:
                axes[axis] = bay_stats[key][0] - empty_stats[key][0]
        if len(axes) == 3:
            axes["norm"] = (axes["x"] ** 2 + axes["y"] ** 2 + axes["z"] ** 2) ** 0.5
        result[channel] = axes
    return result


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="192.168.3.118")
    parser.add_argument("--bays", nargs="+", default=["B1", "B2", "B3", "B4"])
    parser.add_argument("--seconds", type=float, default=10.0, help="seconds per phase")
    parser.add_argument("--rate", type=float, default=4.0)
    parser.add_argument("--write", action="store_true",
                        help="rewrite the mapping from what the phases agreed on (needs a clean result)")
    parser.add_argument("--matrix", action="store_true",
                        help="only print the full channel x bay magnitude matrix and stop")
    parser.add_argument("--write-map", metavar="B4=4,B2=1",
                        help="write an explicit mapping, for when the matrix has been read by hand")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    probe = read_mag(args.host)
    if probe is None:
        print("error: no answer from /api/mag", file=sys.stderr)
        return 2
    channels = sorted({r["ch"] for r in probe.get("sensors") or []})
    if not channels:
        print("error: the board reports no sensors - power-cycle it", file=sys.stderr)
        return 2
    print(f"{len(channels)} sensor(s): {channels}\n")

    input("  take EVERY car off the sandbox, then press Enter... ")
    empty_raw, scales, dupes, misses, fresh = sample(args.host, args.seconds, args.rate, "empty", True)
    empty = stats(empty_raw)
    print(f"  {fresh} fresh samples, {dupes} duplicate polls dropped, {misses} miss(es)\n")

    findings = {}
    matrix = {}
    for bay in args.bays:
        input(f"  put ONE car on {bay} and nothing else, then press Enter... ")
        raw, scales_here, dupes, misses, fresh = sample(args.host, args.seconds, args.rate, bay, True)
        scales.update(scales_here)
        change = delta(empty, stats(raw))
        matrix[bay] = {ch: abs(v.get("norm", 0.0)) * (scales.get(ch) or 0.0)
                       for ch, v in change.items()}
        scored = sorted(((abs(v.get("norm", 0.0)), ch, v) for ch, v in change.items()), reverse=True)
        print(f"  {bay}: {fresh} fresh samples, {dupes} duplicate polls dropped")
        for magnitude, channel, axes in scored:
            scale = scales.get(channel) or 0.0
            print(f"    ch{channel}: dX={axes.get('x', 0) * scale:+.2f} dY={axes.get('y', 0) * scale:+.2f} "
                  f"dZ={axes.get('z', 0) * scale:+.2f} |d|={magnitude * scale:.2f} uT")
        top = [(m, ch) for m, ch, _ in scored]
        if not top or top[0][0] == 0:
            findings[bay] = None
            print("    -> no channel moved; check that the car is over a sensor\n")
            continue
        best = top[0][0]
        second = top[1][0] if len(top) > 1 else 0.0
        margin = best / second if second > 0 else float("inf")
        if margin < MIN_MARGIN:
            findings[bay] = None
            print(f"    -> ambiguous (best/second = {margin:.2f} < {MIN_MARGIN}): treat as crosstalk\n")
        else:
            findings[bay] = top[0][1]
            print(f"    -> ch{top[0][1]} (margin {margin:.2f})\n")

    if matrix:
        all_channels = sorted({ch for row in matrix.values() for ch in row})
        print("\nchannel x bay, |delta| in uT (rows = bays, columns = channels)")
        print("      " + "".join(f"{'ch%d' % ch:>9}" for ch in all_channels))
        for bay in args.bays:
            row = matrix.get(bay, {})
            cells = "".join(f"{row.get(ch, 0.0):>9.2f}" for ch in all_channels)
            best = max(row.items(), key=lambda kv: kv[1])[0] if row else None
            print(f"  {bay:<4}{cells}   -> ch{best}")

    if args.write_map:
        pairs = {}
        for item in args.write_map.split(","):
            bay, _, channel = item.partition("=")
            bay, channel = bay.strip(), channel.strip()
            if not bay or not channel.isdigit():
                print(f"error: cannot parse {item!r}, want BAY=CHANNEL", file=sys.stderr)
                return 2
            pairs[bay] = int(channel)
        clashes = {}
        for bay, channel in pairs.items():
            clashes.setdefault(channel, []).append(bay)
        duplicated = {c: b for c, b in clashes.items() if len(b) > 1}
        if duplicated:
            print(f"error: one channel for several bays: {duplicated}", file=sys.stderr)
            return 1
        document = {
            "comment": ("Bay -> TCA9548A channel, established by measurement with "
                        "tools/bench/mag_map.py. Never assume B1=CH0."),
            "established_by": "tools/bench/mag_map.py --write-map",
            "established_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "bays": {bay: {"channel": pairs.get(bay)} for bay in args.bays},
        }
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        tmp = args.config + ".writing"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(document, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, args.config)
        print(f"\n  wrote {args.config}")
        print(f"  mapped: {pairs}")
        for bay in args.bays:
            if bay not in pairs:
                print(f"  {bay} left null - it will read UNKNOWN, not occupied")
        return 0

    if args.matrix:
        print("\n  (matrix only - nothing written)")
        return 0

    print("mapping:")
    used = {}
    for bay in args.bays:
        channel = findings.get(bay)
        print(f"  {bay} -> {'ch%d' % channel if channel is not None else 'UNRESOLVED'}")
        if channel is not None:
            used.setdefault(channel, []).append(bay)
    clashes = {ch: bays for ch, bays in used.items() if len(bays) > 1}
    if clashes:
        print(f"\n  ✗ the same channel was assigned to more than one bay: {clashes}")
        print("    nothing written; re-run and make sure only one bay is occupied per phase")
        return 1
    if any(channel is None for channel in findings.values()):
        print("\n  ✗ incomplete mapping - not writing it. A partial map must not produce occupancy.")
        return 1

    if args.write:
        document = {
            "comment": ("Bay -> TCA9548A channel, established by measurement with "
                        "tools/bench/mag_map.py. Never assume B1=CH0."),
            "established_by": "tools/bench/mag_map.py",
            "established_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "bays": {bay: {"channel": findings[bay]} for bay in args.bays},
        }
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        tmp = args.config + ".writing"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(document, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, args.config)
        print(f"\n  wrote {args.config}")
    else:
        print("\n  (dry run - pass --write to save the mapping)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
