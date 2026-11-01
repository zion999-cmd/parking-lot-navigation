#!/usr/bin/env python3
"""Live view of the bay magnetometers, with the baseline subtracted.

The number that matters for bay detection is not |B| but how far it has moved from the value with
nothing on the bay. This polls /api/mag, keeps a running baseline per channel, and prints the
deviation, so a magnet or a toy car passing over a sensor shows up as a number rather than as
something you have to squint at.

    python tools/bench/mag_watch.py                    # board at 192.168.3.200
    python tools/bench/mag_watch.py --host 192.168.3.118 --seconds 120

Keys: Ctrl-C to stop. `--baseline 10` averages the first 10 samples as "empty".
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


def fetch(host: str, timeout: float = 5.0) -> dict | None:
    try:
        with urllib.request.urlopen("http://%s/api/mag" % host, timeout=timeout) as fh:
            return json.loads(fh.read().decode("utf-8", "replace"))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="192.168.3.200")
    ap.add_argument("--seconds", type=float, default=0, help="0 = run until Ctrl-C")
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--baseline-n", type=int, default=8,
                    help="samples averaged as the empty-bay baseline")
    args = ap.parse_args()

    print("polling http://%s/api/mag  (Ctrl-C to stop)" % args.host)
    first = fetch(args.host)
    if first is None:
        print("  no answer - is the board on the network?")
        return 1
    if not first.get("count"):
        print("  board answered but found no sensors; diag:")
        for part in str(first.get("diag", "")).replace(";", "\n").split("\n"):
            if part.strip():
                print("   ", part.strip())
        return 1

    base: dict[int, list[float]] = {}
    locked: dict[int, float] = {}
    spike: dict[int, float] = {}
    t0 = time.time()
    try:
        while True:
            if args.seconds and time.time() - t0 > args.seconds:
                break
            d = fetch(args.host)
            if d is not None:
                row = []
                for s in d.get("sensors", []):
                    ch, b = s["ch"], float(s["field_ut"])
                    if ch not in locked:
                        base.setdefault(ch, []).append(b)
                        if len(base[ch]) >= args.baseline_n:
                            locked[ch] = statistics.median(base[ch])
                            spike[ch] = 0.0
                        row.append("ch%d %5.1f (baseline...)" % (ch, b))
                    else:
                        delta = b - locked[ch]
                        if abs(delta) > abs(spike.get(ch, 0.0)):
                            spike[ch] = delta
                        bar = "#" * min(30, int(abs(delta) * 3))
                        row.append("ch%d %5.1f  d=%+5.1f %-30s" % (ch, b, delta, bar))
                print("  " + " | ".join(row) + "   %s" % time.strftime("%H:%M:%S"))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass

    print()
    print("  largest deviation seen per channel:")
    for ch in sorted(spike):
        print("    ch%d  %+6.1f uT   (baseline %.1f)" % (ch, spike[ch], locked[ch]))
    print("  A magnet held near one sensor should move that channel by tens of uT and leave the")
    print("  others alone. If several channels move together, the mux is not switching and each")
    print("  reading is really the same sensor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
