#!/usr/bin/env python3
"""Is any of this RSSI separable, or is it noise? Measure it, do not argue about it.

For each position the operator names, sample /api/rfid and record, per antenna port: how often
the tag was read, and the spread of the RSSI it reported. Then compare the spread *between*
positions with the spread *within* one position. If the between-position difference is only a
couple of times the within-position scatter, a location estimate would be a random number with a
nice label on it.

    tools/rfid/rssi_survey.py --positions A1 A2 B1 B2 --seconds 12

What this cannot do is make the ports simultaneous: the firmware polls one port at a time, roughly
1.1 s per port with four ports. A tag that moves between two reads carries that movement into the
comparison. Keep the tag still for the whole of each position.

Standard library only; reuses tools/rfid/rfid_presence.py for the HTTP read.
"""

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def fetch(host, timeout=2.0):
    try:
        with urllib.request.urlopen(f"http://{host}/api/rfid", timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:                                        # noqa: BLE001
        return None


def sample_position(host, seconds, interval=0.25, say=print):
    """Reads per port and the RSSI values seen, over one stationary position."""
    by_port = {}
    reads = {}
    deadline = time.time() + seconds
    while time.time() < deadline:
        document = fetch(host)
        if document is None:
            say("  (no answer from the reader)")
        else:
            for reading in document.get("tags") or []:
                port = reading.get("port")
                rssi = reading.get("rssi")
                epc = (reading.get("epc") or "").upper()
                if port is None or rssi is None:
                    continue
                reads.setdefault(port, set()).add(reading.get("reads"))
                by_port.setdefault(port, []).append(float(rssi))
        if sys.stdout.isatty():
            say(f"\r  {seconds - max(0.0, deadline - time.time()):5.1f}/{seconds:g} s  "
                f"{ {p: len(v) for p, v in by_port.items()} }   ", )
        time.sleep(interval)
    if sys.stdout.isatty():
        say("\n")
    return by_port


def summarise(by_port):
    out = {}
    for port, values in by_port.items():
        out[port] = {
            "n": len(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "spread": max(values) - min(values),
        }
    return out


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="192.168.3.118")
    parser.add_argument("--positions", nargs="+", required=True,
                        help="labels for the places you will put the tag, one at a time")
    parser.add_argument("--seconds", type=float, default=12.0, help="seconds per position")
    parser.add_argument("--json", metavar="PATH", help="write the raw measurements here")
    args = parser.parse_args(argv)

    results = {}
    for position in args.positions:
        input(f"  put the tag at {position} and keep it still, then press Enter... ")
        print(f"  measuring {position} for {args.seconds:g} s")
        results[position] = summarise(sample_position(args.host, args.seconds))

    ports = sorted({port for row in results.values() for port in row})
    if not ports:
        print("\nno tag was read at any position - nothing to compare")
        return 1

    print("\nRSSI per position and port (dBm), and how many readings it came from")
    header = "  position  " + "".join(f"{'port %d' % p:>22}" for p in ports)
    print(header)
    for position, row in results.items():
        cells = ""
        for port in ports:
            entry = row.get(port)
            cells += (f"{'—':>22}" if entry is None
                      else f"{entry['median']:>8.0f} [{entry['min']:.0f}..{entry['max']:.0f}] n={entry['n']:<3}")
        print(f"  {position:<10}{cells}")

    # Separability: how much the median moves between positions, against how much one position
    # alone scatters. Ratio near 1 means the difference is indistinguishable from noise.
    print("\nseparability per port (between-position spread vs within-position spread)")
    separable = 0
    for port in ports:
        medians = [row[port]["median"] for row in results.values() if port in row]
        spreads = [row[port]["spread"] for row in results.values() if port in row]
        if len(medians) < 2 or not spreads:
            print(f"  port {port}: only {len(medians)} position(s) had this port - cannot compare")
            continue
        between = max(medians) - min(medians)
        within = statistics.fmean(spreads) or 0.1
        ratio = between / within
        verdict = "coarse separation only" if ratio >= 2 else "not separable from noise"
        if ratio >= 2:
            separable += 1
        print(f"  port {port}: between {between:5.1f} dB, within {within:4.1f} dB, "
              f"ratio {ratio:4.1f}  -> {verdict}")

    print()
    if separable == 0:
        print("No port separates the positions by more than twice its own scatter. A location\n"
              "estimate built on this would be a guess with a label on it.")
    else:
        print(f"{separable} port(s) show coarse separation. That supports zone-level statements\n"
              "('near this antenna'), not bay-level position: one port at a time, 1.1 s apart, and\n"
              "usually only the nearest port reads the tag at all.")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
