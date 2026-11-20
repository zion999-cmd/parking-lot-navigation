#!/usr/bin/env python3
"""Measure whether a bay magnetometer can actually tell "empty" from "car".

The interesting quantity is not the field itself - it is the *change* a car produces compared
with how much the reading wanders on its own. This runs two phases per round (bay empty, then
bay occupied), keeps every sample, and reports:

    baseline   mean |B| with the bay empty, and the noise (sigma) around it
    with car   mean |B| with the car in place
    delta      mean(with car) - baseline, per channel and per axis
    SNR        |delta| / sigma, which is what decides whether a threshold can exist at all

Each sensor has its own baseline: measured on this rig the four channels differ by more than
20 uT, so a single absolute threshold would call some bays permanently occupied.

    # guided, 3 rounds of 30 s each, prompts between phases
    tools/bench/mag_bench.py --host 192.168.3.118

    # one quick check, 5 s per phase, no prompts
    tools/bench/mag_bench.py --host 192.168.3.118 --seconds 5 --rounds 1 --yes

    # keep the raw samples for offline work
    tools/bench/mag_bench.py --host 192.168.3.118 --csv mag_bench.csv

Exit status is 0 when every channel cleared the SNR bar, 1 otherwise, so this can gate a deploy.
"""

import argparse
import csv
import json
import math
import statistics
import sys
import time
import urllib.request


# A change smaller than this multiple of the empty-bay noise cannot be thresholded reliably: the
# distributions overlap and the detector fires on both sides. 5 is the usual engineering floor for
# a presence decision with a single threshold at the midpoint.
SNR_FLOOR = 5.0


def read_once(url, timeout=2.0):
    """One /api/mag sample, or None if the board did not answer."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:                                        # noqa: BLE001 - a miss is data too
        return None


def collect(url, seconds, rate_hz, label, say):
    """Sample for a fixed window. Returns {(ch, 'x'|'y'|'z'|'B'): [values]} plus miss count."""
    interval = 1.0 / max(rate_hz, 0.5)
    deadline = time.time() + seconds
    samples = {}
    misses = 0
    taken = 0
    # A progress line that stays on one row: a bench run is watched, not read.
    while time.time() < deadline:
        doc = read_once(url)
        if doc is None:
            misses += 1
        else:
            for reading in doc.get("sensors") or []:
                channel = reading.get("ch")
                if channel is None:
                    continue
                for axis in ("x", "y", "z"):
                    samples.setdefault((channel, axis), []).append(float(reading.get(axis, 0)))
                samples.setdefault((channel, "B"), []).append(float(reading.get("field_ut", 0.0)))
            taken += 1
        remaining = max(0.0, deadline - time.time())
        # Progress only on a terminal: piped through tee or a log the carriage returns pile up into
        # one unreadable line per sample.
        if sys.stdout.isatty():
            say(f"\r  {label}: {taken:4d} samples, {remaining:4.1f} s left   ")
        time.sleep(interval)
    if sys.stdout.isatty():
        say("\n")
    return samples, misses, taken


def window_means(values, size):
    """Means of consecutive groups of `size` samples.

    A bay decision never rests on one sample: the detector averages, and averaging shrinks white
    noise by sqrt(n). Reporting only the per-sample sigma makes a perfectly usable channel look
    hopeless - measured here at 1.0-1.5 uT per sample around a 2 uT signal, i.e. SNR 2, which
    becomes SNR 6 over a one-second window and 12 over five.
    """
    size = max(1, int(size))
    return [statistics.fmean(values[i:i + size])
            for i in range(0, len(values) - size + 1, size)]


def windowed_snr(empty, car, rate, seconds, sigma_sample=None):
    """delta, sigma and SNR after averaging over `seconds`.

    sigma comes from the spread of the windowed baseline when there are enough windows to measure it,
    and otherwise from the per-sample sigma divided by sqrt(n) - the whole point being that averaging
    a longer window shrinks the noise even when a short phase yields only one or two windows.
    """
    size = max(1, round(rate * seconds))
    e = window_means(empty, size)
    c = window_means(car, size)
    if not e or not c:
        return None, None, None
    delta = statistics.fmean(c) - statistics.fmean(e)
    if len(e) > 1:
        sigma = statistics.stdev(e)
    elif sigma_sample:
        sigma = sigma_sample / math.sqrt(size)
    else:
        sigma = 0.0
    snr = abs(delta) / sigma if sigma > 0 else None
    return delta, sigma, snr


def summarise(samples):
    """mean and sample sigma per key; sigma is None when there are fewer than two samples."""
    out = {}
    for key, values in samples.items():
        mean = statistics.fmean(values)
        sigma = statistics.stdev(values) if len(values) > 1 else None
        out[key] = (mean, sigma, len(values))
    return out


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="192.168.3.118", help="board address")
    parser.add_argument("--seconds", type=float, default=30.0, help="seconds per phase")
    parser.add_argument("--rounds", type=int, default=3, help="empty/car round pairs")
    parser.add_argument("--rate", type=float, default=4.0, help="samples per second")
    parser.add_argument("--yes", action="store_true", help="do not prompt between phases")
    parser.add_argument("--csv", metavar="PATH", help="also write every raw sample here")
    parser.add_argument("--snr-floor", type=float, default=SNR_FLOOR,
                        help=f"minimum |delta|/sigma to call a channel usable (default {SNR_FLOOR})")
    args = parser.parse_args(argv)

    url = f"http://{args.host}/api/mag"
    say = lambda text: (sys.stdout.write(text), sys.stdout.flush())
    say(f"reading {url}\n")

    probe = read_once(url)
    if probe is None:
        print("error: the board did not answer /api/mag - check --host and that the firmware is 0.2.78+",
              file=sys.stderr)
        return 2
    channels = sorted({r["ch"] for r in probe.get("sensors") or []})
    if not channels:
        print("error: the board answered but reports no sensors - power-cycle it so it re-runs\n"
              "       discovery, then check GET /api/mag",
              file=sys.stderr)
        return 2
    say(f"  {len(channels)} sensor(s) on channels {channels}\n\n")

    rows = []
    empty_runs, car_runs = [], []
    empty_raw, car_raw = [], []
    for rnd in range(args.rounds):
        say(f"round {rnd + 1}/{args.rounds}\n")
        if not args.yes:
            input("  make sure the bay is EMPTY, then press Enter... ")
        samples, misses, taken = collect(url, args.seconds, args.rate, "empty", say)
        empty_runs.append(summarise(samples))
        empty_raw.append(samples)
        say(f"  {taken} samples, {misses} miss(es)\n")

        if not args.yes:
            input("  put the car on the sensor, then press Enter... ")
        samples, misses, taken = collect(url, args.seconds, args.rate, "car  ", say)
        car_runs.append(summarise(samples))
        car_raw.append(samples)
        say(f"  {taken} samples, {misses} miss(es)\n\n")

    # Pool the rounds rather than averaging means: with few rounds a mean of means hides how much
    # the baseline itself moves between runs, which is part of the noise.
    def pooled(runs, channel, key):
        values = [runs[i][(channel, key)][0] for i in range(len(runs)) if (channel, key) in runs[i]]
        return statistics.fmean(values) if values else None

    print(f"{'ch':>3} {'empty |B|':>12} {'sigma':>7} {'car |B|':>12} {'delta':>9} {'SNR':>6}  verdict")
    print("-" * 66)
    usable = 0
    report = []
    for channel in channels:
        base = pooled(empty_runs, channel, "B")
        car = pooled(car_runs, channel, "B")
        sigma = statistics.fmean([r[(channel, "B")][1] for r in empty_runs
                                  if (channel, "B") in r and r[(channel, "B")][1] is not None] or [0.0])
        delta = (car - base) if (base is not None and car is not None) else None
        snr = (abs(delta) / sigma) if (delta is not None and sigma > 0) else None
        verdict = "usable" if (snr is not None and snr >= args.snr_floor) else "too weak"
        if verdict == "usable":
            usable += 1
        report.append({"channel": channel, "empty": base, "sigma": sigma, "car": car,
                       "delta": delta, "snr": snr, "verdict": verdict})
        print(f"{channel:>3} {base:>12.2f} {sigma:>7.2f} {car:>12.2f} {delta:>+9.2f} "
              f"{(snr if snr is not None else 0):>6.1f}  {verdict}")

    # Per-axis detail: a car usually changes one axis far more than the magnitude, so if |B| is too
    # weak the axes are where a usable signal would still be hiding.
    print("\nper-axis delta (mean of car minus empty) and its SNR")
    print(f"{'ch':>3} {'dX':>9} {'SNR':>6} {'dY':>9} {'SNR':>6} {'dZ':>9} {'SNR':>6}")
    for channel in channels:
        cells = []
        for axis in ("x", "y", "z"):
            base = pooled(empty_runs, channel, axis)
            car = pooled(car_runs, channel, axis)
            sigma = statistics.fmean([r[(channel, axis)][1] for r in empty_runs
                                      if (channel, axis) in r and r[(channel, axis)][1] is not None]
                                     or [0.0])
            delta = (car - base) if (base is not None and car is not None) else 0.0
            snr = (abs(delta) / sigma) if sigma > 0 else 0.0
            cells.append(f"{delta:>+9.1f} {snr:>6.1f}")
        print(f"{channel:>3} " + " ".join(cells))

    # The per-sample table above answers "is there a signal at all"; this one answers the question the
    # detector actually asks - can a threshold live here once the samples are averaged?
    for seconds in (1.0, 3.0):
        print(f"\naveraged over {seconds:g} s (sigma of the windowed baseline, SNR per axis and |B|)")
        print(f"{'ch':>3} {'axis':>5} {'delta_uT':>10} {'sigma_uT':>10} {'SNR':>6}")
        for channel in channels:
            for axis in ("x", "y", "z", "B"):
                # Window the raw samples, not the per-round means: a mean per round would leave one
                # value per round and nothing to average within.
                empty_vals, car_vals = [], []
                for run in empty_raw:
                    empty_vals.extend(run.get((channel, axis), []))
                for run in car_raw:
                    car_vals.extend(run.get((channel, axis), []))
                if not empty_vals or not car_vals:
                    continue
                sample_sigmas = [r[(channel, axis)][1] for r in empty_runs
                                 if (channel, axis) in r and r[(channel, axis)][1] is not None]
                sigma_sample = statistics.fmean(sample_sigmas) if sample_sigmas else None
                delta, sigma, snr = windowed_snr(empty_vals, car_vals, args.rate, seconds,
                                                 sigma_sample)
                if delta is None:
                    continue
                scale = 1.0 if axis == "B" else 0.1     # raw counts -> uT at the 30 G scale
                print(f"{channel:>3} {axis:>5} {delta * scale:>+10.2f} {sigma * scale:>10.2f} "
                      f"{(snr or 0):>6.1f}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["channel", "empty_mean_B", "sigma_B", "car_mean_B", "delta_B", "snr",
                             "verdict"])
            for row in report:
                writer.writerow([row["channel"], f"{row['empty']:.3f}", f"{row['sigma']:.3f}",
                                 f"{row['car']:.3f}", f"{row['delta']:+.3f}",
                                 f"{row['snr']:.2f}", row["verdict"]])
        print(f"\nwrote {args.csv}")

    print(f"\n{usable}/{len(channels)} channel(s) at or above SNR {args.snr_floor:g}")
    if usable < len(channels):
        print("A weak channel is not necessarily broken: check placement, that the car actually covers\n"
              "it, and whether the empty baseline was taken with the strip lit (light current is the\n"
              "largest interference source on this rig).")
    return 0 if usable == len(channels) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
