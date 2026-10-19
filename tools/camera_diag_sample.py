#!/usr/bin/env python3
"""Read diagnostic counters around the existing camera-only benchmark.

No register writes; requires zero stream clients. Output stays under the OTA release.
"""
import argparse
import json
from pathlib import Path
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--host", required=True)
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

def fetch(path, post=False):
    request = urllib.request.Request(args.host.rstrip("/") + path,
                                     method="POST" if post else "GET")
    with urllib.request.urlopen(request, timeout=40) as response:
        return json.load(response)

def save(name, value):
    (args.out / name).write_text(json.dumps(value, indent=2) + "\n")

before = fetch("/api/camera-diag")
save("before.json", before)
bench = fetch("/api/bench", True)
save("bench.json", bench)
after = fetch("/api/camera-diag")
save("after.json", after)
assert before["firmware"] == after["firmware"]
dt = (after["sample_us"] - before["sample_us"]) / 1e6
assert dt > 0, "Device restarted during sampling"
delta = {key: (after["driver"][key] - value) % 2**32
         for key, value in before["driver"].items()}
summary = {
    "firmware": after["firmware"], "interval_seconds": dt,
    "gpio_rising_hz": ((after["gpio_rising"] - before["gpio_rising"]) % 2**32) / dt,
    "gpio_falling_hz": ((after["gpio_falling"] - before["gpio_falling"]) % 2**32) / dt,
    "driver_delta": delta,
    "driver_rates": {key: value / dt for key, value in delta.items()},
    "bench_fps": bench["fps"],
    "changed_regs": {key: [value, after["regs"][key]]
                     for key, value in before["regs"].items() if value != after["regs"][key]},
}
save("summary.json", summary)
print(json.dumps(summary, indent=2))
