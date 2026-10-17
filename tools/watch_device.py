"""Watch the ESP32 without the serial cable: is it off, looping, or up?

The device's own log - the panic that says why it reset - only exists on the serial line. When that
is not plugged in, the next best evidence is the network: a board that is up answers, a board in a
reboot loop appears and disappears, and a board that never reaches Wi-Fi answers nothing at all.
Those three cases have different causes and only the third needs no further explanation.

    python tools/watch_device.py                 # 120 s, 2 s interval
    python tools/watch_device.py --seconds 600

Each change of state is printed with a timestamp, and every response prints the uptime from the
status page when it can be parsed, so a board that reboots every few seconds cannot hide inside an
"up" that is sampled too coarsely.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
import urllib.request


def ping(host: str, timeout_ms: int = 800) -> bool:
    r = subprocess.run(["ping", "-c", "1", "-W", str(timeout_ms), host],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0


def fetch_uptime(host: str) -> str:
    """Whatever the status page says about version/uptime, as one short line."""
    try:
        with urllib.request.urlopen(f"http://{host}/", timeout=2.0) as fh:
            body = fh.read(4096).decode("utf-8", "replace")
    except Exception as exc:
        return f"http: {type(exc).__name__}"
    for pattern in (r"uptime[^0-9]{0,12}([0-9]+)s", r"([0-9]+)\s*s\s*a?g?o",
                    r"Firmware[^<]{0,40}"):
        m = re.search(pattern, body, re.IGNORECASE)
        if m:
            return m.group(1) if m.groups() else m.group(0)
    return "page ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.3.118")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()

    print(f"  watching {args.host} for {args.seconds:.0f}s every {args.interval:.0f}s "
          f"(Ctrl-C to stop)")
    started = time.time()
    state = None
    changes = 0
    ups = 0
    while time.time() - started < args.seconds:
        up = ping(args.host)
        now = time.strftime("%H:%M:%S")
        if up and state is not False:
            ups += 1
            detail = fetch_uptime(args.host)
            print(f"  [{now}] UP    {detail}")
        if state is not None and up != state:
            changes += 1
            print(f"  [{now}] {'UP' if up else 'DOWN'}  <-- state change #{changes}")
        elif state is None:
            print(f"  [{now}] {'UP' if up else 'DOWN'}  (initial)")
        state = up
        time.sleep(args.interval)

    print()
    if changes == 0 and state is False:
        print("  never answered: the board is not reaching Wi-Fi at all. Not a crash loop in the")
        print("  usual sense - check power first (is the serial cable also its supply?), then that")
        print("  it is not held in reset.")
        print("  With the serial line back, the panic reason is in the boot log.")
    elif changes == 0:
        print("  answered every time: it is up and stable. If it still shows as restarting,")
        print("  the restarts are on the PC side - check ./start.sh status and run/vision.log.")
    else:
        print(f"  {changes} state changes in {args.seconds:.0f}s: the board is appearing and")
        print("  disappearing, which is a reboot loop that reaches Wi-Fi. Fast changes (seconds)")
        print("  point at a reset or power problem; slow ones at a crash after boot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
