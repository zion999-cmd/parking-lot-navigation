"""Read the board's console over USB serial, with timestamps, without holding it in reset.

On this board RTS high holds the chip in reset, so the port is opened and both handshake lines are
driven low immediately; pyserial asserts DTR on open, which is why the lines are set right after it
rather than before.

    python tools/serial_capture.py --seconds 120
    python tools/serial_capture.py --seconds 120 --out run/serial_capture.log

The output is also written to a file, because the interesting part is the *next* reset and the
evidence has to survive the terminal scroll.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/cu.usbserial-14320")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "run", "serial_capture.log"))
    args = ap.parse_args()

    try:
        import serial                                     # pyserial
    except ImportError:
        print("  pyserial is not available in this interpreter:")
        print(f"    {sys.executable}")
        print("  try: /usr/local/Caskroom/miniconda/base/envs/vision-dev/bin/python tools/serial_capture.py")
        return 2

    ser = serial.Serial(args.port, args.baud, timeout=0.2)
    # RTS high resets this board; DTR is asserted by open(). Both low, immediately.
    ser.dtr = False
    ser.rts = False

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    started = time.time()
    print(f"  reading {args.port} at {args.baud} for {args.seconds:.0f}s -> {args.out}")
    lines = 0
    # Only reset reasons and panics. The first version included "E (" and "W (" and the 58 matches
    # it reported were ordinary log lines - the one line that matters, the ROM's rst:0x mask, scrolled
    # past unseen. A filter that matches everything explains nothing.
    marker = ("rst:0x", "Brownout", "Guru Meditation", "Backtrace", "panic", "assert failed",
              "abort()", "ESP-ROM", "boot: Load", "reboot", "WDT", "watchdog")
    interesting = 0
    with open(args.out, "w", encoding="utf-8", errors="replace") as fh:
        while time.time() - started < args.seconds:
            raw = ser.read(4096)
            if not raw:
                continue
            text = raw.decode("utf-8", errors="replace")
            stamp = time.strftime("%H:%M:%S")
            for line in text.replace("\r", "").split("\n"):
                if not line.strip():
                    continue
                lines += 1
                fh.write(f"[{stamp}] {line}\n")
                fh.flush()
                if any(m in line for m in marker):
                    interesting += 1
                    print(f"  [{stamp}] {line[:150]}")
    ser.close()
    print(f"\n  {lines} lines, {interesting} matched reset/panic markers")
    print(f"  full capture: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
