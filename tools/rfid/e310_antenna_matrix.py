#!/usr/bin/env python3
"""Which antenna actually sees which tag? A 4xN matrix, measured rather than assumed.

Four working ports is not the same as four usable zones. If antenna 2 can hear the tag in front of
antenna 1, then "which bay is this car in" cannot be answered by which antenna read it, and any
occupancy logic built on that would be guessing.

    python tools/rfid/e310_antenna_matrix.py --port /dev/cu.usbserial-14330 --rounds 4

Put one tag in front of each antenna first, well separated, then run it. The output is

    antenna   tag                       reads   RSSI
    1         E280F337...0D2E             4/4   -52 dBm
    2         E280F337...0D2E             0/4   -        <- 0 is what separation looks like
    2         E280F337...09B0             4/4   -58 dBm

A tag appearing under two antennas is leakage, and the number to watch is how strong that second
reading is compared with the first.
"""

from __future__ import annotations

import argparse
import os
import termios
import time
import tty

BAUDS = {9600: termios.B9600, 19200: termios.B19200, 38400: termios.B38400,
         57600: termios.B57600, 115200: termios.B115200}


def frame(cmd: int, params: bytes = b"") -> bytes:
    pl = len(params)
    body = bytes([0x00, cmd, (pl >> 8) & 0xFF, pl & 0xFF]) + params
    return b"\xbb" + body + bytes([sum(body) & 0xFF]) + b"\x7e"


def parse(data: bytes):
    out, i = [], 0
    while i < len(data):
        if data[i] != 0xBB:
            i += 1
            continue
        if i + 5 > len(data):
            break
        pl = (data[i + 3] << 8) | data[i + 4]
        end = i + 5 + pl + 2
        if end > len(data) or data[end - 1] != 0x7E:
            i += 1
            continue
        out.append((data[i + 1], data[i + 2], data[i + 5:i + 5 + pl]))
        i = end
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--rounds", type=int, default=4, help="polls per antenna")
    ap.add_argument("--ants", default="1,2,3,4")
    args = ap.parse_args()
    ants = [int(a) for a in args.ants.split(",")]

    fd = os.open(args.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    tty.setraw(fd)
    t = termios.tcgetattr(fd)
    t[4] = t[5] = BAUDS[args.baud]
    termios.tcsetattr(fd, termios.TCSANOW, t)

    def drain(seconds: float) -> bytes:
        buf, end = b"", time.time() + seconds
        while time.time() < end:
            try:
                c = os.read(fd, 4096)
                if c:
                    buf += c
            except BlockingIOError:
                pass
            time.sleep(0.005)
        return buf

    drain(0.4)
    table: dict[tuple[int, str], dict] = {}
    for ant in ants:
        os.write(fd, frame(0x1B, bytes([0x01, ant])))
        drain(0.4)
        for _ in range(args.rounds):
            os.write(fd, frame(0x22))
            for kind, cmd, params in parse(drain(0.45)):
                if kind != 0x02 or len(params) not in (17, 18):
                    continue
                epc = params[3:15].hex().upper()
                rssi = params[0] - 256 if params[0] > 127 else params[0]
                rec = table.setdefault((ant, epc), {"n": 0, "rssi": []})
                rec["n"] += 1
                rec["rssi"].append(rssi)
        print("  antenna %d done" % ant)

    os.close(fd)
    print()
    print("  %-8s %-36s %7s %9s" % ("antenna", "EPC", "reads", "best RSSI"))
    tags = sorted({epc for _, epc in table})
    for ant in ants:
        for epc in tags:
            rec = table.get((ant, epc))
            if not rec:
                continue
            print("  %-8d %-36s %4d/%-3d %7d dBm"
                  % (ant, epc, rec["n"], args.rounds, max(rec["rssi"])))
    print()
    print("  tags seen by more than one antenna (leakage):")
    leaked = 0
    for epc in tags:
        ants_seen = [a for a in ants if (a, epc) in table]
        if len(ants_seen) > 1:
            leaked += 1
            best = {a: max(table[(a, epc)]["rssi"]) for a in ants_seen}
            spread = max(best.values()) - min(best.values())
            print("    %s seen by %s, best %s, spread %d dB"
                  % (epc, ants_seen, best, spread))
    if not leaked:
        print("    none - each tag answered on exactly one antenna")
    print()
    print("  A spread of 10 dB or more means the intended antenna is clearly the closest; a tag")
    print("  heard on two antennas at nearly equal strength cannot be assigned to a bay by antenna.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
