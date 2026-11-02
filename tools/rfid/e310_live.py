#!/usr/bin/env python3
"""Live tag view for the 4D four-antenna UHF module.

Polls once every 200 ms and prints what is in the field right now, with a running table of every
EPC seen. Press q to stop.

    python tools/rfid/e310_live.py --port /dev/cu.usbserial-14330
    python tools/rfid/e310_live.py --port ... --interval 0.5

Frame format (vendor protocol manual V4.1.3):

    BB | Type | Command | PL(MSB) PL(LSB) | Parameters | Checksum | 7E

    tag notification   BB 02 22 00 12 <RSSI> <PC hi lo> <EPC 12> <CRC 2> <ANT> <sum> 7E
                       PL = 18 because the four-antenna firmware appends the antenna number;
                       a single-antenna module answers with PL = 17 and no ANT byte.

    no tag             BB 01 FF 00 01 15 16 7E        (error 0x15)
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import time
import tty

BAUDS = {9600: termios.B9600, 19200: termios.B19200, 38400: termios.B38400,
         57600: termios.B57600, 115200: termios.B115200}


def frame(cmd: int, params: bytes = b"") -> bytes:
    pl = len(params)
    body = bytes([0x00, cmd, (pl >> 8) & 0xFF, pl & 0xFF]) + params
    return b"\xbb" + body + bytes([sum(body) & 0xFF]) + b"\x7e"


GET_HW = frame(0x03, b"\x00")
GET_SW = frame(0x03, b"\x01")
SINGLE_POLL = frame(0x22)

# Which antenna the next single poll uses. From the vendor's four-port quick reference:
#   BB 00 1B 00 02 01 <n> CRC 7E     - antenna n (1..4) for tag operations and single polls
#   BB 00 1B 00 05 02 <a1><a2><a3><a4> CRC 7E - which antennas take part in continuous inventory
# Without one of these the module answers on antenna 1 only, which is what "only one antenna
# works" looks like from the outside.
def select_antenna(n: int) -> bytes:
    return frame(0x1B, bytes([0x01, n]))


def work_mode(ants: bytes) -> bytes:
    return frame(0x1B, bytes([0x02]) + ants)


def open_port(path: str, baud: int):
    fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    tty.setraw(fd)
    a = termios.tcgetattr(fd)
    a[4] = a[5] = BAUDS[baud]
    termios.tcsetattr(fd, termios.TCSANOW, a)
    return fd


def drain(fd, seconds: float) -> bytes:
    buf, end = b"", time.time() + seconds
    while time.time() < end:
        try:
            chunk = os.read(fd, 4096)
            if chunk:
                buf += chunk
        except BlockingIOError:
            pass
        except OSError:
            break
        time.sleep(0.005)
    return buf


def parse(data: bytes) -> list[dict]:
    out, i = [], 0
    while i < len(data):
        if data[i] != 0xBB:
            i += 1
            continue
        if i + 5 > len(data):
            break
        kind, cmd = data[i + 1], data[i + 2]
        pl = (data[i + 3] << 8) | data[i + 4]
        end = i + 5 + pl + 2                       # params + checksum + 7E
        if end > len(data) or data[end - 1] != 0x7E:
            i += 1
            continue
        params = data[i + 5:i + 5 + pl]
        ok = (sum(data[i + 1:i + 5 + pl]) & 0xFF) == data[i + 5 + pl]
        out.append({"type": kind, "cmd": cmd, "pl": pl, "params": params, "crc_ok": ok})
        i = end
    return out


def tags_in(frames: list[dict]) -> list[dict]:
    tags = []
    for f in frames:
        if f["type"] != 0x02 or f["pl"] not in (17, 18):
            continue
        p = f["params"]
        tags.append({
            "rssi": p[0] - 256 if p[0] > 127 else p[0],
            "pc": (p[1] << 8) | p[2],
            "epc": p[3:15].hex().upper(),
            "ant": p[17] if f["pl"] == 18 else None,
            "crc_ok": f["crc_ok"],
        })
    return tags


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--interval", type=float, default=0.2, help="seconds between polls")
    ap.add_argument("--window", type=float, default=0.16, help="seconds to listen per poll")
    ap.add_argument("--seconds", type=float, default=0, help="0 = until q")
    ap.add_argument("--antenna", type=int, default=1, choices=[1, 2, 3, 4],
                    help="antenna port to poll on (see the ANT column to confirm)")
    ap.add_argument("--csv", help="append every tag read to this file (time,ant,epc,rssi)")
    ap.add_argument("--cycle", action="store_true",
                    help="rotate through antennas 1..4, one poll each - the ANT column says which "
                         "port the answer came from")
    args = ap.parse_args()

    fd = open_port(args.port, args.baud)
    os.write(fd, GET_HW)
    hw = drain(fd, 0.4)
    os.write(fd, GET_SW)
    sw = drain(fd, 0.3)
    def ver(frames):
        for f in frames:
            if f["type"] == 0x01 and f["cmd"] == 0x03:
                return bytes(f["params"][1:]).decode("ascii", "replace").strip("\x00")
        return "?"
    print("module: hw %s   sw %s" % (ver(parse(hw)), ver(parse(sw))))
    if args.cycle:
        print("cycling antennas 1..4, logging which port answers")
    else:
        os.write(fd, select_antenna(args.antenna))
        ack = parse(drain(fd, 0.25))
        ok = any(f["type"] == 0x01 for f in ack)
        print("select antenna %d -> %s" % (args.antenna, "acknowledged" if ok else "no reply"))
    print("polling every %.0f ms - press q to stop" % (args.interval * 1000))
    print()
    print("  %-8s %6s %-36s %s" % ("time", "ant", "EPC", "RSSI"))

    if args.csv:
        new = not os.path.exists(args.csv)
        csv_fh = open(args.csv, "a", encoding="utf-8")
        if new:
            csv_fh.write("time,ant,epc,rssi\n")
    else:
        csv_fh = None

    seen: dict[str, dict] = {}
    polls = 0
    hits = 0
    quiet = 0

    ant_try = 0
    interactive = sys.stdin.isatty()
    saved = termios.tcgetattr(sys.stdin) if interactive else None
    if interactive:
        tty.setcbreak(sys.stdin)

    t_end = time.time() + args.seconds if args.seconds else None
    try:
        while True:
            if interactive and select.select([sys.stdin], [], [], 0)[0]:
                if sys.stdin.read(1).lower() == "q":
                    break
            if t_end and time.time() > t_end:
                break

            if args.cycle:
                ant_try = (ant_try % 4) + 1
                os.write(fd, select_antenna(ant_try))
                drain(fd, 0.12)
            os.write(fd, SINGLE_POLL)
            tags = tags_in(parse(drain(fd, args.window)))
            polls += 1
            now = time.strftime("%H:%M:%S")
            if tags:
                hits += 1
                quiet = 0
                for t in tags:
                    seen.setdefault(t["epc"], {"n": 0, "ant": set(), "rssi": []})
                    rec = seen[t["epc"]]
                    rec["n"] += 1
                    rec["rssi"].append(t["rssi"])
                    if t["ant"] is not None:
                        rec["ant"].add(t["ant"])
                    if csv_fh:
                        csv_fh.write("%s,%s,%s,%d\n" % (now, t["ant"], t["epc"], t["rssi"]))
                        csv_fh.flush()
                    print("  %-8s %6s %-36s %4d dBm%s%s"
                          % (now, "-" if t["ant"] is None else t["ant"], t["epc"], t["rssi"],
                             "" if t["crc_ok"] else "  (CRC BAD)",
                             ("  (polled ant %d)" % ant_try) if args.cycle else ""))
            else:
                quiet += 1
                if quiet == 1 or quiet % 25 == 0:
                    print("  %-8s %6s %-36s %s" % (now, "", "(no tag)", "-"))
            time.sleep(max(0.0, args.interval - args.window))
    except KeyboardInterrupt:
        pass
    finally:
        if interactive:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, saved)
        if csv_fh:
            csv_fh.close()
        os.close(fd)

    print()
    print("polls %d, polls with at least one tag %d (%.0f%%)" % (polls, hits,
                                                                100.0 * hits / max(1, polls)))
    if seen:
        print()
        print("  %-36s %5s %6s %-10s %s" % ("EPC", "reads", "best", "antennas", "RSSI range"))
        for epc, rec in sorted(seen.items(), key=lambda kv: -kv[1]["n"]):
            print("  %-36s %5d %4d dBm %-10s %d..%d"
                  % (epc, rec["n"], max(rec["rssi"]),
                     ",".join(str(a) for a in sorted(rec["ant"])) or "-",
                     min(rec["rssi"]), max(rec["rssi"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
