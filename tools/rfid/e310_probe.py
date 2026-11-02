#!/usr/bin/env python3
"""Talk to the 4D four-antenna UHF module over a USB-TTL adapter.

Verified against the vendor's protocol manual (V4.1.3, "超高频读写模块多天线系列通讯配置协议"):

    BB | Type | Command | PL(MSB) PL(LSB) | Parameters... | Checksum | 7E

  Type      0x00 command, 0x01 response, 0x02 notification
  PL        two bytes, most significant first
  Checksum  low byte of the sum from Type through the last parameter

Two frames prove the link:

    BB 00 03 00 01 00 04 7E     get hardware version  (param 00=hw, 01=sw, 02=manufacturer)
    BB 00 22 00 00 22 7E        single inventory poll (opens the PA, polls once, closes it)

A tag answer arrives as a notification frame:

    BB 02 22 00 11 <RSSI> <PC hi lo> <EPC 12> <CRC 2> <sum> 7E     PL = 0x11 = 17 bytes

and "no tag in the field" answers as

    BB 01 FF 00 01 15 16 7E     error code 0x15

    python tools/rfid/e310_probe.py --port /dev/cu.SLAB_USBtoUART
    python tools/rfid/e310_probe.py --port ... --polls 20
    python tools/rfid/e310_probe.py --port ... --baud 9600     # if 115200 is silent

Before powering on: attach an antenna. Transmitting into an open connector is how these modules get
damaged. The module wants 5 V at up to ~300 mA when the PA is on; a USB-TTL's 5 V pin is usually
marginal, and "talks fine but resets when it transmits" is the usual symptom.
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


GET_HW = frame(0x03, b"\x00")
GET_SW = frame(0x03, b"\x01")
SINGLE_POLL = frame(0x22)


def open_port(path: str, baud: int):
    fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    tty.setraw(fd)
    a = termios.tcgetattr(fd)
    a[4] = a[5] = BAUDS[baud]
    termios.tcsetattr(fd, termios.TCSANOW, a)
    return fd


def read_for(fd, seconds: float) -> bytes:
    buf = b""
    end = time.time() + seconds
    while time.time() < end:
        try:
            chunk = os.read(fd, 4096)
            if chunk:
                buf += chunk
        except BlockingIOError:
            pass
        except OSError:
            break
        time.sleep(0.02)
    return buf


def parse(data: bytes, verbose: bool = False) -> list[dict]:
    """Split a byte stream into frames. Two-byte PL, so a wrong guess would desynchronise."""
    out, i = [], 0
    while i < len(data):
        if data[i] != 0xBB:
            i += 1
            continue
        if i + 5 > len(data):
            break
        kind, cmd = data[i + 1], data[i + 2]
        pl = (data[i + 3] << 8) | data[i + 4]
        end = i + 5 + pl + 1 + 1                 # params + checksum + 7E
        if end > len(data) or data[end - 1] != 0x7E:
            if verbose:
                print("    incomplete frame at %d (pl=%d, need %d bytes, have %d)"
                      % (i, pl, end - i, len(data) - i))
            i += 1
            continue
        params = data[i + 5:i + 5 + pl]
        checksum = data[i + 5 + pl]
        calc = sum(data[i + 1:i + 5 + pl]) & 0xFF
        out.append({"type": kind, "cmd": cmd, "pl": pl, "params": params,
                    "crc_ok": checksum == calc, "got": checksum, "want": calc})
        i = end
    return out


def show(frames: list[dict]) -> None:
    for f in frames:
        tag = {0x00: "command", 0x01: "response", 0x02: "notification"}.get(f["type"], "?")
        crc = "CRC ok" if f["crc_ok"] else "CRC BAD (got %02X want %02X)" % (f["got"], f["want"])
        print("    %-12s cmd=0x%02X pl=%-3d %s" % (tag, f["cmd"], f["pl"], crc))
        p = f["params"]
        if f["type"] == 0x01:
            if f["cmd"] == 0xFF:
                code = p[0] if p else 0
                what = {0x15: "no tag in the field / tag CRC error",
                        0x16: "no antenna", 0x17: "access denied"}.get(code, "")
                print("      error code 0x%02X  %s" % (code, what))
            elif f["cmd"] == 0x03:
                print("      version: " + " ".join("%02X" % b for b in p))
                if len(p) >= 2:
                    print("      readable: %d.%d" % (p[0], p[1]))
            elif f["cmd"] == 0x22:
                print("      poll acknowledged")
            else:
                print("      params: " + " ".join("%02X" % b for b in p[:16]))
        elif f["type"] == 0x02:
            # 17 bytes = RSSI + PC + EPC12 + CRC2 (no antenna field). 18 bytes means the
            # four-antenna firmware appended the antenna number - measured on the bench, the
            # last byte is 0x00 and does not change while only antenna port 1 is connected, and
            # the vendor datasheet says "标签数据携带天线号上报". So the extra byte is the antenna.
            if f["pl"] in (17, 18) and len(p) >= f["pl"]:
                rssi = p[0] - 256 if p[0] > 127 else p[0]
                pc = (p[1] << 8) | p[2]
                epc = p[3:15]
                ant = p[17] if f["pl"] == 18 else None
                print("      TAG  RSSI=%3d dBm  PC=0x%04X  EPC=%s%s"
                      % (rssi, pc, epc.hex().upper(),
                         "" if ant is None else "  ANT=%d" % ant))
            else:
                print("      notification pl=%d params: %s"
                      % (f["pl"], " ".join("%02X" % b for b in p[:24])))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=0, help="0 = 115200 then 57600 then 38400")
    ap.add_argument("--polls", type=int, default=5)
    ap.add_argument("--wait", type=float, default=0.35, help="seconds to listen after a poll")
    ap.add_argument("--loopback", action="store_true",
                    help="short the adapter's TX to RX, then this proves the adapter itself works")
    args = ap.parse_args()

    if args.loopback:
        baud = args.baud or 115200
        fd = open_port(args.port, baud)
        try:
            probe = b"LOOPBACK-1234567890"
            os.write(fd, probe)
            got = read_for(fd, 1.0)
            print("== loopback @ %d ==" % baud)
            print("   sent: %s" % probe.decode())
            print("   back: %r" % got[:40])
            if probe[:6] in got:
                print("   ✓ adapter and cable carry data - the silence is on the module side")
                return 0
            print("   ✗ nothing came back: the adapter, the cable, or the TX/RX short is at fault")
            return 1
        finally:
            os.close(fd)

    for baud in ([args.baud] if args.baud else [115200, 57600, 38400, 19200, 9600]):
        print("== %s @ %d 8N1 ==" % (args.port, baud))
        fd = open_port(args.port, baud)
        try:
            os.write(fd, GET_HW)
            got = read_for(fd, 0.8)
            frames = parse(got, verbose=True)
            if not frames:
                print("  no reply to get-hardware-version (%d bytes back)" % len(got))
                if got:
                    print("  raw: " + got[:32].hex().upper())
                continue
            show(frames)
            print("  == linked at %d baud ==" % baud)
            os.write(fd, GET_SW)
            show(parse(read_for(fd, 0.5)))
            print()
            print("  polling %d times - put a tag near the antenna" % args.polls)
            tags = 0
            for n in range(args.polls):
                os.write(fd, SINGLE_POLL)
                fr = parse(read_for(fd, args.wait))
                for f in fr:
                    if f["type"] == 0x02 and f["pl"] in (17, 18):
                        tags += 1
                if fr:
                    print("  poll %d:" % (n + 1))
                    show(fr)
            print()
            print("  %d tag reads in %d polls" % (tags, args.polls))
            return 0 if tags else 0
        finally:
            os.close(fd)
    print()
    print("nothing answered. Check in this order:")
    print("  1. adapter TX -> module RX, adapter RX -> module TX, common ground")
    print("  2. antenna attached before transmitting")
    print("  3. 5 V supply able to give ~300 mA (the PA current) - many USB-TTL pins cannot")
    print("  4. baud: --baud 9600 / 19200 / 38400")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
