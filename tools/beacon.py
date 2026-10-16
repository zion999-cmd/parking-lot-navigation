#!/usr/bin/env python3
"""Broadcast this machine so the camera node can find it without an address in the firmware.

    python3 tools/beacon.py              # every 2 s, UDP 9999
    python3 tools/beacon.py --every 5    # slower
    python3 tools/beacon.py --to 192.168.3.255   # add a directed broadcast

Why a plain UDP broadcast rather than mDNS or a config file: the sandbox changes
Wi-Fi, the PC's address changes with it, and this has to keep working at the venue
with nothing to configure. A broadcast needs no component in the firmware, no
service registration, and no router settings, and the device learns the address
from the packet's source - the one address that is demonstrably able to reach it.

What the device does with it is in firmware/esp32-s3-cam/main/discovery.c; it
replaces the *host* in its configured URLs and leaves ports and paths alone.

The JSON in the packet is for humans reading the device's log. The device decides
by the source address, deliberately: a field claiming to be the host would be a
second source of truth for the same fact, and the arriving address is the one that
has been proved to work.

Run it next to the vision server and leave it running. If it is not running,
nothing breaks - the device falls back to the addresses compiled into it, which is
exactly how it behaved before this existed.
"""

import argparse
import json
import socket
import sys
import time

MAGIC = "PARKINGLOT/1"

VISION_PORT = 8010
OTA_PORT = 8000


def local_address() -> str:
    """This machine's address on the interface that would carry the traffic.

    A UDP socket connect() sends nothing; it only asks the routing table which
    source address would be used, which is the one the device can reach.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))         # TEST-NET-1, reserved and unroutable
        return s.getsockname()[0]
    except OSError:
        return "0.0.0.0"
    finally:
        s.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Broadcast this PC so the camera node can find it")
    ap.add_argument("--port", type=int, default=9999, help="must match DISCOVERY_PORT")
    ap.add_argument("--every", type=float, default=2.0, help="seconds between beacons")
    ap.add_argument("--to", action="append", default=[], metavar="ADDR",
                    help="extra broadcast address (repeatable); 255.255.255.255 is always sent")
    args = ap.parse_args()

    targets = ["255.255.255.255"] + args.to

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    host = local_address()
    payload = (MAGIC + " " + json.dumps({
        "host": host,                    # for the device's log; the source address decides
        "vision_port": VISION_PORT,
        "ota_port": OTA_PORT,
    }, separators=(",", ":"))).encode()

    print(f"beacon: {host} every {args.every}s")
    print(f"  sending  {', '.join(f'{t}:{args.port}' for t in targets)}")
    print(f"  payload  {payload.decode()}")
    print(f"  device   uses the source address, so the firmware works on any Wi-Fi this PC is on")
    print("ctrl-c to stop")

    n = 0
    try:
        while True:
            for target in targets:
                try:
                    sock.sendto(payload, (target, args.port))
                except OSError as exc:
                    print(f"  could not broadcast to {target}: {exc}", file=sys.stderr)
            n += 1
            if n % 30 == 0:                     # one line a minute at the default rate
                print(f"  {n} beacons sent")
            time.sleep(args.every)
    except KeyboardInterrupt:
        print()
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
