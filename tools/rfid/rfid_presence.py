#!/usr/bin/env python3
"""RFID presence on the sandbox: which tags are here now, from /api/rfid.

Deliberately the smallest thing that answers the question. The board already polls all its
antennas and reports EPC, RSSI, port, read count and age; this keeps a per-EPC last-seen table
and decides PRESENT or ABSENT. No position, no RSSI filtering, no port-to-bay mapping.

Two things it refuses to get wrong:

* One empty inventory is not "the tag left". The reader reports tags only when it happens to read
  them, so a single quiet round is normal. A tag becomes ABSENT only after it has been silent for
  longer than the configured window.
* A reader that stops answering is UNKNOWN, not "everything left". If /api/rfid stops replying, or
  ok turns false, or its frame counters stop moving, the screen says so and no tag is aged out.

    tools/rfid-test.sh                          # uses the board found by the launcher
    tools/rfid/rfid_presence.py --host 192.168.3.118
    tools/rfid/rfid_presence.py --host 192.168.3.118 --absent-after 3 --once
"""

import argparse
import json
import os
import sys
import time
import urllib.request

# A tag is still PRESENT for this long after its last successful read. Set from the reader's own
# polling behaviour rather than a guess: the firmware runs a 200 ms round-robin over the ports and
# reports the tag on every round it reads it, so 3 s is about fifteen missed rounds. Treat this as
# a starting value to be checked against a stationary tag, not a measured optimum.
DEFAULT_ABSENT_AFTER = 3.0
# The reader is unhealthy if its counters stop moving for this long, whatever ok says.
DEFAULT_STALL_AFTER = 2.5
DEFAULT_INTERVAL = 0.4


def fetch(host, timeout=2.0):
    try:
        with urllib.request.urlopen(f"http://{host}/api/rfid", timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:                                        # noqa: BLE001
        return None


class Presence:
    """EPC -> last_seen table, plus the reader-health gate that decides whether to trust it."""

    def __init__(self, absent_after=DEFAULT_ABSENT_AFTER, stall_after=DEFAULT_STALL_AFTER):
        self.absent_after = absent_after
        self.stall_after = stall_after
        self.tags = {}                 # epc -> {last_seen, reads, port, rssi}
        self.last_progress = None      # last time the reader's frame counter moved
        self.last_frames = None
        self.last_polls = None
        self.reader_ok = True
        self.health_note = ""
        self.previous_ports = None

    def update(self, document, now=None):
        now = time.time() if now is None else now
        if document is None:
            self.reader_ok = False
            self.health_note = "/api/rfid did not answer"
            return
        frames = document.get("frames")
        # ok alone is not enough: the endpoint reports ok=true while the reader answers nothing at
        # all, which is exactly the state a stuck multi-inventory leaves it in.
        moved = frames is not None and frames != self.last_frames
        if moved:
            self.last_progress = now
        self.last_frames = frames
        self.last_polls = document.get("polls")
        if not document.get("ok", False):
            self.reader_ok = False
            self.health_note = "board reports ok=false"
        elif self.last_progress is None:
            self.reader_ok = False
            self.health_note = "waiting for the first frame from the reader"
        elif now - self.last_progress > self.stall_after:
            self.reader_ok = False
            self.health_note = f"no frames for {now - self.last_progress:.1f} s - power-cycle the reader"
        else:
            self.reader_ok = True
            self.health_note = ""

        for reading in document.get("tags") or []:
            epc = (reading.get("epc") or "").upper()
            if not epc:
                continue
            entry = self.tags.setdefault(epc, {"first_seen": now})
            entry["last_seen"] = now
            entry["reads"] = reading.get("reads", entry.get("reads", 0))
            entry["port"] = reading.get("port")
            entry["rssi"] = reading.get("rssi")

    def state_of(self, epc, now=None):
        now = time.time() if now is None else now
        if not self.reader_ok:
            return "UNKNOWN"
        silent = now - self.tags[epc]["last_seen"]
        return "PRESENT" if silent <= self.absent_after else "ABSENT"

    def rows(self, now=None):
        now = time.time() if now is None else now
        out = []
        for epc, entry in sorted(self.tags.items(), key=lambda kv: kv[1]["last_seen"], reverse=True):
            out.append({
                "epc": epc,
                "state": self.state_of(epc, now),
                "silent": now - entry["last_seen"],
                "reads": entry.get("reads"),
                "port": entry.get("port"),
                "rssi": entry.get("rssi"),
            })
        return out


def port_deltas(document, previous):
    """Per-port counter changes since the previous sample, so a rate is visible, not just a total."""
    if document is None:
        return [], previous
    current = {p["port"]: p for p in document.get("ports") or []}
    deltas = []
    for port, now in sorted(current.items()):
        was = (previous or {}).get(port, {})
        deltas.append({
            "port": port,
            "select_ok": now.get("select_ok", 0) - was.get("select_ok", 0),
            "select_fail": now.get("select_fail", 0) - was.get("select_fail", 0),
            "inventory_ok": now.get("inventory_ok", 0) - was.get("inventory_ok", 0),
            "inventory_timeout": now.get("inventory_timeout", 0) - was.get("inventory_timeout", 0),
            "tags": now.get("tags", 0),
        })
    return deltas, current


def render(document, presence, deltas, now, interactive, previous_is_none=False):
    lines = []
    health = "reader ok" if presence.reader_ok else f"reader UNKNOWN - {presence.health_note}"
    tags = presence.rows(now)
    present = sum(1 for row in tags if row["state"] == "PRESENT")
    lines.append(f"{time.strftime('%H:%M:%S')}  {health}")
    if document is not None:
        lines.append(f"  polls={document.get('polls')} frames={document.get('frames')} "
                     f"notifications={document.get('notifications')} "
                     f"timeouts={document.get('timeouts')}  →  {present} present, {len(tags)} known")
    else:
        lines.append(f"  no data from /api/rfid")
    if tags:
        lines.append("  EPC                       state     silent   reads  port  rssi")
        for row in tags:
            lines.append(f"  {row['epc']:<24}  {row['state']:<8} {row['silent']:5.1f}s  "
                         f"{str(row['reads']):>5}  {str(row['port']):>4}  "
                         f"{'' if row['rssi'] is None else str(row['rssi']):>4}")
    else:
        lines.append(f"  no tags yet")
    if deltas:
        # The first sample has no previous, so its "delta" is the running total - say so, because a
        # six-figure select_fail total from an old stuck reader reads like a current fault.
        label = "totals since boot" if previous_is_none else "this sample"
        lines.append(f"  port  select_ok/fail  inventory_ok/timeout   ({label})")
        for port in deltas:
            lines.append(f"  {port['port']:>4}  {port['select_ok']:>5}/{port['select_fail']:<9}"
                         f"  {port['inventory_ok']:>5}/{port['inventory_timeout']:<9}")
    text = "\n".join(lines)
    if interactive:
        # Redraw in place on a terminal; append when piped so the output stays copyable.
        sys.stdout.write("\033[H\033[2J" + text + "\n")
    else:
        print(text)
        print()
    sys.stdout.flush()


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="192.168.3.118", help="board address")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    parser.add_argument("--absent-after", type=float, default=DEFAULT_ABSENT_AFTER,
                        help=f"seconds of silence before a tag is ABSENT (default {DEFAULT_ABSENT_AFTER:g})")
    parser.add_argument("--stall-after", type=float, default=DEFAULT_STALL_AFTER,
                        help="seconds without a new reader frame before the reader is UNKNOWN")
    parser.add_argument("--once", action="store_true", help="print one sample and exit")
    parser.add_argument("--duration", type=float, default=None, help="stop after this many seconds")
    args = parser.parse_args(argv)

    presence = Presence(args.absent_after, args.stall_after)
    previous = None
    started = time.time()
    interactive = sys.stdout.isatty() and not args.once

    while True:
        now = time.time()
        document = fetch(args.host)
        presence.update(document, now)
        deltas, previous = port_deltas(document, previous)
        if args.once:
            render(document, presence, deltas, now, False, previous_is_none=True)
            return 0 if presence.reader_ok else 1
        if interactive:
            render(document, presence, deltas, now, True)
        else:
            render(document, presence, deltas, now, False)
        if args.duration and now - started >= args.duration:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("\nstopped")
