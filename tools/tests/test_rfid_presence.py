"""Presence logic: a quiet round is not a departure, and a dead reader is not an empty sandbox.

These run against synthetic /api/rfid documents - the physical part (put a tag in, take it out) has
to be done by hand on the sandbox and is not covered here.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rfid"))

import rfid_presence as R                                          # noqa: E402


def doc(frames=100, tags=None, ok=True, polls=10):
    return {
        "ok": ok,
        "polls": polls,
        "frames": frames,
        "notifications": 0,
        "timeouts": 0,
        "count": len(tags or []),
        "tags": tags or [],
        "ports": [{"port": p, "select_ok": frames, "select_fail": 0,
                   "inventory_ok": frames, "inventory_timeout": 0, "tags": 0} for p in (1, 2, 3, 4)],
    }


def tag(epc, port=2, rssi=-60, reads=5):
    return {"epc": epc, "port": port, "rssi": rssi, "reads": reads, "age_ms": 100}


def test_tag_appears_present():
    presence = R.Presence(absent_after=3.0)
    presence.update(doc(frames=100, tags=[tag("E280A")]), now=1000.0)
    assert presence.state_of("E280A", now=1000.5) == "PRESENT"


def test_one_quiet_round_is_not_a_departure():
    """The reader reports a tag only on rounds it happens to read it; a gap of a round or two is normal."""
    presence = R.Presence(absent_after=3.0)
    presence.update(doc(frames=100, tags=[tag("E280A")]), now=1000.0)
    presence.update(doc(frames=101, tags=[]), now=1000.5)
    presence.update(doc(frames=102, tags=[]), now=1001.0)
    assert presence.state_of("E280A", now=1001.0) == "PRESENT"


def test_silence_longer_than_the_window_is_absent():
    presence = R.Presence(absent_after=3.0)
    presence.update(doc(frames=100, tags=[tag("E280A")]), now=1000.0)
    presence.update(doc(frames=110, tags=[]), now=1002.0)
    assert presence.state_of("E280A", now=1002.0) == "PRESENT"
    presence.update(doc(frames=120, tags=[]), now=1004.0)
    assert presence.state_of("E280A", now=1004.0) == "ABSENT"


def test_reader_silent_is_unknown_and_tags_are_not_aged_out():
    presence = R.Presence(absent_after=3.0, stall_after=2.5)
    presence.update(doc(frames=100, tags=[tag("E280A")]), now=1000.0)
    presence.update(None, now=1001.0)                     # endpoint stopped answering
    assert presence.reader_ok is False
    assert presence.state_of("E280A", now=1001.0) == "UNKNOWN"
    assert presence.state_of("E280A", now=1099.0) == "UNKNOWN"


def test_reader_ok_but_frames_frozen_is_unknown():
    """ok=true with a stalled frame counter is exactly what a stuck reader looks like."""
    presence = R.Presence(absent_after=3.0, stall_after=2.5)
    presence.update(doc(frames=100, tags=[]), now=1000.0)
    assert presence.reader_ok is True
    presence.update(doc(frames=100, tags=[]), now=1004.0)
    assert presence.reader_ok is False
    assert "power-cycle" in presence.health_note


def test_reader_recovers_after_frames_move_again():
    presence = R.Presence(absent_after=3.0, stall_after=2.5)
    presence.update(doc(frames=100), now=1000.0)
    presence.update(doc(frames=100), now=1004.0)
    assert presence.reader_ok is False
    presence.update(doc(frames=101), now=1004.5)
    assert presence.reader_ok is True


def test_two_epcs_are_tracked_separately():
    presence = R.Presence(absent_after=3.0)
    presence.update(doc(frames=100, tags=[tag("E280A", port=1), tag("E280B", port=3)]), now=1000.0)
    presence.update(doc(frames=110, tags=[tag("E280B", port=3)]), now=1004.0)
    assert presence.state_of("E280A", now=1004.0) == "ABSENT"
    assert presence.state_of("E280B", now=1004.0) == "PRESENT"
    rows = {row["epc"]: row for row in presence.rows(now=1004.0)}
    assert rows["E280B"]["port"] == 3


def test_rows_report_last_port_and_rssi():
    presence = R.Presence(absent_after=3.0)
    presence.update(doc(frames=100, tags=[tag("E280A", port=4, rssi=-47, reads=9)]), now=1000.0)
    row = presence.rows(now=1000.2)[0]
    assert row["port"] == 4 and row["rssi"] == -47 and row["reads"] == 9
    assert row["state"] == "PRESENT"


def test_port_deltas_are_differences_not_totals():
    first = doc(frames=100)
    deltas, previous = R.port_deltas(first, None)
    assert [d["select_ok"] for d in deltas] == [100, 100, 100, 100]
    second = doc(frames=101)
    for port in second["ports"]:
        port["select_ok"] = 101
        port["inventory_ok"] = 101
        port["select_fail"] = 5
    deltas, _ = R.port_deltas(second, previous)
    assert [d["select_ok"] for d in deltas] == [1, 1, 1, 1]
    assert [d["select_fail"] for d in deltas] == [5, 5, 5, 5]
