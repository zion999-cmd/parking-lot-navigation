"""P0007: each source is reported with its own freshness, and none of them is invented."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import overview                                                  # noqa: E402


def state_doc(timestamp, bays=None):
    spots = bays or {"B1": {"state": "EMPTY", "changed_ratio": 0.01},
                     "B2": {"state": "OCCUPIED", "changed_ratio": 0.58},
                     "9": {"state": "EMPTY", "changed_ratio": 0.0}}
    return {"schema": "state/1", "timestamp": timestamp, "spots": spots, "vehicle": {}}


class FakePresence:
    def __init__(self, rows, ok=True, note=""):
        self._rows = rows
        self.reader_ok = ok
        self.health_note = note

    def rows(self, now=None):
        return self._rows


def row(epc, state="PRESENT", silent=0.2, port=2, rssi=-60, reads=5):
    return {"epc": epc, "state": state, "silent": silent, "port": port, "rssi": rssi,
            "reads": reads}


def test_vision_bays_are_reported_and_staleness_is_measured():
    doc = overview.build(state_doc(1000), None, {}, None, None, None, now=1002.0)
    assert doc["vision"]["bays"]["B1"]["state"] == "EMPTY"
    assert doc["vision"]["bays"]["B2"]["state"] == "OCCUPIED"
    assert doc["vision"]["link"] == "ok"
    assert doc["vision"]["age_s"] == 2.0
    # B3/B4 are not in this document: reported as UNKNOWN rather than guessed.
    assert doc["vision"]["bays"]["B3"]["state"] == "UNKNOWN"


def test_old_vision_state_is_stale_not_current():
    doc = overview.build(state_doc(1000), None, {}, None, None, None, now=1000 + 60)
    assert doc["vision"]["link"] == "stale"
    assert doc["all_present"] is False


def test_missing_sources_are_missing_not_empty():
    doc = overview.build(None, None, {}, None, None, None, now=1000.0)
    assert doc["links"] == {"vision": "missing", "mag": "missing", "rfid": "missing"}
    assert doc["rfid"]["count"] == 0
    assert "no answer" in doc["rfid"]["error"]


def test_mag_states_pass_through_unchanged():
    states = {"B1": {"state": "CHANGED", "delta_counts": 800.0, "delta_ut": 8.0, "reason": ""},
              "B2": {"state": "NORMAL", "delta_counts": 12.0, "delta_ut": 1.2, "reason": ""},
              "B3": {"state": "UNKNOWN", "delta_counts": None, "delta_ut": None,
                     "reason": "no channel mapped for this bay"},
              "B4": {"state": "NORMAL", "delta_counts": 5.0, "delta_ut": 0.5, "reason": ""}}
    class Baseline:
        threshold_counts = 70.0
        hysteresis_counts = 10.0
        calibrated_at = 1.0
        source = "x"
        bays = {"B1": {"valid": True}}
    doc = overview.build(None, states, {"B1": 0, "B2": 1, "B3": None, "B4": 4}, Baseline(),
                         None, None, now=1000.0)
    assert doc["mag"]["bays"]["B1"]["state"] == "CHANGED"
    assert doc["mag"]["bays"]["B1"]["delta_counts"] == 800.0
    assert doc["mag"]["bays"]["B3"]["detail"].startswith("no channel mapped")
    assert doc["mag"]["threshold_counts"] == 70.0


def test_rfid_present_count_only_counts_present_tags():
    document = {"ok": True, "polls": 1, "frames": 2, "notifications": 3, "timeouts": 0,
                "tags": [], "ports": []}
    presence = FakePresence([row("E280A", "PRESENT"), row("E280B", "ABSENT", silent=9.0)])
    doc = overview.build(None, None, {}, None, document, presence, now=1000.0)
    assert doc["rfid"]["count"] == 1
    assert doc["rfid"]["known"] == 2
    assert [t["epc"] for t in doc["rfid"]["tags"]] == ["E280A", "E280B"]


def test_unhealthy_reader_is_missing_and_says_why():
    document = {"ok": True, "polls": 1, "frames": 2, "tags": [], "ports": []}
    presence = FakePresence([row("E280A")], ok=False, note="no frames for 5.0 s - power-cycle the reader")
    doc = overview.build(None, None, {}, None, document, presence, now=1000.0)
    assert doc["links"]["rfid"] == "missing"
    assert "power-cycle" in doc["rfid"]["error"]
    assert doc["rfid"]["ok"] is False


def test_no_bay_attribution_for_tags():
    """The document must not carry a bay for a tag - the spec forbids the inference."""
    document = {"ok": True, "polls": 1, "frames": 2, "tags": [], "ports": []}
    presence = FakePresence([row("E280A", port=2)])
    doc = overview.build(None, None, {}, None, document, presence, now=1000.0)
    tag = doc["rfid"]["tags"][0]
    assert "bay" not in tag and "spot" not in tag
    assert tag["port"] == 2
