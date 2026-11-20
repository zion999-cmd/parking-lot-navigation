"""P0005: the magnetometer judge must be UNKNOWN rather than wrong.

The states these tests pin down are the ones that decide whether the experiment means anything:
a bay with no baseline, no mapped channel, a dead sensor or stale data must never be reported
OCCUPIED, and the read counter must not be mistaken for new measurements.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import mag_baseline as M                                            # noqa: E402


class FakePoller:
    """Stands in for MagPoller: hands out a fixed frame, optionally changing over time."""

    def __init__(self, readings, scales=None):
        self.readings = readings
        self.scales = scales or {}
        self.frames = []

    def push(self, readings):
        self.frames.append(readings)

    def snapshot(self, now=None):
        now = time.time() if now is None else now
        frame = {}
        for channel, reading in self.readings.items():
            entry = dict(reading)
            entry.setdefault("t", now)
            entry.setdefault("ok", True)
            entry["age"] = now - entry["t"]
            frame[channel] = entry
        frame["_meta"] = {"error": None, "polls": 1, "duplicates": 0, "scales": self.scales}
        return frame


def make_baseline(threshold=3.8, hysteresis=0.6, **bays):
    entries = {}
    for bay, spec in bays.items():
        entries[bay] = spec
    return M.MagBaseline(bays=entries, calibrated_at=time.time(), threshold_ut=threshold,
                         hysteresis_ut=hysteresis)


def valid_entry(channel, x=0.0, y=0.0, z=0.0, scale=0.1):
    return {"channel": channel, "valid": True, "ut_per_lsb": scale, "samples": 40,
            "mean_xyz_ut": {"x": x, "y": y, "z": z},
            "noise_ut": {"x": 0.1, "y": 0.1, "z": 0.1}}


def reading(x, y, z, reads=1, age=0.0, ok=True, t=None):
    when = (time.time() - age) if t is None else t
    return {"x": x, "y": y, "z": z, "reads": reads, "t": when, "ok": ok}


def test_no_baseline_is_unknown():
    judge = M.MagJudge(None)
    states = judge.update(FakePoller({}).snapshot())
    assert set(states) == {"B1", "B2", "B3", "B4"}
    assert all(entry["state"] == "UNKNOWN" for entry in states.values())
    assert all("no baseline" in entry["reason"] for entry in states.values())


def test_unmapped_bay_is_unknown_not_occupied():
    baseline = make_baseline(B1={"channel": None, "valid": False,
                                 "reason": "no channel mapped for this bay"})
    judge = M.MagJudge(baseline)
    states = judge.update(FakePoller({0: reading(999, 999, 999)}).snapshot())
    assert states["B1"]["state"] == "UNKNOWN"
    assert "no channel mapped" in states["B1"]["reason"]


def test_invalid_baseline_reason_is_reported():
    baseline = make_baseline(B1={"channel": 1, "valid": False, "reason": "only 3 fresh samples"})
    judge = M.MagJudge(baseline)
    states = judge.update(FakePoller({1: reading(0, 0, 0)}).snapshot())
    assert states["B1"]["state"] == "UNKNOWN"
    assert "3 fresh samples" in states["B1"]["reason"]


def test_stale_reading_is_unknown():
    baseline = make_baseline(B1=valid_entry(1))
    judge = M.MagJudge(baseline, stale_s=2.0)
    states = judge.update(FakePoller({1: reading(0, 0, 0, age=30.0)}).snapshot())
    assert states["B1"]["state"] == "UNKNOWN"
    assert "stale" in states["B1"]["reason"]


def test_unhealthy_sensor_is_unknown():
    baseline = make_baseline(B1=valid_entry(1))
    judge = M.MagJudge(baseline)
    states = judge.update(FakePoller({1: reading(0, 0, 0, ok=False)}).snapshot())
    assert states["B1"]["state"] == "UNKNOWN"
    assert "not ok" in states["B1"]["reason"]


def test_missing_channel_is_unknown():
    baseline = make_baseline(B1=valid_entry(1))
    judge = M.MagJudge(baseline)
    states = judge.update(FakePoller({0: reading(0, 0, 0)}).snapshot())
    assert states["B1"]["state"] == "UNKNOWN"
    assert "not reported" in states["B1"]["reason"]


def test_empty_field_is_empty():
    baseline = make_baseline(B1=valid_entry(1, x=10.0, y=2.0, z=3.0))
    judge = M.MagJudge(baseline, window_s=5.0)
    now = time.time()
    frame = FakePoller({1: reading(100, 20, 30, reads=1, t=now)}).snapshot(now=now)
    judge.update(frame, now=now)                           # first point fills the window
    frame = FakePoller({1: reading(101, 20, 30, reads=2, t=now + 0.1)}).snapshot(now=now + 0.1)
    states = judge.update(frame, now=now + 0.1)
    assert states["B1"]["state"] == "EMPTY"
    assert states["B1"]["delta_ut"] < 3.8


def test_occupied_then_hysteresis_keeps_it_until_well_below():
    """A step above the threshold latches, and only clears below threshold - hysteresis."""
    baseline = make_baseline(B1=valid_entry(1, x=0.0), threshold=3.8, hysteresis=0.6)
    judge = M.MagJudge(baseline, window_s=5.0)

    clock = [time.time()]

    def run(values):
        last = None
        for index, value in enumerate(values):
            clock[0] += 0.1
            frame = FakePoller({1: reading(value, 0, 0, reads=index + 1,
                                           t=clock[0])}).snapshot(now=clock[0])
            last = judge.update(frame, now=clock[0])
        return last["B1"]

    # 6 uT above baseline: occupied.
    state = run([60, 60])
    clock[0] += 10.0                                       # let the old window drain
    assert state["state"] == "OCCUPIED"
    # 3.5 uT is below the threshold but above threshold - hysteresis: still occupied.
    state = run([35, 35])
    clock[0] += 10.0
    assert state["state"] == "OCCUPIED"
    # 2.0 uT is below threshold - hysteresis: clears.
    state = run([20, 20])
    assert state["state"] == "EMPTY"


def test_delta_is_vector_distance_not_per_axis():
    baseline = make_baseline(B1=valid_entry(1, x=0.0, y=0.0, z=0.0))
    judge = M.MagJudge(baseline, window_s=5.0)
    now = time.time()
    judge.update(FakePoller({1: reading(20, 20, 20, reads=1, t=now)}).snapshot(now=now), now=now)
    states = judge.update(FakePoller({1: reading(20, 20, 20, reads=2, t=now + 0.1)})
                          .snapshot(now=now + 0.1), now=now + 0.1)
    # 20 counts on every axis at 0.1 uT/count = 2 uT per axis, 3.464 uT in length. Per-axis
    # comparison would have said 2.0, which is why the judge uses the vector length.
    assert abs(states["B1"]["delta_ut"] - (2.0 ** 2 * 3) ** 0.5) < 0.01


def test_repeated_read_counter_is_not_new_data():
    """Two polls returning the same counter must count as one sample, not two."""
    poller = FakePoller({1: reading(100, 0, 0, reads=7)}, scales={1: 0.1})
    baseline = M.MagBaseline.capture(poller, {"B1": 1}, duration=0.6, min_samples=2)
    assert baseline.bays["B1"]["valid"] is False
    assert "fresh samples" in baseline.bays["B1"]["reason"]


def test_dead_sensor_cannot_become_a_baseline():
    """A chip that ACKs but never changes its output must not be trusted as an empty reference."""
    class CountingPoller(FakePoller):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.counter = 0

        def snapshot(self, now=None):
            self.counter += 1
            self.readings[1]["reads"] = self.counter
            return super().snapshot(now)

    poller = CountingPoller({1: reading(5, 5, 5, reads=0)}, scales={1: 0.1})
    baseline = M.MagBaseline.capture(poller, {"B1": 1}, duration=0.6, min_samples=2)
    assert baseline.bays["B1"]["valid"] is False
    assert "never changed" in baseline.bays["B1"]["reason"]


def test_baseline_round_trip(tmp_path=None):
    import json
    import tempfile

    baseline = make_baseline(B1=valid_entry(1, x=1.5, y=2.5, z=3.5))
    baseline.source = "http://example/stream"
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "mag_baseline.json")
        baseline.save(path)
        loaded = M.MagBaseline.load(path)
        assert loaded is not None
        assert loaded.threshold_ut == baseline.threshold_ut
        assert loaded.bays["B1"]["mean_xyz_ut"] == baseline.bays["B1"]["mean_xyz_ut"]
        assert json.load(open(path, encoding="utf-8"))["schema"] == "mag_baseline/1"


def test_missing_baseline_file_loads_as_none():
    assert M.MagBaseline.load("/nonexistent/mag_baseline.json") is None
