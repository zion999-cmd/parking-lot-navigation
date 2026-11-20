"""Magnetometer baseline and per-bay occupancy judgement for P0005.

Kept deliberately small: a background poller, a baseline captured over a few seconds, and a
per-bay threshold with hysteresis. No training, no automatic threshold learning, no fusion.

Three things this module refuses to do, because the experiment depends on them:

* It never treats a repeated read counter as a new sample. The board's HTTP interface and its
  sensor loop run at unrelated rates, so several polls legitimately return the same reading.
* It never reports OCCUPIED without a valid baseline, a mapped channel, and fresh, healthy data -
  those cases are UNKNOWN. An unmapped bay is inert, not occupied.
* It never touches the vision result. It only reports what the field looks like.
"""

import json
import math
import os
import statistics
import threading
import time
import urllib.request


# A reading older than this is not evidence of anything any more.
STALE_S = 2.0
# Roll the delta over about a second, which is the timescale a parked car changes on and where the
# measured per-sample noise (1.0-1.5 uT) falls to well under the signal the big bays produce.
WINDOW_S = 1.0
# Measured on this rig with every bay empty, 60 s, 1 s window, LED strip running: delta-B has
# median 1.22 uT, p95 2.29, p99 3.44, max 3.44. The field drifts over tens of seconds rather than
# wobbling per sample, so the empty-state delta is 1-3 uT and not the 0.3 uT the white-noise
# estimate suggested. 1.5 was tried first and reported OCCUPIED constantly; the default is now the
# measured p99 with margin. Re-measure after moving the sandbox.
DEFAULT_THRESHOLD_UT = 3.8
DEFAULT_HYSTERESIS_UT = 0.6
# Below this many fresh samples a baseline is not a baseline.
MIN_SAMPLES = 8


def mag_url(source):
    """http://host:port/stream -> http://host:port/api/mag, or None for a still image."""
    if not isinstance(source, str) or not source.lower().startswith(("http://", "https://")):
        return None
    rest = source.split("://", 1)[1]
    host = rest.split("/", 1)[0]
    if not host:
        return None
    scheme = source.split("://", 1)[0]
    return f"{scheme}://{host}/api/mag"


class MagPoller(threading.Thread):
    """Polls /api/mag in the background and keeps the newest reading per channel.

    The vision loop must never wait on this: a board on Wi-Fi can stall for a second, and the
    detector already has a frame budget.
    """

    def __init__(self, url, rate_hz=4.0, timeout=1.5):
        super().__init__(daemon=True, name="mag-poller")
        self.url = url
        self.period = 1.0 / max(rate_hz, 0.5)
        self.timeout = timeout
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._readings = {}
        self._last_error = None
        self._polls = 0
        self._duplicates = 0
        self._scales = {}

    # -- thread -------------------------------------------------------------------------
    def run(self):
        while not self._stop.is_set():
            try:
                with urllib.request.urlopen(self.url, timeout=self.timeout) as response:
                    document = json.loads(response.read().decode("utf-8"))
            except Exception as exc:                     # noqa: BLE001 - a miss is data too
                with self._lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                self._stop.wait(self.period)
                continue
            now = time.time()
            with self._lock:
                self._last_error = None
                self._polls += 1
                for reading in document.get("sensors") or []:
                    channel = reading.get("ch")
                    if channel is None:
                        continue
                    reads = reading.get("reads")
                    previous = self._readings.get(channel)
                    if reads is not None and previous is not None and previous["reads"] == reads:
                        self._duplicates += 1
                        continue
                    self._scales[channel] = float(reading.get("ut_per_lsb") or 0.0)
                    self._readings[channel] = {
                        "reads": reads,
                        "t": now,
                        "ok": bool(reading.get("ok", True)),
                        "x": float(reading.get("x", 0)),
                        "y": float(reading.get("y", 0)),
                        "z": float(reading.get("z", 0)),
                    }
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()

    # -- reader side --------------------------------------------------------------------
    def snapshot(self, now=None):
        """{(channel): reading} with an 'age' in seconds added, evaluated at `now`."""
        now = time.time() if now is None else now
        with self._lock:
            copy = {channel: dict(value) for channel, value in self._readings.items()}
            error = self._last_error
            polls, duplicates = self._polls, self._duplicates
            scales = dict(self._scales)
        for value in copy.values():
            value["age"] = now - value["t"]
        copy["_meta"] = {"error": error, "polls": polls, "duplicates": duplicates, "scales": scales}
        return copy


class MagBaseline:
    """Per-bay empty-field reference: mean X/Y/Z, in microtesla relative to each sensor's scale."""

    def __init__(self, bays=None, calibrated_at=None, source=None, threshold_ut=None,
                 hysteresis_ut=None):
        self.bays = bays or {}
        self.calibrated_at = calibrated_at
        self.source = source
        self.threshold_ut = DEFAULT_THRESHOLD_UT if threshold_ut is None else threshold_ut
        self.hysteresis_ut = DEFAULT_HYSTERESIS_UT if hysteresis_ut is None else hysteresis_ut

    # -- capture ------------------------------------------------------------------------
    @classmethod
    def capture(cls, poller, mapping, duration=3.0, min_samples=MIN_SAMPLES, source=None,
                threshold_ut=None, hysteresis_ut=None, log=print):
        """Average the field over `duration` seconds and build a baseline per mapped bay.

        The windows are collected from the poller the vision loop already feeds, so nothing here
        blocks the camera; the caller runs this in its own thread.
        """
        wanted = {bay: channel for bay, channel in mapping.items() if channel is not None}
        started = time.time()
        series = {bay: {"x": [], "y": [], "z": []} for bay in wanted}
        seen = set()
        while time.time() - started < duration:
            frame = poller.snapshot()
            for bay, channel in wanted.items():
                reading = frame.get(channel)
                if not reading or not reading.get("ok") or reading.get("age", 99) > STALE_S:
                    continue
                # Keyed per bay as well as per reading: two bays mapped to the same channel is a
                # broken mapping, but it must not silently starve the second bay of all samples.
                key = (bay, channel, reading.get("reads"))
                if key in seen:
                    continue
                seen.add(key)
                for axis in ("x", "y", "z"):
                    series[bay][axis].append(reading[axis])
            time.sleep(0.1)

        bays = {}
        for bay, channel in wanted.items():
            axes = series[bay]
            counts = min(len(axes[axis]) for axis in ("x", "y", "z"))
            entry = {"channel": channel, "samples": counts, "valid": False}
            if counts < min_samples:
                entry["reason"] = f"only {counts} fresh samples in {duration:g} s"
                bays[bay] = entry
                continue
            scale = (poller.snapshot()["_meta"]["scales"] or {}).get(channel) or 0.0
            means = {axis: statistics.fmean(axes[axis]) for axis in ("x", "y", "z")}
            sigma = {axis: (statistics.stdev(axes[axis]) if counts > 1 else 0.0)
                     for axis in ("x", "y", "z")}
            # A chip that answers but never changes may be stuck; a chip reading zero may be
            # disturbed. Both are visible here, and both must not become a baseline.
            span = max(max(axes[axis]) - min(axes[axis]) for axis in ("x", "y", "z"))
            if span == 0:
                entry["reason"] = "sensor output never changed (stuck or disturbed)"
                bays[bay] = entry
                continue
            entry.update({
                "valid": True,
                "ut_per_lsb": scale,
                "mean_xyz_ut": {axis: means[axis] * scale for axis in ("x", "y", "z")},
                "noise_ut": {axis: sigma[axis] * scale for axis in ("x", "y", "z")},
                "span_counts": span,
            })
            bays[bay] = entry
        for bay, channel in mapping.items():
            if channel is None:
                bays[bay] = {"channel": None, "samples": 0, "valid": False,
                             "reason": "no channel mapped for this bay"}
        log(f"  mag baseline: {sum(1 for b in bays.values() if b['valid'])}/{len(bays)} bay(s) valid")
        return cls(bays=bays, calibrated_at=time.time(), source=source,
                   threshold_ut=threshold_ut, hysteresis_ut=hysteresis_ut)

    # -- helpers ------------------------------------------------------------------------
    def invalid_bays(self):
        return {bay: entry.get("reason") or "invalid"
                for bay, entry in self.bays.items() if not entry.get("valid")}

    # -- persistence --------------------------------------------------------------------
    def to_dict(self):
        return {
            "schema": "mag_baseline/1",
            "calibrated_at": self.calibrated_at,
            "source": self.source,
            "threshold_ut": self.threshold_ut,
            "hysteresis_ut": self.hysteresis_ut,
            "bays": self.bays,
        }

    def save(self, path):
        """Write atomically: a half-written baseline would be read as a valid one on restart."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".writing"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=1)
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, ValueError):
            return None
        return cls(bays=document.get("bays") or {}, calibrated_at=document.get("calibrated_at"),
                   source=document.get("source"), threshold_ut=document.get("threshold_ut"),
                   hysteresis_ut=document.get("hysteresis_ut"))


class MagJudge:
    """Turns the rolling field into EMPTY / OCCUPIED / UNKNOWN per bay."""

    def __init__(self, baseline, window_s=WINDOW_S, stale_s=STALE_S):
        self.baseline = baseline
        self.window_s = window_s
        self.stale_s = stale_s
        self._history = {}
        self._state = {}
        self._last_reads = {}

    def update(self, frame, now=None):
        """frame is MagPoller.snapshot(). Returns {bay: {state, delta_ut, channel, reason}}."""
        now = time.time() if now is None else now
        if self.baseline is None:
            return {bay: {"state": "UNKNOWN", "delta_ut": None, "channel": None,
                          "reason": "no baseline (press r with the bays empty)"}
                    for bay in ("B1", "B2", "B3", "B4")}
        result = {}
        for bay, entry in self.baseline.bays.items():
            channel = entry.get("channel")
            out = {"channel": channel, "delta_ut": None, "state": "UNKNOWN", "reason": ""}
            if not entry.get("valid"):
                out["reason"] = entry.get("reason") or "baseline invalid"
                result[bay] = out
                continue
            reading = frame.get(channel)
            if reading is None:
                out["reason"] = f"channel {channel} not reported"
                result[bay] = out
                continue
            if not reading.get("ok"):
                out["reason"] = "sensor reports not ok"
                result[bay] = out
                continue
            if reading.get("age", 99) > self.stale_s:
                out["reason"] = f"stale ({reading['age']:.1f} s)"
                result[bay] = out
                continue

            # Only a reading the chip has actually refreshed belongs in the window. Appending the
            # same reading on every call let the window collapse back to one point whenever the
            # poller had not ticked yet, which flipped the bay to UNKNOWN between updates.
            reads = reading.get("reads")
            if reads is not None and self._last_reads.get(bay) == reads:
                out["reason"] = "no fresh sample yet"
                result[bay] = out
                continue
            self._last_reads[bay] = reads

            scale = entry.get("ut_per_lsb") or 0.0
            series = self._history.setdefault(bay, [])
            series.append((reading["t"], reading["x"] * scale, reading["y"] * scale,
                           reading["z"] * scale))
            cutoff = now - self.window_s
            while series and series[0][0] < cutoff:
                series.pop(0)
            if len(series) < 2:
                out["reason"] = "collecting the 1 s window"
                result[bay] = out
                continue

            # series holds (t, x, y, z) in uT; the baseline is keyed by axis name.
            mean = {"x": statistics.fmean(point[1] for point in series),
                    "y": statistics.fmean(point[2] for point in series),
                    "z": statistics.fmean(point[3] for point in series)}
            base = entry["mean_xyz_ut"]
            delta = math.sqrt(sum((mean[axis] - base[axis]) ** 2 for axis in ("x", "y", "z")))
            out["delta_ut"] = delta
            threshold = self.baseline.threshold_ut
            hysteresis = self.baseline.hysteresis_ut
            previous = self._state.get(bay, "EMPTY")
            if previous == "OCCUPIED":
                state = "OCCUPIED" if delta > threshold - hysteresis else "EMPTY"
            else:
                state = "OCCUPIED" if delta >= threshold else "EMPTY"
            self._state[bay] = state
            out["state"] = state
            result[bay] = out
        return result
