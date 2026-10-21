"""Heading commit state: a gate, and persistence, independent of any detector.

Both mechanisms existed before, but inside `hero_hog` (`HEADING_GATE`, `HEADING_PERSIST`,
`_heading_disagreements`), so when HOG left the runtime path they left with it. They are not
properties of HOG - they are properties of the *output* - so they live here now and any
detector can feed them.

Two jobs, and they are deliberately separate:

  gate        a reading within 30 degrees of the committed direction is accepted at once, so
              normal jitter passes through and gets absorbed by the committed value
  persistence a reading further away than that is held pending; only a run of many consecutive
              frames agreeing on the same new direction is believed to be a real turn

The failure this guards against is not noise, it is a clean 180 degree flip: on this marker the
head/tail comparison reversed between frames that differed only in blur (measured 185, 5, 182,
4, 186, 5 over twenty frames). A car cannot change ends between two frames, so a single
contradictory reading must not be believed - but a car being pushed around a corner must
eventually be.

Deliberate difference from the original implementation: that one counted frames exceeding the
gate without requiring them to agree with each other, so twelve mutually inconsistent readings
would commit. This requires the pending readings to agree with the pending direction too.
"""

from __future__ import annotations

import math
from typing import Optional

HEADING_GATE = 30.0        # degrees; well under 180, leaves room for a car turning on the table
HEADING_PERSIST = 12       # consecutive frames of agreement before a reversal is believed


def circ_diff(a: float, b: float) -> float:
    """Signed difference a-b in (-180, 180]."""
    return ((a - b + 180.0) % 360.0) - 180.0


class HeadingState:
    """Feed it readings, get a committed heading back.

    `update()` returns a dict rather than a bare float because the caller has to be able to
    distinguish "this is today's measurement" from "this is yesterday's, carried forward":

        heading  the committed direction, or None if nothing has ever been read
        stale    True when the returned heading is not this frame's reading
        action   initial | accepted | pending | committed | hold | unavailable
    """

    def __init__(self, gate: float = HEADING_GATE, persist: int = HEADING_PERSIST):
        self.gate = float(gate)
        self.persist = int(persist)
        self.reset()

    def reset(self) -> None:
        self.committed: Optional[float] = None
        self.pending: Optional[float] = None
        self.pending_count = 0
        self.readings = 0
        self.failures = 0
        self.accepted = 0
        self.gated = 0
        self.commits = 0
        self.holds = 0
        self.unavailable = 0
        self.consecutive_failures = 0

    def seed(self, heading: Optional[float]) -> None:
        """Set the committed direction without counting it as a reading (tests, resumes)."""
        self.committed = None if heading is None else float(heading) % 360.0
        self.pending = None
        self.pending_count = 0

    @property
    def stale(self) -> bool:
        """True when the committed direction was not confirmed by the latest reading."""
        return self.consecutive_failures > 0 or self.pending is not None

    def update(self, reading: Optional[float]) -> dict:
        if reading is None:
            self.failures += 1
            self.consecutive_failures += 1
            if self.committed is None:
                self.unavailable += 1
                return {"heading": None, "stale": False, "action": "unavailable"}
            # The car did not stop existing because the arrow was unreadable. Hold, and say so.
            self.holds += 1
            return {"heading": self.committed, "stale": True, "action": "hold"}

        reading = float(reading) % 360.0
        self.readings += 1
        self.consecutive_failures = 0

        if self.committed is None:
            self.committed = reading
            self.pending = None
            self.pending_count = 0
            self.accepted += 1
            return {"heading": reading, "stale": False, "action": "initial"}

        diff = circ_diff(reading, self.committed)
        if abs(diff) <= self.gate:
            # Normal jitter. Accept it, and let the committed value track the car.
            self.committed = reading
            self.pending = None
            self.pending_count = 0
            self.accepted += 1
            return {"heading": reading, "stale": False, "action": "accepted"}

        # Far from the committed direction. Is it the same far direction as last time?
        if self.pending is not None and abs(circ_diff(reading, self.pending)) <= self.gate:
            self.pending_count += 1
        else:
            self.pending = reading
            self.pending_count = 1
        self.gated += 1

        if self.pending_count >= self.persist:
            self.committed = self.pending
            self.commits += 1
            self.pending = None
            self.pending_count = 0
            return {"heading": self.committed, "stale": False, "action": "committed"}

        return {"heading": self.committed, "stale": True, "action": "pending"}

    def metrics(self) -> dict:
        return {
            "heading_readings_total": self.readings,
            "heading_failures_total": self.failures,
            "heading_accepted_total": self.accepted,
            "heading_gated_total": self.gated,
            "heading_commits_total": self.commits,
            "heading_hold_total": self.holds,
            "heading_unavailable_total": self.unavailable,
            "heading_pending_count": self.pending_count,
            "heading_consecutive_failures": self.consecutive_failures,
            "heading_stale": self.stale,
        }
