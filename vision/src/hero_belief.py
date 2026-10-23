"""HERO belief: remember where the car was, so one frame cannot move it across the table.

The detector is a per-frame question - "which red blob looks like the arrow right now" - and
that question has no memory, which is how a bay light at (295,252) came to be reported as the
car while the car stood still at (68,475). This module is the memory: it holds a position, a
velocity, an appearance profile and a confidence, and it decides whether the best candidate this
frame is consistent with what it knows.

The rule everything here is built around:

    the car may move, but it cannot drift half the parking lot between two frames.

So motion is not gated by an absolute distance ("more than N px is a reject"). It is gated by a
prediction: where the car should be now, given where it was and how fast it was going, with a
tolerance that widens when it is moving fast and narrows when it is standing still. A car being
pushed across the table passes; a blob appearing 200 px away in one frame does not.

Three separate kinds of evidence, kept apart on purpose:

    position   is this where the car should be, given its motion
    appearance what the car looked like lately - size, colour, and how arrow-like it scores
    commitment  a candidate that disagrees with the committed position cannot take over on one
                frame, however good it looks; it has to persist

That last one is the same pattern as the heading gate in `heading_state`, and for the same
reason: a single frame is the wrong amount of evidence to overturn state.
"""

from __future__ import annotations

import math
from typing import Optional

# --- state machine -----------------------------------------------------------

ACQUIRED = "ACQUIRED"
TRACKING = "TRACKING"
UNCERTAIN = "UNCERTAIN"
LOST = "LOST"
REACQUIRING = "REACQUIRING"


# How much each kind of evidence is worth, per state. Deliberately conservative: the shape weight
# starts where the previous pipeline had it (0.20) and only grows when position has failed.
WEIGHTS = {
    ACQUIRED:   {"position": 0.40, "size": 0.20, "shape": 0.20, "colour": 0.10, "area": 0.10},
    TRACKING:   {"position": 0.40, "size": 0.20, "shape": 0.20, "colour": 0.10, "area": 0.10},
    UNCERTAIN:  {"position": 0.25, "size": 0.20, "shape": 0.30, "colour": 0.15, "area": 0.10},
    LOST:       {"position": 0.15, "size": 0.20, "shape": 0.40, "colour": 0.15, "area": 0.10},
    REACQUIRING:{"position": 0.15, "size": 0.20, "shape": 0.40, "colour": 0.15, "area": 0.10},
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _num(value, fallback):
    """A number, or the fallback.

    `dict.get(key, default)` returns the stored value when the key EXISTS, even when that value is
    None - and the recovery candidates from the whole-frame fallbacks carry s/v/h as None because
    they come from a Pose, not from a mask. That turned a missing colour into a TypeError in the
    middle of the loop, which is how main.py died.
    """
    if value is None:
        return fallback
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if out == out else fallback      # NaN


def _gauss(d: float, sigma: float) -> float:
    """1.0 at zero distance, falling off smoothly. Never negative, never exactly zero."""
    if sigma <= 0:
        return 0.0
    return math.exp(-0.5 * (d / sigma) ** 2)


class HeroBelief:
    """Where the HERO is, and how much that claim is worth.

    Feed it candidate pools; it returns the committed position. Every number it reports about
    itself is in `metrics()` so a silent degradation shows up as a number rather than a car that
    quietly stops moving.
    """

    def __init__(self,
                 # motion model
                 velocity_alpha: float = 0.45,
                 max_speed: float = 60.0,          # px per frame at 10 fps, i.e. ~600 px/s
                 max_prediction: float = 120.0,    # px; never predict further than this
                 radius_base: float = 22.0,        # px, search radius when standing still
                 radius_speed: float = 2.2,        # extra px per px/frame of speed
                 radius_max: float = 180.0,
                 # commitment
                 accept_confidence: float = 0.45,
                 min_motion_score: float = 0.10,   # below this a candidate cannot take over alone
                 challenger_frames: int = 6,
                 # lifecycle
                 uncertain_after: int = 2,
                 lost_after: int = 12,
                 # appearance memory
                 memory_alpha: float = 0.25,
                 # Learning is not free: it is the one part of this class that can be poisoned.
                 # A candidate is only allowed to teach the appearance memory when its identity is
                 # as settled as it gets - a plain accept, close to the prediction, confident.
                 learn_confidence: float = 0.55,
                 # After this many updates the profile is treated as established: further learning
                 # needs a higher confidence, so a slow parade of wrong blobs cannot drift it.
                 lock_after: int = 25,
                 locked_learn_confidence: float = 0.75,
                 # A brand-new target (nothing to compare against, e.g. after the hero was taken
                 # away and a bay light is the only red thing left) must be seen this many frames
                 # in the same place before it is believed.
                 acquire_frames: int = 1,
                 # A takeover is a jump in position. Until the new target has been tracked stably
                 # for this many frames it is on probation and may not teach the profile: if it was
                 # a bay light, the hero comes back and the profile must be unchanged; if it was a
                 # real move, twenty frames later it starts teaching like anything else.
                 takeover_probation: int = 20,
                 # ablation switches, so "which layer contributed" is answerable
                 use_motion_gate: bool = True,
                 use_appearance: bool = True,
                 use_commitment: bool = True):
        self.velocity_alpha = velocity_alpha
        self.max_speed = max_speed
        self.max_prediction = max_prediction
        self.radius_base = radius_base
        self.radius_speed = radius_speed
        self.radius_max = radius_max
        self.accept_confidence = accept_confidence
        self.min_motion_score = min_motion_score
        self.challenger_frames = challenger_frames
        self.uncertain_after = uncertain_after
        self.lost_after = lost_after
        self.memory_alpha = memory_alpha
        self.learn_confidence = learn_confidence
        self.lock_after = lock_after
        self.locked_learn_confidence = locked_learn_confidence
        self.acquire_frames = max(1, int(acquire_frames))
        self.takeover_probation = max(0, int(takeover_probation))
        self.use_motion_gate = use_motion_gate
        self.use_appearance = use_appearance
        self.use_commitment = use_commitment
        self.reset()

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        self.state = LOST
        self.position: Optional[tuple[float, float]] = None
        self.velocity = (0.0, 0.0)
        self.area: Optional[float] = None
        self.width: Optional[float] = None
        self.height: Optional[float] = None
        self.aspect: Optional[float] = None
        self.shape: Optional[float] = None
        self.sat: Optional[float] = None
        self.val: Optional[float] = None
        self.hue: Optional[float] = None
        self.confidence = 0.0
        self.stable_frames = 0
        self.missing_frames = 0
        self.candidate_frames = 0
        self.challenger: Optional[dict] = None
        self.last_score: Optional[dict] = None
        self.last_candidates = 0
        self.profile_updates = 0        # how many candidates have taught the profile
        self.probation = 0              # frames left before a taken-over target may teach
        self.learning_skipped = 0       # candidates that were not allowed to teach it
        self._acq_anchor: Optional[tuple[float, float]] = None
        self._acq_count = 0
        self.challenger_frames_count = 0
        self._last_dt = 0.1
        # metrics
        self.frames = 0
        self.accepts = 0
        self.rejects = 0
        self.challengers_rejected = 0
        self.takeovers = 0
        self.lost_events = 0
        self.reacquired = 0
        self.false_jumps = 0
        self.jump_total_px = 0.0
        self.max_jump_px = 0.0

    # -- motion ------------------------------------------------------------

    def speed(self) -> float:
        return math.hypot(self.velocity[0], self.velocity[1])

    def predict(self, dt: float) -> Optional[tuple[float, float]]:
        if self.position is None:
            return None
        dx, dy = self.velocity[0] * dt, self.velocity[1] * dt
        d = math.hypot(dx, dy)
        if d > self.max_prediction:
            k = self.max_prediction / d
            dx, dy = dx * k, dy * k
        return (self.position[0] + dx, self.position[1] + dy)

    def search_radius(self) -> float:
        """Widens with speed, so a moving car is not lost and a still one is not jumped to."""
        return _clamp(self.radius_base + self.radius_speed * self.speed(),
                      self.radius_base, self.radius_max)

    def should_search_globally(self) -> bool:
        """Global search is a recovery tool, not the first thing tried."""
        return self.state in (LOST, REACQUIRING)

    # -- scoring -----------------------------------------------------------

    def _size_score(self, c: dict) -> float:
        if not self.use_appearance:
            return 0.5
        if self.area is None or self.area <= 0:
            return 0.5
        ratios = []
        for key, expected in (("area", self.area), ("width", self.width),
                              ("height", self.height), ("aspect", self.aspect)):
            got = _num(c.get(key), None)
            if got is None or not expected or expected <= 0:
                continue
            ratios.append(math.log(max(got, 1e-6) / float(expected)))
        if not ratios:
            return 0.5
        # A blob three times the remembered size is not a slightly worse HERO, it is a different
        # thing: log-ratio makes that symmetric and steep.
        worst = max(abs(r) for r in ratios)
        return _gauss(worst, 0.35)

    def _colour_score(self, c: dict) -> float:
        if not self.use_appearance:
            return 0.5
        if self.sat is None or self.val is None:
            return 0.5
        got_s = _num(c.get("s"), None)
        got_v = _num(c.get("v"), None)
        if got_s is None and got_v is None:
            return 0.5                      # no colour evidence: neutral, not a crash
        ds = ((self.sat if got_s is None else got_s) - self.sat) / 45.0
        dv = ((self.val if got_v is None else got_v) - self.val) / 70.0
        return _gauss(math.hypot(ds, dv), 1.0)

    def _shape_score(self, c: dict) -> float:
        got = _num(c.get("shape_score"), None)
        if got is None:
            return 0.5
        # 0 is a perfect match, 0.6 and beyond is a different shape entirely.
        return _clamp(1.0 - float(got) / 0.6, 0.0, 1.0)

    def score(self, c: dict, dt: float) -> dict:
        """Every candidate gets the same six numbers, so no single one can carry a decision."""
        predicted = self.predict(dt)
        if predicted is None:
            position_score = 0.5
            distance = 0.0
        else:
            distance = math.hypot(c["centre"][0] - predicted[0], c["centre"][1] - predicted[1])
            sigma = max(self.search_radius() / 2.0, 8.0)
            position_score = _gauss(distance, sigma)

        size = self._size_score(c)
        colour = self._colour_score(c)
        shape = self._shape_score(c)
        position = position_score

        # Weights depend on what the state can honestly claim. While tracking, position is the
        # strongest evidence there is and shape is a tie-breaker. Once the car is uncertain the
        # position prior has already failed, so shape and appearance carry more.
        w = WEIGHTS.get(self.state, WEIGHTS[TRACKING])
        confidence = (w["position"] * position + w["size"] * size + w["shape"] * shape
                      + w["colour"] * colour
                      + w["area"] * min(1.0, _num(c.get("area"), 1.0)
                                        / max(self.area or _num(c.get("area"), 1.0), 1.0)))
        return {
            "candidate": c,
            "distance": distance,
            "position_score": position,
            "motion_score": position_score,        # same evidence, named for the report
            "size_score": size,
            "colour_score": colour,
            "shape_score": shape,
            "confidence": _clamp(confidence, 0.0, 1.0),
        }

    # -- memory ------------------------------------------------------------

    def _remember(self, sc: dict) -> None:
        c = sc["candidate"]
        # An established profile only listens to strong evidence. Without this, a slow parade of
        # wrong blobs - a bay light taking over every few minutes - would drag the profile to them
        # and the real hero would stop matching.
        if self.profile_updates >= self.lock_after:
            if sc["confidence"] < self.locked_learn_confidence:
                self.learning_skipped += 1
                return
            a = self.memory_alpha * 0.35
        else:
            a = self.memory_alpha
        def blend(old, new):
            if old is None:
                return float(new)
            return (1 - a) * old + a * float(new)
        self.area = blend(self.area, _num(c.get("area"), self.area or 1))
        self.width = blend(self.width, _num(c.get("width"), self.width or 1))
        self.height = blend(self.height, _num(c.get("height"), self.height or 1))
        self.aspect = blend(self.aspect, _num(c.get("aspect"), self.aspect or 1))
        shape_now = _num(c.get("shape_score"), None)
        if shape_now is not None:
            self.shape = blend(self.shape, shape_now)
        for name in ("s", "v", "h"):
            got = _num(c.get(name), None)
            if got is not None:
                setattr(self, {"s": "sat", "v": "val", "h": "hue"}[name], blend(
                    getattr(self, {"s": "sat", "v": "val", "h": "hue"}[name]), got))

    def _move_to(self, sc: dict, dt: float, learn: bool = True) -> None:
        c = sc["candidate"]
        x, y = float(c["centre"][0]), float(c["centre"][1])
        if self.position is not None and dt > 0:
            measured = ((x - self.position[0]) / dt, (y - self.position[1]) / dt)
            speed = math.hypot(*measured)
            if speed > self.max_speed:
                k = self.max_speed / speed
                measured = (measured[0] * k, measured[1] * k)
            a = self.velocity_alpha
            self.velocity = (a * measured[0] + (1 - a) * self.velocity[0],
                             a * measured[1] + (1 - a) * self.velocity[1])
        self.position = (x, y)
        self.confidence = sc["confidence"]
        # Learning is opt-in and only for the boring case. A takeover is the single most suspicious
        # event this class produces, and it is exactly the path the hero disappearing takes: the
        # bay light wins six frames, is committed, and then teaches the profile its own colour.
        if self.probation > 0:
            self.learning_skipped += 1
            learn = False                # a just-taken-over target has not earned the profile yet
        if learn and sc["confidence"] >= self.learn_confidence:
            self._remember(sc)
            self.profile_updates += 1
        else:
            self.learning_skipped += 1

    # -- the decision ------------------------------------------------------

    def update(self, candidates: list[dict], dt: float = 0.1) -> dict:
        dt = _clamp(dt, 1e-3, 0.5)
        self._last_dt = dt
        self.frames += 1
        if self.probation > 0:
            self.probation -= 1

        self.last_candidates = len(candidates)
        if not candidates:
            return self._miss()

        scored = [self.score(c, dt) for c in candidates]
        scored.sort(key=lambda s: s["confidence"], reverse=True)
        best = scored[0]

        # --- no committed position yet: acquire
        if self.position is None:
            if best["confidence"] < self.accept_confidence:
                return self._miss(reason="weak")
            cx, cy = best["candidate"]["centre"]
            # Nothing to compare against yet. A bay light is a perfectly confident candidate the
            # moment the hero is taken off the table, so a brand-new target has to be in the same
            # place for a few frames before it is believed - and it never teaches the profile.
            if (self._acq_anchor is not None
                    and math.hypot(cx - self._acq_anchor[0], cy - self._acq_anchor[1]) < 25.0):
                self._acq_count += 1
            else:
                self._acq_anchor = (cx, cy)
                self._acq_count = 1
            if self._acq_count < self.acquire_frames:
                self.state = REACQUIRING
                self.missing_frames += 1
                self.learning_skipped += 1
                return self._result(None, action="acquire_pending")
            self.state = ACQUIRED
            self._acq_anchor = None
            self._acq_count = 0
            # Cold start may learn - there is no profile to poison yet. A re-acquisition may not:
            # if the profile already exists, the only reason for position to be None is that the
            # hero was lost, and whatever is red now is the wrong teacher.
            self._move_to(best, dt, learn=(self.profile_updates == 0))
            self.stable_frames = 1
            self.missing_frames = 0
            self.accepts += 1
            if self.reacquired or self.lost_events:
                self.reacquired += 1
            return self._result(best, action="acquire")

        # --- is this consistent with the committed position?
        compatible = best["confidence"] >= self.accept_confidence
        if self.use_motion_gate:
            compatible = compatible and best["motion_score"] >= self.min_motion_score
        if not self.use_commitment:
            # Ablation only: no challenger stage, any confident candidate takes over at once.
            compatible = best["confidence"] >= self.accept_confidence

        if compatible:
            jump = best["distance"]
            self._move_to(best, dt)
            self.stable_frames += 1
            self.missing_frames = 0
            self.confidence = best["confidence"]
            self.state = ACQUIRED if self.stable_frames == 1 else TRACKING
            self.accepts += 1
            self.challenger = None
            self.challenger_frames_count = 0
            if jump > 40.0:
                self.jump_total_px += jump
                self.max_jump_px = max(self.max_jump_px, jump)
            return self._result(best, action="accept")

        # --- a candidate that disagrees. It may be a real move; it may be a bay light.
        chall = best
        if chall["confidence"] < self.accept_confidence:
            self.rejects += 1
            return self._miss(reason="weak")

        if (self.challenger is not None
                and math.hypot(chall["candidate"]["centre"][0] - self.challenger["centre"][0],
                               chall["candidate"]["centre"][1] - self.challenger["centre"][1]) < 40.0):
            self.challenger_frames_count += 1
            self.challenger = chall["candidate"]
        else:
            self.challenger = chall["candidate"]
            self.challenger_frames_count = 1
        self.candidate_frames = self.challenger_frames_count

        if self.challenger_frames_count >= self.challenger_frames:
            # Persistent support on one new place: that is a move, not a glitch.
            self.takeovers += 1
            self.false_jumps += 1                       # a move across the gate, counted
            self._move_to(chall, dt, learn=False)
            self.probation = self.takeover_probation
            self.state = TRACKING
            self.stable_frames += 1
            self.missing_frames = 0
            self.challenger = None
            self.challenger_frames_count = 0
            return self._result(chall, action="takeover")

        self.challengers_rejected += 1
        self.rejects += 1
        self.missing_frames += 1
        self.confidence *= 0.9
        if self.missing_frames >= self.uncertain_after:
            self.state = UNCERTAIN
        return self._result(None, action="challenge_held")

    def _miss(self, reason: str = "none") -> dict:
        self.missing_frames += 1
        self.stable_frames = 0
        self.confidence *= 0.85
        if self.position is None:
            self.state = REACQUIRING
            return self._result(None, action="unavailable")
        was_lost = self.state in (LOST, REACQUIRING)
        if self.missing_frames >= self.lost_after:
            self.state = LOST
            if not was_lost:
                self.lost_events += 1
        elif self.missing_frames >= self.uncertain_after:
            self.state = UNCERTAIN
        else:
            self.state = TRACKING if self.state in (ACQUIRED, TRACKING) else self.state
        return self._result(None, action="hold" if self.state != LOST else "lost")

    def _result(self, sc: Optional[dict], action: str) -> dict:
        # Keep the winning candidate's evidence around. When a wrong blob is promoted, the first
        # question is always "which score pushed it up", and without this the answer is gone.
        self.last_score = sc
        return {
            "position": self.position,
            "heading_candidate": None,
            "state": self.state,
            "confidence": self.confidence,
            "stale": action not in ("accept", "acquire", "takeover"),
            "action": action,
            "score": sc,
            "missing_frames": self.missing_frames,
            "stable_frames": self.stable_frames,
        }

    def metrics(self) -> dict:
        return {
            "hero_state": self.state,
            "hero_confidence": round(self.confidence, 3),
            "hero_stable_frames": self.stable_frames,
            "hero_missing_frames": self.missing_frames,
            "hero_candidate_frames": self.candidate_frames,
            "hero_frames": self.frames,
            "hero_accepts": self.accepts,
            "hero_rejects": self.rejects,
            "hero_challengers_rejected": self.challengers_rejected,
            "hero_takeovers": self.takeovers,
            "hero_lost_events": self.lost_events,
            "hero_reacquired": self.reacquired,
            "hero_jumps_over_40px": self.false_jumps,
            "hero_max_jump_px": round(self.max_jump_px, 1),
            "hero_speed_px_per_frame": round(self.speed(), 2),
            "hero_search_radius_px": round(self.search_radius(), 1),
            "hero_profile_updates": self.profile_updates,
            "hero_learning_skipped": self.learning_skipped,
            "hero_profile_locked": self.profile_updates >= self.lock_after,
            "hero_probation_frames": self.probation,
        }
