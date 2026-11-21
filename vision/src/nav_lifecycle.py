"""P0008: end a navigation when the tracked vehicle actually stops, without a browser open.

The page already does this (`checkArrival` in navigation.html), but it only runs while somebody has
the navigation tab open and the map loaded. With the tab closed, /web/route.json keeps naming the
last target, the bay lights keep pointing at it, and the only way out is pressing "结束导航" - which
is the reported bug.

Decisions here use only what vision already publishes:

    out/parking_state.json   vehicle.location = {"kind": "spot", "id": "B3"} - the vision system's
                             own answer to "which bay is the tracked vehicle in", plus the bay states
    out/route.json           the active target

There is no new tracking, no ReID, no second camera. What it does assume, and says out loud: the
vehicle vision tracks is the vehicle being navigated, and there is only one of them.

Both endings require the same demonstration that the car has *stopped*, not merely passed through:
the same bay for HOLD_S seconds, with the position spread below STILL_PX. Driving across a bay and
out the other side fails both.
"""

import json
import math
import os
import time

# Long enough that a car driving past does not qualify, short enough that a demo is not waiting.
HOLD_S = 2.5
# A parked car jitters a couple of pixels; a car being pushed across the floor does not.
STILL_PX = 8.0

COMPLETED = "COMPLETED"
ENDED_ELSEWHERE = "ENDED_ELSEWHERE"


def _read(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def vehicle_bay(state_document):
    """The bay the tracked vehicle is in, or None when vision is not saying.

    None covers three different situations on purpose: no vehicle visible, a stale position, and the
    vehicle sitting on a road or the floor. None must never be read as "parked somewhere else".
    """
    vehicle = (state_document or {}).get("vehicle") or {}
    if not vehicle.get("visible"):
        return None
    if vehicle.get("position_stale"):
        return None
    location = vehicle.get("location") or {}
    if location.get("kind") != "spot":
        return None
    identifier = location.get("id")
    return None if identifier is None else str(identifier)


def track_active(state_document, route_document):
    """True while a target is set and the route document has not already been ended."""
    if not route_document:
        return False
    if route_document.get("code") == "navigation-ended" or route_document.get("ok") is False:
        return False
    return bool(str(route_document.get("target_spot") or "").strip())


class NavLifecycle:
    """Watches the association "tracked vehicle is in bay X" and ends the navigation when it settles."""

    def __init__(self, state_path, route_path, hold_s=HOLD_S, still_px=STILL_PX):
        self.state_path = state_path
        self.route_path = route_path
        self.hold_s = hold_s
        self.still_px = still_px
        self._bay = None
        self._since = None
        self._positions = []
        # Which navigation was already ended: (target, generated_at). Keying on the target alone
        # made a new navigation to the same bay the same navigation - it would never end again, and
        # the only way out was to cancel it by hand. The route document already carries
        # generated_at, so a re-selection is distinguishable without adding a new field.
        self._ended_for = None
        self.last_outcome = None

    def _reset_hold(self):
        self._bay = None
        self._since = None
        self._positions = []

    def step(self, now=None):
        """One observation. Returns an outcome dict when it ended a navigation, else None."""
        now = time.time() if now is None else now
        state = _read(self.state_path)
        route = _read(self.route_path)
        if not track_active(state, route):
            self._reset_hold()
            # A new navigation (or a manual end) makes the previous ending irrelevant again.
            # A cleared route, or a different navigation, makes the previous ending irrelevant.
            if route is None or (str(route.get("target_spot") or ""),
                                 route.get("generated_at")) != self._ended_for:
                self._ended_for = None
            return None

        target = str(route.get("target_spot")).strip()
        started = route.get("generated_at")
        key = (target, started)
        if key == self._ended_for:
            return None                       # already ended this one; do not act twice

        bay = vehicle_bay(state)
        if bay is None:
            self._reset_hold()                # not visible / on a road / stale: keep navigating
            return None

        vehicle = (state or {}).get("vehicle") or {}
        position = (float(vehicle.get("x", 0.0)), float(vehicle.get("y", 0.0)))

        if bay != self._bay:
            self._bay = bay
            self._since = now
            self._positions = [position]
            return None
        self._positions.append(position)
        if len(self._positions) > 200:
            self._positions.pop(0)
        held = now - (self._since or now)
        if held < self.hold_s:
            return None

        spread = self._spread()
        if spread > self.still_px:
            # Moving within the same bay: start the clock again rather than counting the crawl.
            self._since = now
            self._positions = [position]
            return None

        spots = (state or {}).get("spots") or {}
        state_of_bay = (spots.get(bay) or {}).get("state")
        if state_of_bay != "OCCUPIED":
            # The vision detector has not called the bay occupied yet, so "parked" is not established.
            return None

        return self._end(target, bay, held, spread, state_of_bay, key)

    def _spread(self):
        if len(self._positions) < 2:
            return 0.0
        xs = [p[0] for p in self._positions]
        ys = [p[1] for p in self._positions]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys))

    def _end(self, target, bay, held, spread, bay_state, key=None):
        if bay == target:
            code, reason = COMPLETED, f"arrived at {target}"
        else:
            code, reason = ENDED_ELSEWHERE, f"parked in {bay}, so the navigation to {target} ended"
        document = {
            "schema": "route/1",
            "generated_at": time.time(),
            "target_spot": "",
            "ok": False,
            "code": "navigation-ended",
            "reason": reason,
            "ended": code,
            "ended_bay": bay,
            "ended_target": target,
            "start_node": None,
            "target_node": None,
            "nodes": [],
            "distance": None,
        }
        tmp = self.route_path + ".writing"
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(document, handle, indent=1)
            os.replace(tmp, self.route_path)
        except OSError:
            return None
        self._ended_for = key if key is not None else target
        self._reset_hold()
        self.last_outcome = {"ended": code, "bay": bay, "target": target,
                             "held_s": round(held, 2), "spread_px": round(spread, 1),
                             "bay_state": bay_state, "reason": reason}
        return self.last_outcome
