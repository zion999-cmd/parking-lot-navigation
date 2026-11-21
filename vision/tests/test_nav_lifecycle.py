"""P0008: a navigation ends when the tracked vehicle stops, and only then.

The cases that matter are the ones where a naive implementation would be wrong: a car driving past
the target, another car occupying a bay, and the vehicle position being stale. Each of those must
leave the navigation running.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import nav_lifecycle as N                                       # noqa: E402


def state(hero_bay=None, x=300.0, y=500.0, occupied=("B3",), visible=True, stale=False,
          kind=None, bays=None):
    spots = {bay: {"state": "EMPTY", "changed_ratio": 0.0} for bay in ("B1", "B2", "B3", "B4")}
    for bay in occupied:
        spots[bay] = {"state": "OCCUPIED", "changed_ratio": 0.6}
    for bay, value in (bays or {}).items():
        spots[bay] = value
    location = ({"kind": "road", "id": None} if kind == "road"
                else {"kind": "floor", "id": None} if kind == "floor"
                else {"kind": "spot", "id": hero_bay} if hero_bay
                else {"kind": "unknown", "id": None})
    return {"schema": "state/1", "timestamp": 1.0, "spots": spots,
            "vehicle": {"id": "hero", "x": x, "y": y, "heading": 0.0, "visible": visible,
                        "position_stale": stale, "location": location}}


def route(target="B3"):
    return {"schema": "route/1", "ok": True, "target_spot": target, "start_node": "n1",
            "target_node": "n2", "nodes": ["n1", "n2"], "distance": 100.0}


class Fixture:
    def __init__(self, tmp_path):
        self.state_path = os.path.join(tmp_path, "parking_state.json")
        self.route_path = os.path.join(tmp_path, "route.json")
        self.write(state(), route())

    def write(self, state_doc, route_doc):
        with open(self.state_path, "w", encoding="utf-8") as handle:
            json.dump(state_doc, handle)
        with open(self.route_path, "w", encoding="utf-8") as handle:
            json.dump(route_doc, handle)

    def route_doc(self):
        with open(self.route_path, encoding="utf-8") as handle:
            return json.load(handle)


def run(monkeypatch, lifecycle, fixture, steps):
    """Feed a sequence of (state document, elapsed seconds) observations."""
    outcomes = []
    clock = [1000.0]
    for state_doc, elapsed in steps:
        clock[0] += elapsed
        fixture.write(state_doc, fixture.route_doc())
        outcome = lifecycle.step(now=clock[0])
        if outcome:
            outcomes.append(outcome)
    return outcomes


def make(tmp_path):
    fixture = Fixture(str(tmp_path))
    lifecycle = N.NavLifecycle(fixture.state_path, fixture.route_path, hold_s=2.5, still_px=8.0)
    return fixture, lifecycle


def test_no_target_means_nothing_happens(tmp_path):
    fixture, lifecycle = make(tmp_path)
    fixture.write(state(hero_bay="B3"), {"schema": "route/1", "ok": False, "target_spot": "",
                                         "code": "navigation-ended"})
    assert lifecycle.step(now=1000.0) is None


def test_arrival_at_target_completes(tmp_path):
    fixture, lifecycle = make(tmp_path)
    steps = [(state(hero_bay="B3"), 0.5)] * 6          # 2.5 s parked in B3
    outcomes = run(None, lifecycle, fixture, steps)
    assert len(outcomes) == 1
    assert outcomes[0]["ended"] == "COMPLETED"
    assert outcomes[0]["bay"] == "B3"
    document = fixture.route_doc()
    assert document["target_spot"] == "" and document["ok"] is False
    assert document["code"] == "navigation-ended"
    assert "arrived at B3" in document["reason"]


def test_parking_in_another_bay_ends_elsewhere(tmp_path):
    fixture, lifecycle = make(tmp_path)
    steps = [(state(hero_bay="B2", occupied=("B2",)), 0.5)] * 6
    outcomes = run(None, lifecycle, fixture, steps)
    assert len(outcomes) == 1
    assert outcomes[0]["ended"] == "ENDED_ELSEWHERE"
    assert outcomes[0]["bay"] == "B2" and outcomes[0]["target"] == "B3"
    assert "parked in B2" in fixture.route_doc()["reason"]


def test_driving_past_the_target_does_not_complete(tmp_path):
    fixture, lifecycle = make(tmp_path)
    steps = [(state(hero_bay="B3"), 0.4), (state(hero_bay="B3"), 0.4),
             (state(hero_bay=None, kind="road"), 0.4), (state(hero_bay="B4"), 0.4)]
    assert run(None, lifecycle, fixture, steps) == []
    assert fixture.route_doc()["ok"] is True


def test_moving_inside_the_bay_restarts_the_clock(tmp_path):
    fixture, lifecycle = make(tmp_path)
    steps = [(state(hero_bay="B3", x=300.0), 0.5),
             (state(hero_bay="B3", x=330.0), 0.5),      # 30 px: still moving
             (state(hero_bay="B3", x=331.0), 0.5),
             (state(hero_bay="B3", x=331.0), 0.5)]
    assert run(None, lifecycle, fixture, steps) == []


def test_another_car_in_the_target_does_not_count(tmp_path):
    """B3 occupied by somebody else, our vehicle on the road: the navigation must keep going."""
    fixture, lifecycle = make(tmp_path)
    steps = [(state(hero_bay=None, kind="road", occupied=("B3",)), 0.5)] * 6
    assert run(None, lifecycle, fixture, steps) == []
    assert fixture.route_doc()["ok"] is True


def test_another_car_parked_elsewhere_does_not_end_ours(tmp_path):
    fixture, lifecycle = make(tmp_path)
    steps = [(state(hero_bay=None, kind="road", occupied=("B2", "B3")), 0.5)] * 6
    assert run(None, lifecycle, fixture, steps) == []


def test_stale_position_does_not_count(tmp_path):
    fixture, lifecycle = make(tmp_path)
    steps = [(state(hero_bay="B3", stale=True), 0.5)] * 6
    assert run(None, lifecycle, fixture, steps) == []


def test_invisible_vehicle_does_not_count(tmp_path):
    fixture, lifecycle = make(tmp_path)
    steps = [(state(hero_bay="B3", visible=False), 0.5)] * 6
    assert run(None, lifecycle, fixture, steps) == []


def test_bay_not_occupied_yet_is_not_parked(tmp_path):
    """The association says B3 but the detector has not called it occupied: wait."""
    fixture, lifecycle = make(tmp_path)
    steps = [(state(hero_bay="B3", occupied=()), 0.5)] * 6
    assert run(None, lifecycle, fixture, steps) == []


def test_a_second_arrival_is_ignored(tmp_path):
    fixture, lifecycle = make(tmp_path)
    steps = [(state(hero_bay="B3"), 0.5)] * 6
    assert len(run(None, lifecycle, fixture, steps)) == 1
    # Same documents, more observations: the route is already ended, nothing new may happen.
    more = [(state(hero_bay="B3"), 0.5)] * 6
    assert run(None, lifecycle, fixture, more) == []


def test_manual_end_then_new_target_works(tmp_path):
    fixture, lifecycle = make(tmp_path)
    steps = [(state(hero_bay="B3"), 0.5)] * 6
    assert len(run(None, lifecycle, fixture, steps)) == 1
    # User picks another bay: the lifecycle must not think it already ended this one.
    fixture.write(state(hero_bay=None, kind="road"), route(target="B1"))
    assert lifecycle.step(now=2000.0) is None
    steps = [(state(hero_bay="B1", occupied=("B1",)), 0.5)] * 6
    outcomes = run(None, lifecycle, fixture, steps)
    assert len(outcomes) == 1 and outcomes[0]["target"] == "B1"


# --- P0009-B: the same target selected again is a new navigation ---------------------------

def test_same_target_selected_again_is_a_new_navigation(tmp_path):
    """Re-selecting B3 straight after finishing B3 used to be treated as the finished one.

    The lifecycle remembered the target it ended, so a new navigation with the same target could
    never end: the arrival was ignored and the only way out was cancelling by hand.
    """
    fixture, lifecycle = make(tmp_path)
    steps = [(state(hero_bay="B3"), 0.5)] * 6
    first = run(None, lifecycle, fixture, steps)
    assert len(first) == 1 and first[0]["ended"] == "COMPLETED"

    # A new navigation to the same bay. generated_at differs; nothing observed in between.
    new_route = route(target="B3")
    new_route["generated_at"] = fixture.route_doc().get("generated_at", 0) + 1
    fixture.write(state(hero_bay=None, kind="road"), new_route)
    assert lifecycle.step(now=2000.0) is None

    second = run(None, lifecycle, fixture, [(state(hero_bay="B3"), 0.5)] * 6)
    assert len(second) == 1, "the second navigation must end on its own"
    assert second[0]["ended"] == "COMPLETED" and second[0]["target"] == "B3"


def test_same_target_navigation_is_still_ended_only_once(tmp_path):
    fixture, lifecycle = make(tmp_path)
    steps = [(state(hero_bay="B3"), 0.5)] * 6
    assert len(run(None, lifecycle, fixture, steps)) == 1
    # Same document, more observations: no second ending, even though the target matches.
    assert run(None, lifecycle, fixture, [(state(hero_bay="B3"), 0.5)] * 6) == []


def test_cancel_then_same_target_again_works(tmp_path):
    fixture, lifecycle = make(tmp_path)
    fixture.write(state(hero_bay=None, kind="road"), {"schema": "route/1", "ok": False,
                                                      "target_spot": "", "code": "navigation-ended"})
    assert lifecycle.step(now=1000.0) is None
    fixture.write(state(hero_bay=None, kind="road"), route(target="B3"))
    assert lifecycle.step(now=1001.0) is None
    outcomes = run(None, lifecycle, fixture, [(state(hero_bay="B3"), 0.5)] * 6)
    assert len(outcomes) == 1 and outcomes[0]["target"] == "B3"
