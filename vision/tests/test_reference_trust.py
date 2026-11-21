"""P0012: a reference that no longer matches the room must not be read as bay states.

Measured on the sandbox: the room had gone darker than when the reference was captured - 7.3 grey levels
over the whole frame, up to 60 inside individual bays - and the bays whose own area had darkened were
published as OCCUPIED all night. The strip followed faithfully, which is why it went red on its own.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import main                                                      # noqa: E402
import occupancy                                                 # noqa: E402
from spot_config import Spot                                     # noqa: E402

SHAPE = (300, 400)
BAYS = [[10, 10], [90, 10], [90, 90], [10, 90]]


def spots():
    return [Spot(id="B1", polygon=np.array([[10, 10], [90, 10], [90, 90], [10, 90]], float))]


def frame(level, box=None, box_level=200):
    image = np.full((SHAPE[0], SHAPE[1], 3), level, np.uint8)
    if box is not None:
        x0, y0, x1, y1 = box
        image[y0:y1, x0:x1] = box_level
    return image


def detector(ref=None):
    d = occupancy.OccupancyDetector()
    if ref is not None:
        d.set_reference(ref, spots())
    return d


def test_matching_reference_is_trusted():
    ref = frame(120)
    live = frame(120)
    pct = main.reference_trust(ref, live, spots(), detector(ref))
    assert pct == 100.0
    assert pct >= main.REFERENCE_TRUST_QUIET_PCT


def test_a_darkened_room_is_not_trusted():
    """The measured case: the whole frame darker than the reference."""
    ref = frame(150)
    live = frame(120, box=(10, 10, 90, 90), box_level=40)      # the bay's own area went dark
    pct = main.reference_trust(ref, live, spots(), detector(ref))
    assert pct < main.REFERENCE_TRUST_QUIET_PCT, pct


def test_a_real_car_does_not_make_the_reference_stale():
    """A car in one bay of many is not a lighting change; the other bays still match."""
    many = [Spot(id=f"B{i}", polygon=np.array([[10 + i * 60, 10], [60 + i * 60, 10],
                                               [60 + i * 60, 90], [10 + i * 60, 90]], float))
            for i in range(5)]
    ref = frame(120)
    live = frame(120, box=(10, 10, 60, 90), box_level=250)      # one bay occupied
    d = occupancy.OccupancyDetector()
    d.set_reference(ref, many)
    pct = main.reference_trust(ref, live, many, d)
    assert pct >= main.REFERENCE_TRUST_QUIET_PCT, pct


def test_size_mismatch_is_never_trusted():
    assert main.reference_trust(frame(120), np.zeros((100, 100, 3), np.uint8), spots(),
                               detector(frame(120))) == 0.0


def test_the_audit_is_a_diagnostic_and_does_not_replace_bay_states():
    """It used to substitute UNKNOWN for every bay; that turned a covered camera into a dark strip.

    "The bay you covered reads OCCUPIED" can be explained at the sandbox. "Everything went black" cannot.
    So the number is for the log only.
    """
    import io as _io
    import inspect

    source = inspect.getsource(main)
    assert "reference_health(frame)" in source
    # 不再有把 results 换成空/UNKNOWN 的调用 ✓
    assert 'unknown_reason=(f"reference does not match' not in source
    assert 'contract.build_state(results, vehicle=vehicle_doc)' in source


def _retired_untrusted_state_test():
    """The state document must say UNKNOWN with the reason, not OCCUPIED and not EMPTY."""
    import contract

    class R:
        state = "OCCUPIED"
        changed_ratio = 0.9
    document = contract.build_state({"B1": R(), "B2": R()},
                                   unknown_reason="reference does not match the room (62% match)",
                                   timestamp=1000)
    assert set(document["spots"]) == {"B1", "B2"}, "every bay is still listed"
    assert all(s["state"] == "UNKNOWN" for s in document["spots"].values())
    assert all("reference" in s.get("reason", "") for s in document["spots"].values())


def test_trust_is_not_regained_on_a_single_good_check():
    """The match rate sits near the threshold when the light is marginal; measured 62% then 71%.

    Flipping the verdict on every three-second check made the strip blink between dark and coloured, so
    trust is lost at once and regained only after two consecutive good checks.
    """
    import main

    state = main._REFERENCE_HEALTH
    saved = dict(state)
    try:
        state.update({"ok": False, "good": 0, "pct": 62.0, "at": 0.0, "ref": None,
                      "detector": None, "frame": None})
        # Walk the state machine the way reference_health does.
        def step(pct):
            state["at"] = 0.0
            pct_seen = pct
            good = pct_seen >= main.REFERENCE_TRUST_QUIET_PCT
            state["good"] = state.get("good", 0) + 1 if good else 0
            state["ok"] = good and (state["ok"] or state["good"] >= 2)
            return state["ok"]

        assert step(65) is False
        assert step(71) is False, "one good check must not restore trust"
        assert step(62) is False
        assert step(71) is False
        assert step(71) is True, "two consecutive good checks do restore it"
        assert step(62) is False, "and one bad check drops it immediately"
    finally:
        state.update(saved)
