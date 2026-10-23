"""Tests for the HERO belief layer: motion, appearance memory, and the commitment gate."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))

import hero_belief            # noqa: E402


def cand(x, y, area=270, w=32, h=17, shape=0.18, s=110.0, v=160.0):
    return {"bbox": (int(x - w / 2), int(y - h / 2), w, h), "centre": (float(x), float(y)),
            "area": area, "width": w, "height": h, "aspect": w / float(h),
            "shape_score": shape, "s_min": 60, "h": 25.0, "s": s, "v": v}


def tracked_belief():
    b = hero_belief.HeroBelief()
    b.update([cand(100, 300)], 0.1)
    assert b.state == hero_belief.ACQUIRED
    return b


def test_acquires_from_a_confident_candidate() -> None:
    b = hero_belief.HeroBelief()
    v = b.update([cand(100, 300)], 0.1)
    assert v["action"] == "acquire"
    assert v["position"] == (100.0, 300.0)
    assert b.state == hero_belief.ACQUIRED


def test_nothing_is_reported_without_a_position() -> None:
    b = hero_belief.HeroBelief()
    v = b.update([], 0.1)
    assert v["position"] is None
    assert v["action"] == "unavailable"


def test_small_movement_is_accepted_and_tracked() -> None:
    b = tracked_belief()
    v = b.update([cand(110, 302)], 0.1)
    assert v["action"] == "accept"
    assert v["position"] == (110.0, 302.0)
    assert b.state == hero_belief.TRACKING


def test_a_far_jump_is_not_accepted_on_one_frame() -> None:
    """The failure this layer exists for: (68,475) reported as (295,252)."""
    b = tracked_belief()
    v = b.update([cand(295, 252)], 0.1)
    assert v["action"] == "challenge_held"
    assert v["position"] == (100.0, 300.0), "one frame must never move the car 200 px"


def test_a_far_jump_repeated_enough_becomes_a_real_move() -> None:
    b = tracked_belief()
    for _ in range(b.challenger_frames - 1):
        v = b.update([cand(295, 252)], 0.1)
        assert v["position"] == (100.0, 300.0)
    v = b.update([cand(295, 252)], 0.1)
    assert v["action"] == "takeover"
    assert v["position"] == (295.0, 252.0)


def test_a_one_frame_excursion_is_filtered_out() -> None:
    b = tracked_belief()
    b.update([cand(295, 252)], 0.1)          # one stray frame
    v = b.update([cand(101, 300)], 0.1)      # back where the car is
    assert v["action"] == "accept"
    assert v["position"] == (101.0, 300.0)
    assert b.challenger is None, "the stray challenger must be forgotten"


def test_missing_frames_decay_to_uncertain_then_lost() -> None:
    b = tracked_belief()
    assert b.update([], 0.1)["position"] is not None     # still held
    for _ in range(b.uncertain_after):
        b.update([], 0.1)
    assert b.state == hero_belief.UNCERTAIN
    for _ in range(b.lost_after):
        b.update([], 0.1)
    assert b.state == hero_belief.LOST


def test_global_search_only_after_losing_the_car() -> None:
    b = tracked_belief()
    assert b.should_search_globally() is False
    for _ in range(b.lost_after + 1):
        b.update([], 0.1)
    assert b.should_search_globally() is True


def test_search_radius_widens_with_speed() -> None:
    b = tracked_belief()
    slow = b.search_radius()
    for x in range(110, 160, 5):
        b.update([cand(x, 300)], 0.1)
    assert b.speed() > 0
    assert b.search_radius() > slow, "a moving car must get a wider window, not a tighter one"


def test_a_blob_three_times_the_remembered_size_is_downgraded() -> None:
    b = tracked_belief()
    same = b.score(cand(102, 300), 0.1)
    huge = b.score(cand(102, 300, area=810, w=96, h=51), 0.1)
    assert huge["size_score"] < same["size_score"]
    assert huge["confidence"] < same["confidence"]


def test_appearance_memory_learns_the_arrow_it_actually_sees() -> None:
    b = hero_belief.HeroBelief()
    for _ in range(6):
        b.update([cand(100, 300, s=55.0, v=150.0)], 0.1)
    assert b.sat is not None and abs(b.sat - 55.0) < 8.0


def test_position_and_heading_are_separate() -> None:
    """A frozen heading must not hold the position, and vice versa."""
    b = tracked_belief()
    v = b.update([cand(105, 300)], 0.1)
    assert v["action"] == "accept"
    assert v["heading_candidate"] is None, "the belief must not invent a heading"


def test_metrics_are_reported() -> None:
    b = tracked_belief()
    b.update([cand(295, 252)], 0.1)
    m = b.metrics()
    assert m["hero_state"] in (hero_belief.ACQUIRED, hero_belief.TRACKING,
                                hero_belief.UNCERTAIN), "a held challenge does not make it LOST"
    assert m["hero_challengers_rejected"] == 1
    assert "hero_max_jump_px" in m


def test_a_candidate_with_none_fields_does_not_crash() -> None:
    """The recovery candidates carry s/v/h as None because they come from a Pose, not a mask.

    This is the crash that killed main.py: dict.get("s", default) returns the stored None when the
    key exists, and float(None) raises. A missing colour is missing evidence, not an exception.
    """
    b = hero_belief.HeroBelief()
    b.update([cand(100, 300)], 0.1)
    broken = {"bbox": (90, 290, 20, 20), "centre": (100.0, 300.0), "area": 400,
              "width": 20, "height": 20, "aspect": 1.0, "shape_score": 0.30,
              "s_min": None, "h": None, "s": None, "v": None}
    v = b.update([broken], 0.1)
    assert v["position"] is not None


def test_a_candidate_with_none_area_does_not_crash() -> None:
    b = hero_belief.HeroBelief()
    b.update([cand(100, 300)], 0.1)
    broken = {"bbox": (90, 290, 20, 20), "centre": (101.0, 300.0), "area": None,
              "width": None, "height": None, "aspect": None, "shape_score": None,
              "s": None, "v": None}
    v = b.update([broken], 0.1)
    assert v["position"] is not None


def test_a_takeover_never_teaches_the_appearance_memory() -> None:
    """The hero being taken away: a bay light wins six frames and is committed.

    If that candidate is allowed to teach, the profile drifts to a bay light and the real hero
    stops matching. A takeover may move the position; it may not change what the hero looks like.
    """
    b = hero_belief.HeroBelief()
    b.update([cand(100, 300, area=270, w=32, h=17, s=110.0, v=160.0)], 0.1)
    b.update([cand(105, 300)], 0.1)
    before = (b.area, b.sat, b.val)
    # A challenger that would otherwise look convincing: right size, plausible shape. Only the
    # distance marks it as wrong - which is exactly the bay-light case.
    actions = [b.update([cand(600, 120, area=300, w=32, h=17, shape=0.20, s=100.0, v=150.0)],
                        0.1)["action"] for _ in range(b.challenger_frames + 4)]
    assert "takeover" in actions, actions
    # And the frames after the takeover look like ordinary tracking - they must not teach either.
    assert actions[-1] == "accept", actions
    assert (b.area, b.sat, b.val) == before, "the profile must not learn from a takeover"


def test_a_reacquisition_never_teaches_the_appearance_memory() -> None:
    """Cold start may learn; coming back from LOST may not.

    If the profile already exists, the only way position becomes None is that the hero was lost.
    Whatever is red at that moment is the wrong teacher - this is the bay light that is left when
    the hero is taken off the table.
    """
    b = hero_belief.HeroBelief()
    b.update([cand(100, 300, area=270, w=32, h=17, s=110.0, v=160.0)], 0.1)
    b.update([cand(102, 300)], 0.1)
    assert b.profile_updates > 0
    before = (b.area, b.sat, b.val)
    b.position = None                      # as after LOST
    v = b.update([cand(600, 120, area=1600, w=48, h=40, s=40.0, v=90.0)], 0.1)
    assert v["action"] in ("acquire", "acquire_pending")
    assert (b.area, b.sat, b.val) == before, "a re-acquisition must not teach the profile"


def test_a_new_target_must_hold_still_before_it_is_believed() -> None:
    """acquire_frames > 1: the first confident blob after the hero disappears is usually wrong."""
    b = hero_belief.HeroBelief(acquire_frames=3)
    assert b.update([cand(600, 120)], 0.1)["action"] == "acquire_pending"
    assert b.update([cand(602, 121)], 0.1)["action"] == "acquire_pending"
    assert b.update([cand(601, 120)], 0.1)["action"] == "acquire"
    # A different place restarts the count.
    b2 = hero_belief.HeroBelief(acquire_frames=3)
    b2.update([cand(600, 120)], 0.1)
    b2.update([cand(300, 300)], 0.1)
    assert b2.update([cand(120, 500)], 0.1)["action"] == "acquire_pending"


def test_an_established_profile_needs_more_confidence_to_change() -> None:
    b = hero_belief.HeroBelief(lock_after=3, locked_learn_confidence=0.75)
    for _ in range(4):
        b.update([cand(100, 300)], 0.1)
    assert b.profile_updates >= 3
    sat_before = b.sat
    # A weak but accepted candidate must not drag the profile.
    weak = {"bbox": (90, 290, 20, 20), "centre": (101.0, 300.0), "area": 300, "width": 20,
            "height": 20, "aspect": 1.0, "shape_score": 0.55, "s": 40.0, "v": 200.0}
    b.update([weak], 0.1)
    assert b.sat == sat_before or abs(b.sat - sat_before) < 25.0
    assert b.learning_skipped > 0


def test_a_taken_over_target_is_on_probation_before_it_can_teach() -> None:
    """The bay light does not just win one frame - the frames after it look like normal tracking.

    Without a probation window those confident accepts would drift the profile to the bay light,
    and when the hero came back it would no longer match. The target may hold the position; it may
    not become the definition of what the hero looks like.
    """
    b = hero_belief.HeroBelief(takeover_probation=10)
    b.update([cand(100, 300, area=270, w=32, h=17, s=110.0, v=160.0)], 0.1)
    before = (b.area, b.sat, b.val)
    for _ in range(b.challenger_frames + 1):
        b.update([cand(600, 120, area=300, w=32, h=17, shape=0.2, s=100.0, v=150.0)], 0.1)
    seen = [b.update([cand(600, 120, area=300, w=32, h=17, shape=0.2, s=100.0, v=150.0)],
                     0.1)["action"] for _ in range(5)]
    assert all(a in ("accept", "challenge_held") for a in seen), seen
    assert (b.area, b.sat, b.val) == before, "no learning while on probation"
    assert b.probation > 0
