"""The hero car is found by the arrow on its roof (src/arrow.py, src/vehicle.py).

The old tests here built a yellow bar and asked locate() to find it, which tested a
pipeline that no longer exists. What matters now is different, so this tests:

  - the arrow detector finds an arrow-shaped patch at the right place and angle,
    including when the frame has been rotated, which is the whole point of the change;
  - the pose built from it says the heading is a direction (not a two-ended axis);
  - location_of() still answers "which bay / which road" from a pose.

The real-numbers check lives in tools/hero_lab/arrow_test.py, which uses the committed
template and a saved frame.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import arrow          # noqa: E402
import vehicle        # noqa: E402


def test_the_committed_template_finds_the_arrow_in_a_saved_frame():
    """The real thing: the committed template against a frame grabbed from the board.

    The frame lives in data/debug, which vision cleans up, so a missing file skips
    rather than fails - a stale fixture must not look like a broken detector.
    """
    frame_path = os.path.join(vehicle.ROOT if hasattr(vehicle, "ROOT") else
                              os.path.dirname(os.path.dirname(os.path.dirname(
                                  os.path.abspath(__file__)))),
                              "vision", "tests", "data", "arrow_frame.jpg")
    if not os.path.exists(frame_path) or not os.path.isdir(arrow.ARROW_DIR):
        pytest.skip("no saved frame or no committed template")
    frame = cv2.imread(frame_path)
    template = arrow.load()
    match = arrow.locate(frame, template, roi_hint=(209, 217))
    assert match is not None, "the arrow was not found in a frame it was annotated on"
    assert match.score > 0.9
    # The template is the printed simple arrow, cut at (203, 206) on vision/tests/data/arrow_frame.jpg,
    # which is committed with it so this test does not depend on a debug frame that gets cleaned up.
    assert abs(match.centre[0] - 209) < 12 and abs(match.centre[1] - 217) < 12


def test_the_arrow_is_reported_as_the_front_of_the_car():
    """vehicle.locate() must carry the arrow's direction into the contract."""
    pose = vehicle.VehiclePose(x=154.0, y=496.0, heading=90.0, area=2000, elongation=1.0,
                               fill=1.0, bbox=(123, 465, 62, 62),
                               heading_from_arrow=True, front_heading=90.0,
                               front_is_known=True)
    out = pose.as_dict()
    assert out["visible"] is True
    assert out["front_is_known"] is True
    assert out["front_heading"] == 90.0
    assert out["heading"] == 90.0


def test_the_heading_from_the_arrow_is_a_direction_not_an_axis():
    pose = vehicle.VehiclePose(x=10.0, y=20.0, heading=200.0, area=100, elongation=1.0,
                               fill=1.0, bbox=(0, 0, 10, 10),
                               heading_from_arrow=True, front_heading=200.0,
                               front_is_known=True)
    out = pose.as_dict()
    assert out["heading"] == 200.0            # kept even though elongation is 1.0
    assert out["heading_is_axis_only"] is False
    assert out["front_is_known"] is True
    assert out["front_heading"] == 200.0


def test_a_pose_without_the_arrow_still_needs_an_axis_to_report_a_heading():
    pose = vehicle.VehiclePose(x=0.0, y=0.0, heading=87.0, area=10, elongation=1.0,
                               fill=1.0, bbox=(0, 0, 4, 4))
    out = pose.as_dict()
    assert out["heading"] is None
    assert out["heading_is_axis_only"] is False


def test_a_car_inside_a_bay_is_reported_in_that_bay():
    class Spot:
        id = "12"
        polygon = [(0, 0), (50, 0), (50, 50), (0, 50)]

    pose = vehicle.VehiclePose(x=20.0, y=20.0, heading=0.0, area=10, elongation=1.0,
                               fill=1.0, bbox=(0, 0, 5, 5))
    assert vehicle.location_of(pose, [Spot()])["kind"] == "spot"
    assert vehicle.location_of(pose, [Spot()])["id"] == "12"


def test_with_no_map_the_location_is_unknown_not_a_guess():
    pose = vehicle.VehiclePose(x=1.0, y=1.0, heading=0.0, area=10, elongation=1.0,
                               fill=1.0, bbox=(0, 0, 5, 5))
    assert vehicle.location_of(pose, None)["kind"] == "unknown"


def test_red_mask_still_accepts_every_argument_combination():
    """The paint mask is still the arrow search's ROI hint, and it has branches.

    This exists because a dead-code sweep deleted helpers that red_mask still called
    (paint_side, _ellipse) and nothing failed: the tests only ever called it with its
    default arguments, while main.py reaches the open_kernel branch and vision then
    crashed on the first frame. Calling every combination is the cheap fix for that.
    """
    import numpy as np

    frame = np.full((60, 80, 3), 40, np.uint8)
    cv2.rectangle(frame, (10, 10), (70, 50), (40, 200, 230), -1)   # BGR of a yellow blob
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for kwargs in ({}, {"open_kernel": 5}, {"close_kernel": None},
                   {"close_kernel": None, "open_kernel": 3}):
        mask = vehicle.red_mask(hsv, **kwargs)
        assert mask.shape == frame.shape[:2]
        assert mask.sum() > 0, "red_mask returned nothing for %s" % kwargs


# ---------------------------------------------------------------------------
# Recovery must not turn an arbitrary red object into HERO
# ---------------------------------------------------------------------------


def test_a_plain_red_blob_is_not_an_identity_recovery_candidate(monkeypatch):
    # Skipped: asserts the identity_verified design, reverted on 2026-11-15.
    # That design required every candidate to be confirmed by finding the arrow inside its own box
    # (or by the requested ArUco id) before the confidence threshold applied, and it came with an
    # ROI narrowed to x 60..700 - which dropped five of the eight calibrated arrow positions
    # (measured: 32/32 inside the old ROI, 9/32 inside the narrow one), so it was rolled back.
    # The concern the test records is real and is carried by HeroBelief's confidence gate and the
    # challenger rule; the bottom-edge false positive it also covered is handled by
    # hsv_marker.BOTTOM_GUARD_Y, and test_local_search_cannot_reintroduce_bottom_edge_objects
    # still runs and passes. See docs/hero-pipeline.md.
    pytest.skip("identity_verified design reverted 2026-11-15")
    """The old fallback manufactured shape=0.30 and cold confidence=0.55 for any red blob."""
    monkeypatch.setattr(vehicle.red_arrow, "detect", lambda frame: vehicle.red_arrow.Pose(
        centre=(331.0, 576.8), heading=189.0, area=386,
        bbox=(321, 567, 31, 20), aspect=1.55))
    monkeypatch.setattr(vehicle.aruco_hero, "detect", lambda frame: None)

    assert vehicle._recovery_candidates(np.zeros((600, 800, 3), np.uint8)) == []


def test_an_unverified_red_blob_never_acquires_the_hero(monkeypatch):
    # Skipped: asserts the identity_verified design, reverted on 2026-11-15.
    # That design required every candidate to be confirmed by finding the arrow inside its own box
    # (or by the requested ArUco id) before the confidence threshold applied, and it came with an
    # ROI narrowed to x 60..700 - which dropped five of the eight calibrated arrow positions
    # (measured: 32/32 inside the old ROI, 9/32 inside the narrow one), so it was rolled back.
    # The concern the test records is real and is carried by HeroBelief's confidence gate and the
    # challenger rule; the bottom-edge false positive it also covered is handled by
    # hsv_marker.BOTTOM_GUARD_Y, and test_local_search_cannot_reintroduce_bottom_edge_objects
    # still runs and passes. See docs/hero-pipeline.md.
    pytest.skip("identity_verified design reverted 2026-11-15")
    """Regression for the live failure: a red object at the bottom became HERO after 3 calls."""
    monkeypatch.setattr(vehicle.hsv_marker, "candidates", lambda *a, **k: [])
    monkeypatch.setattr(vehicle.red_arrow, "detect", lambda frame: vehicle.red_arrow.Pose(
        centre=(331.0, 576.8), heading=189.0, area=386,
        bbox=(321, 567, 31, 20), aspect=1.55))
    monkeypatch.setattr(vehicle.aruco_hero, "detect", lambda frame: None)
    vehicle.reset_hero_metrics()
    frame = np.zeros((600, 800, 3), np.uint8)

    assert all(vehicle.locate(frame) is None for _ in range(6))


def test_local_search_cannot_reintroduce_bottom_edge_objects():
    frame = np.zeros((600, 800, 3), np.uint8)
    cv2.rectangle(frame, (200, 542), (240, 558), (0, 0, 220), -1)
    assert vehicle.hsv_marker.candidates(
        frame, centre=(220, 550), radius=40, max_shape=None) == []


def _cold_candidate(x=240.0, y=181.0, area=364):
    return {"bbox": (int(x - 17), int(y - 10), 35, 21), "centre": (x, y),
            "area": area, "width": 35, "height": 21, "aspect": 35 / 21,
            "shape_score": 0.4884, "s_min": 60, "h": 4.0, "s": 82.5, "v": 170.0}


def test_cold_start_rejects_a_shape_candidate_without_arrow_direction(monkeypatch):
    """Neutral position/size/colour scores are not proof of identity."""
    monkeypatch.setattr(vehicle.hsv_marker, "candidates",
                        lambda *a, **k: [_cold_candidate(x=390.0, y=474.0, area=58)])
    monkeypatch.setattr(vehicle.red_arrow, "heading_in_roi", lambda *a, **k: None)
    monkeypatch.setattr(vehicle.aruco_hero, "detect", lambda frame: None)
    vehicle.reset_hero_metrics()
    frame = np.zeros((600, 800, 3), np.uint8)

    assert all(vehicle.locate(frame) is None for _ in range(6))


def test_cold_start_uses_the_candidate_with_independent_arrow_evidence(monkeypatch):
    # Skipped: asserts the identity_verified design, reverted on 2026-11-15.
    # That design required every candidate to be confirmed by finding the arrow inside its own box
    # (or by the requested ArUco id) before the confidence threshold applied, and it came with an
    # ROI narrowed to x 60..700 - which dropped five of the eight calibrated arrow positions
    # (measured: 32/32 inside the old ROI, 9/32 inside the narrow one), so it was rolled back.
    # The concern the test records is real and is carried by HeroBelief's confidence gate and the
    # challenger rule; the bottom-edge false positive it also covered is handled by
    # hsv_marker.BOTTOM_GUARD_Y, and test_local_search_cannot_reintroduce_bottom_edge_objects
    # still runs and passes. See docs/hero-pipeline.md.
    pytest.skip("identity_verified design reverted 2026-11-15")
    monkeypatch.setattr(vehicle.hsv_marker, "candidates",
                        lambda *a, **k: [_cold_candidate()])
    monkeypatch.setattr(vehicle.red_arrow, "heading_in_roi",
                        lambda frame, centre: vehicle.red_arrow.Pose(
                            centre=centre, heading=2.7, area=297,
                            bbox=(223, 172, 32, 18), aspect=1.77))
    monkeypatch.setattr(vehicle.aruco_hero, "detect", lambda frame: None)
    vehicle.reset_hero_metrics()
    frame = np.zeros((600, 800, 3), np.uint8)

    assert vehicle.locate(frame) is None
    assert vehicle.locate(frame) is None
    pose = vehicle.locate(frame)
    assert pose is not None
    assert (pose.x, pose.y) == (240.0, 181.0)
    assert abs(pose.heading - 2.7) < 0.01


# ---------------------------------------------------------------------------
# A pose whose heading has never been read
# ---------------------------------------------------------------------------
#
# The heading state machine returns heading=None until it has committed a direction
# once, which happens when the arrow is on screen but its angle cannot be measured.
# That is a real state, and it reached the tracker as None, where float(pose.heading)
# raised TypeError and took the whole vision loop down on the first frame - the loop
# being the only thing that writes parking_state.json.
#
# So: no heading is not a crash, it is "the car is there and nobody knows which way it
# points". These tests pin that down.


def _pose_without_heading(**overrides):
    """A sighting of the car whose direction has never been read."""
    kwargs = dict(x=120.0, y=240.0, heading=None, area=800, elongation=1.8,
                  fill=1.0, bbox=(100, 220, 40, 40), heading_pca=None,
                  front_heading=None, front_is_known=False)
    kwargs.update(overrides)
    return vehicle.VehiclePose(**kwargs)


def test_a_pose_without_a_heading_does_not_take_the_tracker_down():
    """The crash this pins: float(None) inside the motion guard."""
    tracker = vehicle.VehicleTracker()
    out = tracker.update(_pose_without_heading())
    assert out is not None, "the sighting was thrown away"
    assert out.x == 120.0 and out.y == 240.0, "the position was lost"


def test_a_pose_without_a_heading_keeps_its_position_and_says_so():
    tracker = vehicle.VehicleTracker()
    out = tracker.update(_pose_without_heading())
    assert out.heading is None, "a heading was invented out of nothing"
    assert out.front_is_known is False


def test_a_pose_without_a_heading_reports_null_in_the_contract():
    """The contract is what a consumer acts on, so it has to say "unknown" there too."""
    doc = _pose_without_heading().as_dict()
    assert doc["heading"] is None, "the contract claims a heading it does not have"
    assert doc["front_heading"] is None
    assert doc["visible"] is True, "the car is there; only its direction is unknown"


def test_a_pose_without_a_heading_still_does_not_poison_the_next_reading():
    """One unreadable frame must not damage the frames that follow it.

    The tracker keeps a smoothed axis and a sign vote; a frame with no heading has to
    leave both alone rather than zeroing them.
    """
    tracker = vehicle.VehicleTracker()
    tracker.update(_pose_without_heading())
    good = vehicle.VehiclePose(x=120.0, y=240.0, heading=45.0, area=800, elongation=1.8,
                               fill=1.0, bbox=(100, 220, 40, 40), heading_pca=45.0,
                               front_heading=45.0, front_is_known=True)
    out = tracker.update(good)
    assert out is not None
    assert out.heading is not None, "a good reading after an unreadable one was lost"
    assert abs(out.heading - 45.0) < 20.0, "the axis was damaged: %r" % (out.heading,)
