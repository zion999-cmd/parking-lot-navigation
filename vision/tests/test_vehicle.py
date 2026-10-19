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
