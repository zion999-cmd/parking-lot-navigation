"""The car's other half, found by asking which side of the paint is dark.

The hero car is a yellow nose with a black cabin, so the colour mask can only see the nose. What
works - after four measured failures with other ideas, recorded in config/vehicle.json - is not to
look for dark pixels in general, but to sample the four sides of the paint's own bounding box and
take the one that is dark. Measured on the real car: that side reads 108 grey levels below the road
while the other three read 5, 8 and 13, and the answer did not change with the threshold.

The front follows from the same fact: if the paint is one end of the car, the painted end is the
nose. That is a statement about this car from the person who owns it, in the same category as
front_is_the_wider_end - not something the pixels can work out, and it must NOT be applied to a car
that is painted along its whole length, which has no painted end.

Run:  python -m pytest tests/test_paint_side.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import vehicle                                                     # noqa: E402

ROAD_V = 183
PAINT_BGR = (0, 185, 220)          # V=220, hue 25 in OpenCV's 0..180 - inside the yellow range
CABIN_V = 78                       # the cabin as measured on the real car

YELLOW_CAR = dict(
    vehicle._DEFAULT_VEHICLE_CONFIG,
    hue_ranges=[[[23, 90, 70], [33, 255, 255]]],
    open_kernel=3, close_kernel=30, min_area=80,
    attach_paint_side=True, front_is_the_painted_end=True,
)


def _frame(paint_box, cabin_box, cabin_v=CABIN_V):
    img = np.full((600, 800, 3), ROAD_V, np.uint8)
    cv2.rectangle(img, paint_box[:2], paint_box[2:], PAINT_BGR, -1)
    if cabin_box is not None:
        cv2.rectangle(img, cabin_box[:2], cabin_box[2:], (cabin_v,) * 3, -1)
    return cv2.GaussianBlur(img, (0, 0), 3)


def _run(img, config=YELLOW_CAR):
    previous = vehicle._VEHICLE_CONFIG_CACHE
    vehicle._VEHICLE_CONFIG_CACHE = config
    try:
        return vehicle.locate_all(img)
    finally:
        vehicle._VEHICLE_CONFIG_CACHE = previous


NOSE = (400, 300, 440, 340)


def test_the_dark_side_of_the_paint_is_found():
    """Cabin to the right of the nose, the layout the real car has."""
    previous = vehicle._VEHICLE_CONFIG_CACHE
    vehicle._VEHICLE_CONFIG_CACHE = YELLOW_CAR
    try:
        hsv = cv2.cvtColor(_frame(NOSE, (441, 300, 530, 340)), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([23, 90, 70], np.uint8), np.array([33, 255, 255], np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vehicle._ellipse(1))
        side = vehicle.paint_side(mask, hsv[:, :, 2])
    finally:
        vehicle._VEHICLE_CONFIG_CACHE = previous
    assert side == "right", f"expected the cabin on the right, got {side}"


def test_the_whole_car_and_its_front_come_back():
    """One reading, three things: the centre, a real axis, and the front pointing at the paint."""
    poses = _run(_frame(NOSE, (441, 300, 530, 340)))
    assert poses, "the car was not detected"
    pose = poses[0]
    report = pose.as_dict()
    assert pose.area > 2_500, f"the cabin is missing: area {pose.area}"
    assert pose.elongation > 2.0, f"no axis: elongation {pose.elongation:.2f}"
    assert abs(pose.x - 465) < 15, f"centre should be the car's, not the nose's: {pose.x}"
    assert report["heading"] is not None
    assert report["front_is_known"] is True
    # The paint is on the left, so the front points left - 180 degrees.
    assert abs(report["front_heading"] - 180.0) < 20, report["front_heading"]


def test_turning_the_car_around_turns_the_front_with_it():
    """Paint on the other side: the front must follow, because it is defined by the paint."""
    poses = _run(_frame((490, 300, 530, 340), (400, 300, 489, 340)))
    assert poses
    report = poses[0].as_dict()
    assert report["front_is_known"] is True
    assert report["front_heading"] < 20 or report["front_heading"] > 340, report["front_heading"]


def test_a_car_with_no_dark_side_stays_the_paint():
    """No side is dark: this is not the two-tone car, so nothing is invented."""
    poses = _run(_frame(NOSE, None))
    assert poses
    pose = poses[0]
    assert pose.area < 2_500, f"grew with nothing to grow into: {pose.area}"
    assert pose.elongation < 1.6, "a square patch has no axis"


def test_a_cabin_the_same_brightness_as_the_road_is_not_a_cabin():
    """The failure the brightness attempts kept hitting: no contrast, no cabin."""
    poses = _run(_frame(NOSE, (441, 300, 530, 340), cabin_v=ROAD_V))
    assert poses
    assert poses[0].area < 2_500, f"a cabin the colour of the road was accepted: {poses[0].area}"
