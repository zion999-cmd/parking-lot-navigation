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

import math
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
    # 0 degrees is to the right, which is where the cabin is. The answer is an angle rather than
    # one of four names because the car can be parked at any angle, and the four-name version only
    # worked when it was level - which the rotation sweep below caught.
    assert side is not None and abs(side) < 20, f"expected the cabin to the right, got {side}"


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


# ---------------------------------------------------------------------------------------------
# Yellow is always the nose, and the front must never swing 180 degrees on its own.
#
# The axis heading is only defined in 0..180, where 0 and 180 are the same axis. Reconstructing a
# directed front from it (`heading if evidence > 0 else heading + 180`) therefore flips by 180
# whenever the smoothed axis lands on the other side of that wrap - and the paint did not move.
# Forty live frames happened to read correctly, so this was latent, not visible; the tests below
# are what make it stay fixed.
# ---------------------------------------------------------------------------------------------


def _base_car(angle_deg):
    """The yellow-nosed car, rotated by `angle_deg`, and the exact direction its nose points.

    The truth is geometry, not a measurement: the car is drawn level with the nose at the left
    (x 430..470, pointing at 180 degrees) and then rotated, so the nose ends up at
    (180 - angle) mod 360. The first version of this helper tried to recover the truth from the
    image - the direction from the car's blob towards the paint's blob - and the two centroids are
    close together, so the "truth" was noise: it read 199.8 then 292.6 for a 15-degree rotation
    step, and made a detector that was correct at every angle look wrong at twenty of them.
    """
    img = np.full((700, 900, 3), ROAD_V, np.uint8)
    cv2.rectangle(img, (430, 330), (470, 370), PAINT_BGR, -1)      # the nose, pointing left
    cv2.rectangle(img, (471, 330), (560, 370), (CABIN_V,) * 3, -1)  # the cabin
    matrix = cv2.getRotationMatrix2D((495.0, 350.0), angle_deg, 1.0)
    img = cv2.warpAffine(img, matrix, (900, 700), borderValue=(ROAD_V,) * 3)
    return cv2.GaussianBlur(img, (0, 0), 3), (180.0 - angle_deg) % 360.0


def test_the_front_follows_the_paint_all_the_way_round():
    """A full turn in 5-degree steps: the reported front must track the paint, never oppose it."""
    worst, jumps, previous = 0.0, [], None
    for angle in range(0, 360, 5):
        frame, truth = _base_car(angle)
        poses = _run(frame)
        assert poses, f"not detected at {angle} degrees"
        report = poses[0].as_dict()
        assert report["front_is_known"], f"front lost at {angle} degrees"
        error = abs((report["front_heading"] - truth + 180.0) % 360.0 - 180.0)
        worst = max(worst, error)
        assert error < 25.0, (f"front opposes the paint at {angle} degrees: "
                              f"reported {report['front_heading']}, paint {truth:.1f}")
        if previous is not None:
            jumps.append(abs((report["front_heading"] - previous + 180.0) % 360.0 - 180.0))
        previous = report["front_heading"]
    assert max(jumps) < 45.0, f"the front jumped: {max(jumps):.1f} degrees in one 5-degree step"
    assert worst < 25.0


def test_the_smoother_keeps_the_front_through_the_axis_wrap():
    """The regression itself: a smoothed axis at 179.9 and the same axis at 0.1.

    Both are the same axis, so the front must not change. Before the fix the emitted front came
    from `smoothed_heading + 180` and swung by 180 degrees when the smoothed axis crossed the wrap.
    """
    sweep = vehicle.VehicleTracker()
    front = 180.0                                   # the nose points left, always
    readings = []
    for i in range(20):
        heading = 179.7 if i % 2 else 0.3            # the same axis, either side of the wrap
        pose = vehicle.VehiclePose(x=450.0, y=420.0, heading=heading, area=2400, fill=0.75,
                                   elongation=3.0, bbox=(430, 400, 80, 40),
                                   front_evidence=-1.0, front_heading=front,
                                   front_is_known=True)
        pose = sweep.update(pose)
        if pose is not None and pose.front_heading is not None:
            readings.append(pose.front_heading)
    assert readings, "the smoother returned no front at all"
    errors = [abs((r - front + 180.0) % 360.0 - 180.0) for r in readings]
    assert max(errors) < 30.0, f"the front swung by up to {max(errors):.1f} degrees: {readings}"
    # And not just at the end: no single output may be the opposite direction.
    assert all(e < 90.0 for e in errors), readings
