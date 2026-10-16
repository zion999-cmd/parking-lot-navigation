"""The annotated frame must draw for every shape the detector can return.

This file exists because of a specific failure: the axis line was wrapped in an
`elongation >= AXIS_MIN_ELONGATION` condition (right - a line through a square is a claim the
mask cannot support), but the two end-rings below it still used the axis vector it computed,
so the moment a car had no axis `annotate()` raised

    UnboundLocalError: cannot access local variable 'dx'

and main.py would not start at all. 339 tests passed while that was true, because nothing
called the drawing code - the detector and the contract were covered, the picture was not.

Two shapes, because the branch that crashed is the one the elongated shape does not reach.

Run:  python -m pytest tests/test_annotate.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import main as m                                                   # noqa: E402
import vehicle                                                     # noqa: E402
from spot_config import Spot                                       # noqa: E402

W, H = 800, 600
BACKDROP = 120
YELLOW_BGR = (0, 214, 255)

# The yellow car's paint, pinned here so this file does not depend on which car is installed -
# the same trap tests/test_vehicle.py fell into.
YELLOW_CAR = dict(
    vehicle._DEFAULT_VEHICLE_CONFIG,
    hue_ranges=[[[23, 90, 70], [33, 255, 255]]],
)

SQUARE = [[400, 300], [430, 300], [430, 330], [400, 330]]          # elongation 1.0
BAR = [[400, 300], [480, 300], [480, 330], [400, 330]]             # elongation 2.6

SPOTS = [Spot("9", [(10, 10), (60, 10), (60, 60), (10, 60)])]


def _frame(points):
    img = np.full((H, W, 3), BACKDROP, np.uint8)
    cv2.fillPoly(img, [np.array(points, np.int32)], YELLOW_BGR)
    return img


def _annotate(points, config):
    """Draw one synthetic frame, putting the vehicle config back afterwards.

    Restoring it is not tidiness. tests/test_vehicle.py pins the same module-level cache to
    the red car at import time, and pytest imports both files into one process: setting this
    one at module level - which is how it was written first - meant whichever file was
    imported last won, and test_vehicle.py failed 16 tests because the car it draws was no
    longer the colour its own pin said it was. State shared between test files has to be
    borrowed and returned.
    """
    previous = vehicle._VEHICLE_CONFIG_CACHE
    vehicle._VEHICLE_CONFIG_CACHE = config
    try:
        frame = _frame(points)
        poses = vehicle.locate_all(frame)
        assert poses, "the synthetic car was not detected at all"
        canvas = m.annotate(frame, SPOTS, {}, False, False, poses[0], None)
        assert canvas.shape == (H, W, 3)
        return poses[0], canvas
    finally:
        vehicle._VEHICLE_CONFIG_CACHE = previous


def test_a_shape_with_no_axis_still_draws():
    pose, _ = _annotate(SQUARE, YELLOW_CAR)
    assert pose.elongation < vehicle.AXIS_MIN_ELONGATION
    assert pose.as_dict()["heading"] is None, "an axisless shape must not report an angle"


def test_a_shape_with_an_axis_still_draws():
    pose, _ = _annotate(BAR, YELLOW_CAR)
    assert pose.elongation >= vehicle.AXIS_MIN_ELONGATION
    assert isinstance(pose.as_dict()["heading"], float)


def test_a_known_front_draws_without_an_axis():
    """The front markers use their own direction, so they cannot depend on the axis vector."""
    pose, _ = _annotate(SQUARE, dict(YELLOW_CAR, front_is_the_wider_end=False))
    assert pose.elongation < vehicle.AXIS_MIN_ELONGATION
