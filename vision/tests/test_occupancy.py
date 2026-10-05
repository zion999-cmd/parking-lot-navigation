"""Occupancy maths, tested with synthetic images.

No camera, no network: every case is built as a numpy array so the logic is
pinned down independently of what the sandbox happens to look like today.

Run either way:
    python tests/test_occupancy.py
    pytest tests/test_occupancy.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from occupancy import EMPTY, OCCUPIED, OccupancyDetector, OccupancyParams  # noqa: E402
from spot_config import Spot                                               # noqa: E402

W, H = 400, 300
BAY = [(100, 100), (200, 100), (200, 200), (100, 200)]
INNER_SCALE = 0.6


def make_spot(spot_id: str = "38", polygon=None, inner_scale: float = INNER_SCALE) -> Spot:
    s = Spot(id=spot_id, polygon=np.asarray(polygon or BAY, dtype=np.float64),
             inner_scale=inner_scale)
    s.prepare((H, W))
    return s


def base_frame() -> np.ndarray:
    """A flat 'car park floor' with a bright border, standing in for a bay."""
    frame = np.full((H, W, 3), 110, dtype=np.uint8)
    frame[90:210, 90:210] = 150            # the bay is slightly lighter
    frame[95:105, 95:205] = 240            # top white bay line
    frame[205:212, 95:205] = 240           # bottom white bay line
    return frame


def draw_car(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int,
             colour=(40, 40, 40)) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (x0, y0), (x1, y1), colour, -1)
    return out


def detector_for(frame: np.ndarray, spot: Spot, **kwargs) -> OccupancyDetector:
    params = OccupancyParams(**kwargs)
    d = OccupancyDetector(params)
    d.set_reference(frame, [spot])
    return d


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

def test_inner_polygon_is_a_centred_shrink():
    s = make_spot()
    inner = s.inner()
    assert np.allclose(inner.mean(axis=0), s.centroid(), atol=1e-6)
    # scale 0.6 about the centroid -> area scales by 0.36
    assert abs(s.inner_area() / s.area() - INNER_SCALE ** 2) < 0.05, \
        (s.inner_area(), s.area())
    # and it stays strictly inside the bay
    xs, ys = s.inner_points[:, 0], s.inner_points[:, 1]
    assert xs.min() > 100 and xs.max() < 200 and ys.min() > 100 and ys.max() < 200


def test_explicit_detect_polygon_overrides_scale():
    s = make_spot()
    s.detect_polygon = np.asarray([[120, 120], [180, 120], [180, 180], [120, 180]],
                                  dtype=np.float64)
    s.prepare((H, W))
    assert s.inner_area() == 60 * 60


# --------------------------------------------------------------------------- #
# the actual occupancy decision
# --------------------------------------------------------------------------- #

def test_identical_frame_is_empty():
    frame = base_frame()
    spot = make_spot()
    d = detector_for(frame, spot)
    r = d.evaluate_spot(frame, spot)
    assert r.changed_ratio == 0.0, r
    assert r.state == EMPTY


def test_car_inside_the_inner_area_is_occupied():
    frame = base_frame()
    spot = make_spot()
    d = detector_for(frame, spot)
    car = draw_car(frame, 130, 130, 170, 170)          # well inside the inner area
    r = d.evaluate_spot(car, spot)
    assert r.changed_ratio > 0.3, r
    assert r.state == OCCUPIED, r


def test_paint_on_the_bay_line_does_not_matter():
    """Requirement: do not detect the bay's own markings."""
    frame = base_frame()
    spot = make_spot()
    d = detector_for(frame, spot)
    # scribble over both white lines and the bay number area, all outside the
    # inner detection polygon but inside the bay
    scribble = frame.copy()
    scribble[95:105, 95:205] = 0        # top line blacked out
    scribble[205:212, 95:205] = 0       # bottom line blacked out
    scribble[112:120, 104:120] = 255    # fake bay number
    r = d.evaluate_spot(scribble, spot)
    assert r.changed_ratio == 0.0, f"bay markings leaked into the ROI: {r}"
    assert r.state == EMPTY, r


def test_change_in_the_bay_corner_but_outside_the_polygon_is_ignored():
    """Proves the detection ROI is the polygon, not its bounding box.

    Needs a non-rectangular bay: for a rectangle the polygon and its bounding box
    are the same shape, so the test would prove nothing. A diamond-shaped bay has
    bbox corners that are outside the polygon.
    """
    frame = base_frame()
    diamond = [(150, 90), (210, 150), (150, 210), (90, 150)]
    spot = make_spot(polygon=diamond, inner_scale=0.7)
    d = detector_for(frame, spot)

    xs = spot.inner_points[:, 0]
    ys = spot.inner_points[:, 1]
    x0, y0 = int(xs.min()), int(ys.min())          # a bbox corner of the inner ROI
    corner = draw_car(frame, x0 + 1, y0 + 1, x0 + 6, y0 + 6)
    r = d.evaluate_spot(corner, spot)
    assert r.changed_ratio == 0.0, f"bbox corner leaked into the ROI: {r}"
    assert r.state == EMPTY

    # ...while a change on the diamond's centre is seen
    centre = draw_car(frame, 140, 140, 160, 160)      # 441 px of a 3528 px2 inner area
    r_centre = d.evaluate_spot(centre, spot)
    assert r_centre.changed_ratio > 0.08, r_centre


def test_below_threshold_noise_stays_empty():
    frame = base_frame()
    spot = make_spot()
    d = detector_for(frame, spot)
    rng = np.random.default_rng(7)
    noisy = frame.copy()
    roi = noisy[100:200, 100:200].astype(np.int16)
    noise = rng.integers(-8, 9, roi.shape, dtype=np.int16)
    noisy[100:200, 100:200] = np.clip(roi + noise, 0, 255).astype(np.uint8)
    r = d.evaluate_spot(noisy, spot)
    assert r.changed_ratio < 0.02, f"JPEG-like noise pushed changed_ratio to {r.changed_ratio}"
    assert r.state == EMPTY


def test_partial_occupancy_lands_between():
    frame = base_frame()
    spot = make_spot()
    d = detector_for(frame, spot)
    # a thin sliver covering roughly a tenth of the inner area
    car = draw_car(frame, 130, 132, 170, 136)
    r = d.evaluate_spot(car, spot)
    assert 0.02 < r.changed_ratio < 0.15, r
    assert r.state == EMPTY


def test_threshold_is_respected():
    frame = base_frame()
    spot = make_spot()
    car = draw_car(frame, 140, 140, 160, 160)
    strict = detector_for(frame, spot, changed_ratio_threshold=0.9)
    loose = detector_for(frame, spot, changed_ratio_threshold=0.01)
    assert strict.evaluate_spot(car, spot).state == EMPTY
    assert loose.evaluate_spot(car, spot).state == OCCUPIED


def test_pixel_threshold_controls_sensitivity():
    frame = base_frame()
    spot = make_spot()
    # the bay floor is grey 150, so this car is only ~28 levels different
    car = draw_car(frame, 140, 140, 160, 160, colour=(178, 178, 178))
    low = detector_for(frame, spot, pixel_threshold=5)
    high = detector_for(frame, spot, pixel_threshold=60)
    assert low.evaluate_spot(car, spot).changed_ratio > 0.1
    assert high.evaluate_spot(car, spot).changed_ratio == 0.0


def test_larger_inner_scale_sees_more_of_a_marginal_change():
    frame = base_frame()
    slim = make_spot(inner_scale=0.4)
    fat = make_spot(inner_scale=0.85)
    # change hugging the bay edge: a bigger detection area includes more of it
    car = draw_car(frame, 108, 108, 130, 190)
    r_slim = detector_for(frame, slim).evaluate_spot(car, slim)
    r_fat = detector_for(frame, fat).evaluate_spot(car, fat)
    assert r_fat.changed_ratio > r_slim.changed_ratio


def test_resolution_change_is_refused_not_silently_wrong():
    frame = base_frame()
    spot = make_spot()
    d = detector_for(frame, spot)
    resized = cv2.resize(frame, (W // 2, H // 2))
    try:
        d.evaluate_spot(resized, spot)
    except RuntimeError as exc:
        assert "geometry changed" in str(exc)
    else:
        raise AssertionError("should have refused a frame with different geometry")


def test_multiple_spots_are_independent():
    frame = base_frame()
    a = make_spot("38")
    b = make_spot("39", polygon=[(210, 100), (310, 100), (310, 200), (210, 200)])
    d = OccupancyDetector(OccupancyParams())
    d.set_reference(frame, [a, b])
    car_in_b = draw_car(frame, 240, 130, 280, 170)
    results = d.evaluate(car_in_b, [a, b])
    assert results["38"].state == EMPTY, results["38"]
    assert results["39"].state == OCCUPIED, results["39"]


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:                     # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok    {name}")
    print(f"\n{len(tests)} checks, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
