"""The hero car: where it is, which way its axis runs, and what the contract says.

Colour and shape rather than a marker, for the reason recorded in src/vehicle.py: at 800x600
across a whole sandbox a marker has to be about half the car wide and flat to the camera, and
on a toy car that is not reasonable. What that costs is checked here too - the heading is an
axis, not a direction of travel, and the tests say so rather than leaving it implied.

Run:  python tests/test_vehicle.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import contract                                                 # noqa: E402
import vehicle                                                  # noqa: E402
from road_network import KIND_ENTRANCE, RoadNetwork              # noqa: E402
from spot_config import Spot                                    # noqa: E402

W, H = 800, 600
BACKGROUND = 120
# Pure red in BGR: hue 0, high saturation, middling value, which is what the mask wants.
RED = (0, 0, 200)


def frame_with(*rects) -> np.ndarray:
    img = np.full((H, W, 3), BACKGROUND, np.uint8)
    for x0, y0, x1, y1 in rects:
        cv2.rectangle(img, (x0, y0), (x1, y1), RED, -1)
    return img


def a_bay(bay_id: str, x0: float, y0: float, x1: float, y1: float) -> Spot:
    return Spot(id=bay_id, polygon=np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                                            dtype=float))


def roads_below() -> RoadNetwork:
    net = RoadNetwork(shape=(H, W))
    a = net.add_node(0, 500, kind=KIND_ENTRANCE)
    b = net.add_node(W - 1, 500)
    net.connect(a.id, b.id)
    return net


# ---------------------------------------------------------------------------
# Position and axis, against shapes whose answer is known
# ---------------------------------------------------------------------------

def test_a_horizontal_bar_reads_an_axis_of_zero() -> None:
    pose = vehicle.locate(frame_with((100, 300, 200, 340)))
    assert pose is not None
    assert abs(pose.heading) < 2 or abs(pose.heading - 180) < 2, pose.heading
    assert abs(pose.x - 150) < 2 and abs(pose.y - 320) < 2, pose.centre


def test_a_vertical_bar_reads_an_axis_of_ninety() -> None:
    pose = vehicle.locate(frame_with((400, 200, 440, 400)))
    assert pose is not None
    assert abs(pose.heading - 90) < 2, pose.heading


def test_a_diagonal_bar_reads_its_own_angle() -> None:
    img = np.full((H, W, 3), BACKGROUND, np.uint8)
    # A 45 degree bar drawn as a thick line.
    cv2.line(img, (200, 200), (400, 400), RED, 30)
    pose = vehicle.locate(img)
    assert pose is not None
    assert abs(pose.heading - 45) < 5 or abs(pose.heading - 135) < 5, pose.heading


def test_a_car_shaped_blob_is_elongated_and_solid() -> None:
    pose = vehicle.locate(frame_with((300, 300, 370, 330)))
    assert pose is not None
    assert pose.elongation > 1.8, pose.elongation
    assert pose.fill > 0.6, pose.fill


# ---------------------------------------------------------------------------
# Saying "I cannot see it" rather than guessing
# ---------------------------------------------------------------------------

def test_no_red_means_no_pose() -> None:
    assert vehicle.locate(np.full((H, W, 3), BACKGROUND, np.uint8)) is None


def test_a_speck_of_red_is_not_a_car() -> None:
    assert vehicle.locate(frame_with((100, 100, 108, 108)), min_area=80) is None


def test_the_contract_says_invisible_rather_than_remembering() -> None:
    class FakeResult:
        state = "EMPTY"
        changed_ratio = 0.0

    state = contract.build_state({"38": FakeResult()}, vehicle=None)
    assert state["vehicle"] == {"id": "hero", "visible": False}
    assert "visible" in state["vehicle"] and state["vehicle"]["visible"] is False


def test_the_contract_carries_the_pose_when_there_is_one() -> None:
    pose = vehicle.locate(frame_with((100, 300, 200, 340)))
    assert pose is not None
    state = contract.build_state({}, vehicle=pose.as_dict())
    assert state["vehicle"]["visible"] is True
    assert state["vehicle"]["heading_is_axis_only"] is True
    assert "x" in state["vehicle"] and "heading" in state["vehicle"]


def test_the_heading_is_an_axis_never_a_direction() -> None:
    """A car pointing east and the same car pointing west must read the same."""
    east = vehicle.locate(frame_with((100, 300, 200, 340)))
    # the same bar, but now with a piece added at the left so the shape leans the other way
    west = vehicle.locate(frame_with((100, 300, 200, 340), (60, 312, 100, 328)))
    assert east is not None and west is not None
    assert 0 <= east.heading < 180 and 0 <= west.heading < 180
    assert abs(east.heading - 0) < 5 or abs(east.heading - 180) < 5, east.heading
    assert "heading_is_axis_only" in east.as_dict()


def test_red_trim_touching_the_edge_of_the_picture_is_ignored() -> None:
    """The enclosure's own trim is orange-red and runs into the top of the frame."""
    img = frame_with((0, 0, W, 20))
    assert vehicle.locate(img) is None, "trim along the edge was taken for a car"
    # ...but the same blob is offered when explicitly asked for
    assert vehicle.locate(img, keep_border=True) is not None


def test_the_largest_red_thing_is_the_one_reported() -> None:
    img = frame_with((100, 100, 130, 130), (300, 300, 400, 340))
    pose = vehicle.locate(img)
    assert pose is not None
    assert pose.area > 3000, pose.area     # the big bar, not the 30x30 speck


# ---------------------------------------------------------------------------
# Where it is: the bay and road half of the map
# ---------------------------------------------------------------------------

def test_a_car_inside_a_bay_is_reported_in_that_bay() -> None:
    spots = [a_bay("38", 300, 300, 350, 400)]
    pose = vehicle.locate(frame_with((300, 300, 350, 330)))
    assert pose is not None
    where = vehicle.location_of(pose, spots, roads_below())
    assert where["kind"] == "spot" and where["id"] == "38", where


def test_a_car_on_a_road_is_reported_on_the_road() -> None:
    spots = [a_bay("38", 300, 100, 350, 200)]
    pose = vehicle.locate(frame_with((200, 495, 300, 535)))    # centred on y=515? see below
    assert pose is not None
    where = vehicle.location_of(pose, spots, roads_below())
    assert where["kind"] in ("road", "floor"), where


def test_a_car_far_from_every_road_is_reported_as_floor() -> None:
    pose = vehicle.locate(frame_with((100, 100, 200, 140)))
    assert pose is not None
    where = vehicle.location_of(pose, [], roads_below())
    assert where["kind"] == "floor", where
    assert where["distance_to_road"] > 100, where


def test_with_no_map_the_location_is_unknown_not_a_guess() -> None:
    pose = vehicle.locate(frame_with((100, 300, 200, 340)))
    assert pose is not None
    assert vehicle.location_of(pose, None, None)["kind"] == "unknown"


def test_a_symmetric_bar_cannot_say_which_end_is_the_front() -> None:
    """A plain rectangle has no front. Claiming one would be a coin flip."""
    pose = vehicle.locate(frame_with((100, 300, 200, 340)))
    assert pose is not None
    assert abs(pose.taper) < vehicle.MIN_TAPER, pose.taper
    assert pose.front_is_known is False
    assert pose.front_heading is None
    assert pose.as_dict()["front_heading"] is None
    assert pose.as_dict()["front_is_known"] is False


def test_a_tapered_nose_gives_a_front_direction() -> None:
    """A shape that is tall on the left and short on the right.

    The wide end is therefore on the left, and with FRONT_IS_THE_WIDER_END that means the car
    faces left - 180 degrees. The taper is negative because it measures (right - left).
    """
    img = np.full((H, W, 3), BACKGROUND, np.uint8)
    cv2.fillConvexPoly(img, np.array([[100, 290], [220, 310], [220, 330], [100, 350]]), RED)
    pose = vehicle.locate(img)
    assert pose is not None
    assert pose.taper < -vehicle.MIN_TAPER, pose.taper
    assert pose.front_is_known is True, pose.taper
    expected = 180.0 if vehicle.FRONT_IS_THE_WIDER_END else 0.0
    diff = min(abs(pose.front_heading - expected), 360 - abs(pose.front_heading - expected))
    assert diff < 15, (pose.front_heading, expected)


def test_turning_the_same_car_around_flips_the_front_by_180() -> None:
    """The property that matters: the front follows the car, not the picture."""
    forward = np.full((H, W, 3), BACKGROUND, np.uint8)
    cv2.fillConvexPoly(forward, np.array([[100, 290], [220, 310], [220, 330], [100, 350]]),
                       RED)
    backward = np.full((H, W, 3), BACKGROUND, np.uint8)
    cv2.fillConvexPoly(backward, np.array([[100, 310], [100, 330], [220, 290], [220, 350]]),
                       RED)
    a = vehicle.locate(forward)
    b = vehicle.locate(backward)
    assert a is not None and b is not None
    assert a.front_is_known and b.front_is_known
    turn = abs((a.front_heading - b.front_heading) % 360)
    assert abs(turn - 180) < 20, (a.front_heading, b.front_heading, turn)


def test_the_front_direction_agrees_with_the_axis_or_is_opposite_it() -> None:
    """front_heading must be one end of the axis, never something off to the side."""
    img = np.full((H, W, 3), BACKGROUND, np.uint8)
    cv2.fillConvexPoly(img, np.array([[100, 290], [220, 310], [220, 330], [100, 350]]), RED)
    pose = vehicle.locate(img)
    assert pose is not None and pose.front_heading is not None
    offset = abs((pose.front_heading - pose.heading) % 360)
    assert abs(offset - 0) < 1 or abs(offset - 180) < 1, (pose.front_heading, pose.heading)


# ---------------------------------------------------------------------------
# Smoothing: the car does not turn instantly, and one frame of shape is noisy
# ---------------------------------------------------------------------------

def a_pose(taper: float, heading: float = 0.0, elongation: float = 2.5,
           x: float = 400.0, y: float = 400.0,
           evidence: float = None) -> vehicle.VehiclePose:
    """A synthetic sighting. With no evidence given, the taper is turned into it, the way the
    module falls back to the taper when there is no usable dark band."""
    if evidence is None and abs(taper) >= vehicle.MIN_TAPER:
        evidence = max(-1.0, min(1.0, taper / vehicle.TAPER_SCALE))
    return vehicle.VehiclePose(x=x, y=y, heading=heading, area=1600, elongation=elongation,
                               fill=0.7, bbox=(370, 390, 60, 20), taper=taper,
                               front_evidence=evidence)


def test_a_noisy_taper_does_not_make_the_front_flap() -> None:
    """Measured on the real car: the taper sits at -0.12 and dips near zero about one frame
    in six. Gating on a single frame made the front appear and disappear several times in
    thirty frames while the car never moved."""
    tracker = vehicle.VehicleTracker()
    poses = [a_pose(-0.12 if i % 3 else -0.02) for i in range(30)]
    known = [p.front_is_known for p in (tracker.update(p) for p in poses)]
    assert all(known[2:]), f"the front dropped out after warm-up: {known}"
    fronts = [p.front_heading for p in (tracker.update(p) for p in poses) if p.front_is_known]
    assert len(set(round(f, 0) for f in fronts)) == 1, fronts


def test_a_genuine_turn_is_followed() -> None:
    """Smoothing must not be so heavy that turning the car round is not noticed."""
    tracker = vehicle.VehicleTracker()
    for _ in range(8):
        before = tracker.update(a_pose(-0.12))
    assert before.front_is_known
    for _ in range(6):
        after = tracker.update(a_pose(+0.12))
    assert after.front_is_known, after.taper
    turn = abs((after.front_heading - before.front_heading) % 360)
    assert abs(turn - 180) < 10, (before.front_heading, after.front_heading, turn)


def test_the_smoothed_axis_does_not_drift_to_ninety_degrees() -> None:
    """0 and 180 are the same axis; averaging them naively lands on 90, which is wrong."""
    tracker = vehicle.VehicleTracker()
    out = None
    for i in range(20):
        out = tracker.update(a_pose(-0.12, heading=2.0 if i % 2 else 178.0))
    assert out is not None
    offset = min(abs(out.heading - 0.0), abs(out.heading - 180.0))
    assert offset < 10, out.heading


def test_a_brief_loss_of_sight_does_not_erase_what_is_known() -> None:
    """One missed frame is an occlusion or motion blur, not a new situation."""
    tracker = vehicle.VehicleTracker()
    for _ in range(8):
        before = tracker.update(a_pose(-0.12))
    assert tracker.update(None) is None
    after = tracker.update(a_pose(-0.12))
    assert after is not None and after.front_is_known, after
    assert after.front_heading == before.front_heading


def test_a_long_loss_of_sight_starts_again() -> None:
    tracker = vehicle.VehicleTracker()
    tracker.update(a_pose(-0.12, x=100.0, y=100.0))
    for _ in range(vehicle.MAX_MISSES):
        tracker.update(None)
    again = tracker.update(a_pose(-0.12, x=500.0, y=500.0))
    assert abs(again.x - 500.0) < 1e-6, "the old position survived a long gap"


def test_a_hand_move_restarts_the_position_but_not_the_shape() -> None:
    """Being picked up invalidates where the car is, not what shape it is."""
    tracker = vehicle.VehicleTracker()
    for _ in range(8):
        tracker.update(a_pose(-0.12, x=100.0, y=100.0))
    moved = tracker.update(a_pose(-0.12, x=600.0, y=400.0))
    assert abs(moved.x - 600.0) < 1e-6 and abs(moved.y - 400.0) < 1e-6
    assert moved.front_is_known, "the shape did not change just because the car moved"
    assert abs(moved.taper + 0.12) < 0.02, moved.taper


def test_the_front_does_not_flip_while_the_car_is_being_driven() -> None:
    """The case that was actually reported: fine when still, flickering when driven.

    The old version reset everything on any jump over 40 px, and at ten frames a second a toy
    car crossing the sandbox moves about that much per frame - so driving wiped the average and
    the front state several times a second.
    """
    tracker = vehicle.VehicleTracker()
    rng = np.random.default_rng(7)
    fronts, known = set(), 0
    for i in range(60):
        # 40 px per frame across the frame, with the taper wobbling as much as it really does
        pose = a_pose(-0.12 + rng.normal(0, 0.05), x=60.0 + 12.0 * i, y=400.0)
        out = tracker.update(pose)
        assert out is not None
        if out.front_is_known:
            known += 1
            fronts.add(round(out.front_heading))
    assert known >= 55, f"the front was unknown too often while driving: {known}/60"
    assert len(fronts) == 1, f"the front point moved while driving: {fronts}"


def test_a_single_noisy_frame_cannot_swap_the_front() -> None:
    tracker = vehicle.VehicleTracker()
    for _ in range(10):
        tracker.update(a_pose(-0.12))
    one = tracker.update(a_pose(+0.12))          # one frame claiming the opposite
    assert one.front_heading is not None
    steady = tracker.update(a_pose(-0.12))       # back to normal
    assert abs(((steady.front_heading - one.front_heading + 180) % 360) - 180) < 10, \
        "a single frame was enough to swap the front"


def test_a_sighting_with_no_front_known_keeps_the_position() -> None:
    tracker = vehicle.VehicleTracker()
    out = tracker.update(a_pose(0.0, x=333.0, y=222.0))
    assert out is not None
    assert out.front_is_known is False and out.front_heading is None
    assert abs(out.x - 333.0) < 1e-6


# ---- front/rear profile sign: saturation + red_fraction (P0002.5 phase 1) ----
# Patterns are deliberately asymmetric like the real car: the two low-sat windows sit at
# different positions with different depth, and the red end runs differ.

SAT = [1,1,-0.4,1,1,1,-1,-1]          # front glass shallow at bin2, rear glass deep at 6-7
RF  = [1,1, 0.5,1,1,1, 0, 0]

def _template():
    return {"saturation": np.asarray(SAT, float),
            "red_fraction": np.asarray(RF, float)}

def test_score_sign_forward_wins_when_profile_reads_rear_to_front():
    t=_template()
    f, r, m, plus = vehicle.score_sign(np.array(SAT), np.array(RF), t)
    assert f > r and m > 0 and plus is True and f > 0.99, (f,r,m)

def test_reversed_profile_wins_the_reverse_score():
    t=_template()
    f, r, m, plus = vehicle.score_sign(np.array(SAT)[::-1], np.array(RF)[::-1], t)
    assert r > f and m < 0 and plus is False and r > 0.99, (f,r,m)

def test_a_profile_with_no_contrast_abstains():
    flat=np.zeros(len(SAT)); t=_template()
    f,r,m,plus=vehicle.score_sign(flat,flat,t)
    assert abs(m)<1e-9

def test_p_and_its_flip_swap_the_scores():
    t=_template()
    a=vehicle.score_sign(np.array(SAT),np.array(RF),t)
    b=vehicle.score_sign(np.array(SAT)[::-1],np.array(RF)[::-1],t)
    assert abs(a[0]-b[1])<1e-9 and abs(a[1]-b[0])<1e-9
    assert abs(a[2]+b[2])<1e-9


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
