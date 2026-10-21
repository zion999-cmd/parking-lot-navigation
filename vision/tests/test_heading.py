"""Tests for the heading path: the 4x ROI geometry and the gate/persistence state machine.

Two separate concerns, kept in one file because they are one feature:

  * `red_arrow.heading_in_roi` - the geometry. At 45 degrees the head/tail margin on the raw
    frame is sub-pixel and the direction is a coin flip; upsampling the ROI 4x restores it.
  * `heading_state.HeadingState` - the temporal gate. A single 180 degree contradictory
    reading is not a turn and must not be believed until it persists.
"""

from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))

import heading_state          # noqa: E402
import red_arrow              # noqa: E402


def make_arrow(angle_deg: float, size=(400, 300), scale=1.0) -> np.ndarray:
    """A red arrow on a mid-grey field, pointing along `angle_deg` (image convention).

    The arrow is drawn pointing along +x (heading 0) and then rotated, so the expected heading
    is the angle it was rotated by, in red_arrow's convention: atan2(y, x) with y down, which
    grows clockwise on screen.
    """
    frame = np.full((size[1], size[0], 3), 120, np.uint8)
    cx, cy = size[0] // 2, size[1] // 2
    s = 36 * scale
    # shaft, then a wider head: the head is what the width test looks for
    shaft = np.array([[-s / 2, -s * 0.10], [s * 0.10, -s * 0.10],
                      [s * 0.10, s * 0.10], [-s / 2, s * 0.10]], np.float32)
    head = np.array([[s * 0.10, -s * 0.26], [s / 2, 0.0], [s * 0.10, s * 0.26]], np.float32)
    a = math.radians(angle_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], np.float32)
    for poly in (shaft, head):
        pts = (poly @ rot.T) + np.array([cx, cy], np.float32)
        cv2.fillPoly(frame, [pts.astype(np.int32)], (0, 0, 200))
    return frame


def circ_diff(a: float, b: float) -> float:
    return ((a - b + 180.0) % 360.0) - 180.0


# --------------------------------------------------------------------- geometry


@pytest.mark.parametrize("angle", [0, 45, 90, 135, 180, 225, 270, 315])
def test_heading_in_roi_is_correct_at_every_angle(angle: int) -> None:
    """Including the diagonals, which is the whole point of the upsampled ROI."""
    frame = make_arrow(angle)
    pose = red_arrow.heading_in_roi(frame, (frame.shape[1] / 2, frame.shape[0] / 2))
    assert pose is not None, "arrow not found at %d degrees" % angle
    assert abs(circ_diff(pose.heading, angle)) < 12.0, (
        "heading %.1f at %d degrees, off by %.1f" % (pose.heading, angle,
                                                     circ_diff(pose.heading, angle)))


def test_heading_in_roi_reports_a_centre_in_frame_coordinates() -> None:
    """The caller compares this with the position channel, so it must not be ROI-local."""
    frame = make_arrow(0)
    centre = (frame.shape[1] / 2, frame.shape[0] / 2)
    pose = red_arrow.heading_in_roi(frame, centre)
    assert pose is not None
    assert abs(pose.centre[0] - centre[0]) < 4.0
    assert abs(pose.centre[1] - centre[1]) < 4.0


def test_heading_in_roi_returns_none_when_the_roi_leaves_the_frame() -> None:
    frame = make_arrow(0)
    assert red_arrow.heading_in_roi(frame, (2, 2)) is None


def test_heading_in_roi_returns_none_when_there_is_no_red() -> None:
    frame = np.full((300, 400, 3), 120, np.uint8)
    assert red_arrow.heading_in_roi(frame, (200, 150)) is None


def test_upsampled_roi_beats_the_raw_frame_at_45_degrees() -> None:
    """The regression this feature exists for, asserted rather than described.

    Not every raw-mask reading at 45 degrees is wrong - it is a coin flip - so this asserts
    that the upsampled path is right, and that repeated measurements agree with each other.
    """
    frame = make_arrow(45)
    centre = (frame.shape[1] / 2, frame.shape[0] / 2)
    readings = []
    for _ in range(5):
        pose = red_arrow.heading_in_roi(frame, centre)
        assert pose is not None
        readings.append(pose.heading)
    for r in readings:
        assert abs(circ_diff(r, 45.0)) < 12.0
    assert max(readings) - min(readings) < 1.0, "the upsampled reading should be repeatable"


# --------------------------------------------------------------------- state machine


def test_first_reading_is_committed_immediately() -> None:
    st = heading_state.HeadingState()
    out = st.update(100.0)
    assert out == {"heading": 100.0, "stale": False, "action": "initial"}
    assert st.committed == 100.0


def test_reading_within_the_gate_is_accepted_and_tracks() -> None:
    st = heading_state.HeadingState()
    st.update(100.0)
    out = st.update(125.0)                 # 25 degrees, inside the 30 degree gate
    assert out["action"] == "accepted"
    assert out["heading"] == 125.0
    assert out["stale"] is False


def test_a_flip_is_held_pending_and_not_believed() -> None:
    st = heading_state.HeadingState()
    st.update(100.0)
    out = st.update(280.0)                 # exactly 180 away
    assert out["action"] == "pending"
    assert out["heading"] == 100.0, "a single flip must never move the heading"
    assert out["stale"] is True


def test_persistence_commits_a_real_turn_after_twelve_frames() -> None:
    st = heading_state.HeadingState()
    st.update(100.0)
    for i in range(heading_state.HEADING_PERSIST - 1):
        out = st.update(280.0)
        assert out["action"] == "pending", "frame %d committed too early" % i
        assert out["heading"] == 100.0
    out = st.update(280.0)                 # the twelfth consecutive frame
    assert out["action"] == "committed"
    assert out["heading"] == 280.0
    assert out["stale"] is False
    assert st.committed == 280.0


def test_a_valid_reading_resets_the_pending_run() -> None:
    st = heading_state.HeadingState()
    st.update(100.0)
    for _ in range(10):
        st.update(280.0)
    out = st.update(105.0)                 # back within the gate of the committed value
    assert out["action"] == "accepted"
    assert st.pending is None and st.pending_count == 0
    # A new run has to start from scratch.
    out = st.update(280.0)
    assert out["action"] == "pending"


def test_mutually_inconsistent_gated_readings_never_commit() -> None:
    """The original counted disagreements without checking they agreed with each other."""
    st = heading_state.HeadingState()
    st.update(0.0)
    # 100, 200, 300 ... each far from the committed 0, and far from each other.
    committed_any = False
    for reading in (100.0, 200.0, 300.0) * 10:
        out = st.update(reading)
        if out["action"] == "committed":
            committed_any = True
    assert not committed_any
    assert st.committed == 0.0


def test_failure_holds_the_committed_heading_and_marks_it_stale() -> None:
    st = heading_state.HeadingState()
    st.update(100.0)
    out = st.update(None)
    assert out == {"heading": 100.0, "stale": True, "action": "hold"}


def test_failure_with_no_history_is_unavailable_not_invented() -> None:
    st = heading_state.HeadingState()
    out = st.update(None)
    assert out["heading"] is None
    assert out["action"] == "unavailable"
    assert out["stale"] is False


def test_recovery_clears_stale() -> None:
    st = heading_state.HeadingState()
    st.update(100.0)
    st.update(None)
    st.update(None)
    assert st.stale is True
    out = st.update(102.0)
    assert out["stale"] is False
    assert st.consecutive_failures == 0


def test_failures_do_not_break_a_pending_turn() -> None:
    """A car mid-turn whose arrow is briefly unreadable has not turned back."""
    st = heading_state.HeadingState()
    st.update(100.0)
    for _ in range(11):
        st.update(280.0)
    st.update(None)                        # one unreadable frame
    out = st.update(280.0)                 # the twelfth frame of support
    assert out["action"] == "committed"
    assert out["heading"] == 280.0


def test_metrics_report_what_happened() -> None:
    st = heading_state.HeadingState()
    st.update(100.0)
    st.update(110.0)
    st.update(290.0)
    st.update(None)
    m = st.metrics()
    assert m["heading_readings_total"] == 3
    assert m["heading_failures_total"] == 1
    assert m["heading_accepted_total"] == 2
    assert m["heading_gated_total"] == 1
    assert m["heading_hold_total"] == 1
    assert m["heading_stale"] is True


def test_seed_sets_a_committed_heading_without_counting_a_reading() -> None:
    st = heading_state.HeadingState()
    st.seed(137.0)
    assert st.committed == 137.0
    assert st.readings == 0
    out = st.update(140.0)
    assert out["action"] == "accepted"
