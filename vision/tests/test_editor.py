"""Editing model and GUI wiring, tested without a display.

``CalibrationApp`` keeps its mouse and key handling in ``on_mouse()`` and
``act()``, so the interaction can be driven from a test by calling those directly
with synthetic events. That covers everything except the pixels themselves.

Run:  python tests/test_editor.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from calibration import CalibrationApp                        # noqa: E402
from detector import DetectorParams                           # noqa: E402
from spot_config import load_spots, save_spots                # noqa: E402
from spot_editor import SpotEditor                            # noqa: E402

W, H = 900, 600


def lot_frame(rows: int = 2, bays: int = 6, bay_w: int = 70, bay_h: int = 140,
              x0: int = 90, y0: int = 80, row_gap: int = 40) -> np.ndarray:
    img = np.full((H, W, 3), 120, np.uint8)
    for r in range(rows):
        ry0 = y0 + r * (bay_h + row_gap)
        ry1 = ry0 + bay_h
        xs = [x0 + i * bay_w for i in range(bays + 1)]
        for x in xs:
            cv2.line(img, (x, ry0), (x, ry1), (245, 245, 245), 3)
        cv2.line(img, (xs[0], ry0), (xs[-1], ry0), (245, 245, 245), 2)
        cv2.line(img, (xs[0], ry1), (xs[-1], ry1), (245, 245, 245), 2)
    return img


def editor() -> SpotEditor:
    return SpotEditor((H, W), start_id=38)


def app_with(frame=None) -> CalibrationApp:
    ed = editor()
    app = CalibrationApp(ed, "/tmp/does-not-matter.json")
    app.detector_params = DetectorParams()
    app._frame = frame if frame is not None else lot_frame()
    return app


# --------------------------------------------------------------------------- #
# basic edits
# --------------------------------------------------------------------------- #

def test_add_creates_a_prepared_spot_with_the_next_id():
    ed = editor()
    spot = ed.add_at(400, 300)
    assert spot is not None and spot.id == "38"
    assert ed.selected == 0
    assert spot.inner_points is not None and spot.inner_area() > 0
    assert ed.dirty


def test_ids_continue_after_existing_ones():
    ed = editor()
    ed.add_at(200, 200)
    ed.add_at(400, 200)
    ed.rename_selected("40")
    third = ed.add_at(600, 200)
    assert third.id == "41", third.id


def test_hit_test_prefers_corners_then_body():
    ed = editor()
    spot = ed.add_at(400, 300)
    cx, cy = spot.polygon.mean(axis=0)
    assert ed.hit_test(cx, cy) == (0, None)                 # inside -> body
    x0, y0 = spot.polygon[0]
    assert ed.hit_test(x0 + 3, y0 + 3) == (0, 0)            # near corner 0
    assert ed.hit_test(5, 5) == (None, None)                # empty space


def test_drag_a_corner_moves_only_that_corner():
    ed = editor()
    spot = ed.add_at(400, 300)
    before = spot.polygon.copy()
    x0, y0 = spot.polygon[0]
    ed.begin_drag(x0, y0)
    ed.update_drag(x0 - 20, y0 - 15)
    ed.end_drag()
    moved = ed.spots[0].polygon
    assert np.allclose(moved[0], [x0 - 20, y0 - 15])
    for c in (1, 2, 3):
        assert np.allclose(moved[c], before[c]), f"corner {c} moved unexpectedly"


def test_drag_the_body_moves_every_corner():
    ed = editor()
    spot = ed.add_at(400, 300)
    before = spot.polygon.copy()
    cx, cy = spot.polygon.mean(axis=0)
    ed.begin_drag(cx, cy)
    ed.update_drag(cx + 30, cy + 12)
    ed.end_drag()
    delta = ed.spots[0].polygon - before
    assert np.allclose(delta, [30, 12]), delta


def test_undo_restores_after_a_drag():
    ed = editor()
    spot = ed.add_at(400, 300)
    before = spot.polygon.copy()
    x0, y0 = spot.polygon[0]
    ed.begin_drag(x0, y0)
    ed.update_drag(x0 + 40, y0 + 40)
    ed.end_drag()
    assert not np.allclose(ed.spots[0].polygon, before)
    assert ed.undo()
    assert np.allclose(ed.spots[0].polygon, before), "one undo should undo one drag"


def test_collapsing_a_spot_removes_it():
    ed = editor()
    spot = ed.add_at(400, 300)
    x0, y0 = spot.polygon[0]
    ed.begin_drag(x0, y0)
    ed.update_drag(spot.polygon[2][0], spot.polygon[2][1])   # fold it flat
    ed.end_drag()
    assert len(ed.spots) == 0


def test_delete_and_clear_and_undo():
    ed = editor()
    ed.add_at(200, 200)
    ed.add_at(400, 200)
    assert ed.delete_selected()
    assert len(ed.spots) == 1
    ed.clear()
    assert len(ed.spots) == 0
    ed.undo()
    assert len(ed.spots) == 1
    ed.undo()
    assert len(ed.spots) == 2


def test_rename_rejects_a_duplicate():
    ed = editor()
    ed.add_at(200, 200)
    ed.add_at(400, 200)
    assert ed.rename_selected("77")
    assert ed.spots[1].id == "77"
    ed.select(0)
    assert not ed.rename_selected("77"), "duplicate id must be refused"
    assert ed.spots[0].id == "38"


def test_inner_scale_changes_the_detection_area_and_is_range_checked():
    ed = editor()
    spot = ed.add_at(400, 300)
    before = spot.inner_area()
    assert ed.set_inner_scale(0.4)
    assert ed.spots[0].inner_area() < before
    assert not ed.set_inner_scale(0.99), "out of range must be refused"
    assert not ed.set_inner_scale(0.05)


def test_nudge_moves_the_selected_spot():
    ed = editor()
    spot = ed.add_at(400, 300)
    before = spot.polygon.copy()
    ed.nudge(5, -5)
    assert np.allclose(ed.spots[0].polygon - before, [5, -5])


# --------------------------------------------------------------------------- #
# generate row
# --------------------------------------------------------------------------- #

def test_generate_row_uses_the_bay_pitch():
    ed = editor()
    spot = ed.add_at(150, 150)
    # make it exactly 70 wide so the pitch is known
    spot.polygon = np.array([[90, 80], [160, 80], [160, 220], [90, 220]], dtype=np.float64)
    ed._prepare(spot)
    added = ed.generate_row(+1, 3)
    assert added == 3
    xs = sorted(s.polygon[:, 0].min() for s in ed.spots)
    assert xs == [90, 160, 230, 300], xs


def test_generate_row_follows_a_slanted_bay():
    ed = editor()
    spot = ed.add_at(150, 150)
    spot.polygon = np.array([[100, 80], [170, 90], [170, 230], [100, 220]], dtype=np.float64)
    ed._prepare(spot)
    ed.generate_row(+1, 1)
    assert len(ed.spots) == 2
    dx = ed.spots[1].polygon[:, 0] - ed.spots[0].polygon[:, 0]
    dy = ed.spots[1].polygon[:, 1] - ed.spots[0].polygon[:, 1]
    assert np.allclose(dx, 70) and np.allclose(dy, 10), (dx, dy)


def test_generate_row_stops_at_the_frame_edge():
    ed = editor()
    spot = ed.add_at(150, 150)
    spot.polygon = np.array([[W - 90, 80], [W - 20, 80], [W - 20, 220], [W - 90, 220]],
                            dtype=np.float64)
    ed._prepare(spot)
    assert ed.generate_row(+1, 5) == 0, "must refuse to walk off the frame"


def test_generate_row_can_go_left():
    ed = editor()
    spot = ed.add_at(400, 150)
    spot.polygon = np.array([[300, 80], [370, 80], [370, 220], [300, 220]], dtype=np.float64)
    ed._prepare(spot)
    assert ed.generate_row(-1, 2) == 2
    xs = sorted(s.polygon[:, 0].min() for s in ed.spots)
    assert xs == [160, 230, 300], xs


# --------------------------------------------------------------------------- #
# auto detect merging
# --------------------------------------------------------------------------- #

def test_auto_detect_adds_candidates():
    ed = editor()
    report = ed.auto_detect(lot_frame(), DetectorParams())
    assert report["found"] == 12, report
    assert report["added"] == 12, report
    assert len(ed.spots) == 12
    assert ed.dirty


def test_auto_detect_never_destroys_manual_work():
    ed = editor()
    mine = ed.add_at(400, 300)
    mine.polygon = np.array([[660, 100], [700, 100], [700, 180], [660, 180]],
                            dtype=np.float64)
    ed._prepare(mine)
    ed.rename_selected("99")
    before = ed.spots[0].polygon.copy()
    report = ed.auto_detect(lot_frame(), DetectorParams())
    assert any(s.id == "99" for s in ed.spots), "the hand-made spot disappeared"
    kept = next(s for s in ed.spots if s.id == "99")
    assert np.allclose(kept.polygon, before), "the hand-made spot was modified"
    assert report["skipped_overlap"] >= 0


def test_undo_reverses_a_whole_auto_detect():
    ed = editor()
    ed.add_at(400, 300)
    before = len(ed.spots)
    # the hand-made bay overlaps one synthetic bay, which auto detect must skip
    report = ed.auto_detect(lot_frame(), DetectorParams())
    assert len(ed.spots) == before + report["added"], (len(ed.spots), report)
    assert report["added"] >= 11, report
    assert ed.undo()
    assert len(ed.spots) == before, "one undo should remove everything auto detect added"


def test_auto_detect_on_a_blank_frame_reports_and_survives():
    ed = editor()
    report = ed.auto_detect(np.full((H, W, 3), 128, np.uint8), DetectorParams())
    assert report["found"] == 0
    assert ed.message.level == "warn"
    assert "hand" in ed.message.text or "generate" in ed.message.text


def test_candidates_keep_the_inner_detection_polygon():
    ed = editor()
    ed.auto_detect(lot_frame(), DetectorParams())
    for s in ed.spots:
        assert s.inner_points is not None
        assert s.inner_area() > 0
        assert s.inner_area() < s.area(), "inner ROI must be smaller than the bay"


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #

def test_save_and_load_round_trip():
    ed = editor()
    ed.add_at(200, 200)
    ed.add_at(400, 200)
    ed.rename_selected("57")
    ed.set_inner_scale(0.5)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "parking_spots.json")
        assert ed.save(path)
        assert not ed.dirty

        other = SpotEditor((H, W), start_id=38)
        assert other.load(path)
        assert [s.id for s in other.spots] == ["38", "57"]
        for a, b in zip(ed.spots, other.spots):
            assert np.allclose(a.polygon, b.polygon)
            assert abs(a.inner_scale - b.inner_scale) < 1e-6
            assert a.inner_area() == b.inner_area()
        raw = json.load(open(path))
        assert "spots" in raw and len(raw["spots"]) == 2


def test_load_missing_file_reports_instead_of_raising():
    ed = editor()
    assert not ed.load("/tmp/definitely-not-here.json")
    assert ed.message.level == "warn"


def test_save_an_empty_editor_refuses():
    ed = editor()
    assert not ed.save("/tmp/should-not-be-written.json")
    assert not os.path.exists("/tmp/should-not-be-written.json")


# --------------------------------------------------------------------------- #
# GUI wiring (synthetic events, no window)
# --------------------------------------------------------------------------- #

def test_buttons_exist_and_clicking_auto_detect_works():
    app = app_with()
    # render once so the button rectangles are computed
    canvas = app.render(app._frame)
    assert canvas.shape[:2] == (H, W)
    assert len(app._buttons) >= 5
    labels = [b[4] for b in app._buttons]
    for want in ("Auto Detect", "Save", "Undo", "Reset"):
        assert want in labels, labels

    auto = next(b for b in app._buttons if b[4] == "Auto Detect")
    cx, cy = auto[0] + auto[2] // 2, auto[1] + auto[3] // 2
    app.on_mouse(cv2.EVENT_LBUTTONDOWN, cx, cy, 0, None)
    assert len(app.ed.spots) == 12, len(app.ed.spots)


def test_key_a_runs_auto_detect():
    app = app_with()
    assert app.act("a")
    assert len(app.ed.spots) == 12


def test_mouse_drag_through_the_app_moves_a_corner():
    app = app_with()
    app.act("a")
    # Drag the *selected* spot: adjacent detected bays share their corner
    # coordinates exactly, so hit_test would otherwise grab the selected
    # neighbour's corner and this test would be measuring the wrong thing.
    spot = app.ed.spots[app.ed.selected]
    x0, y0 = spot.polygon[0]
    app.on_mouse(cv2.EVENT_LBUTTONDOWN, int(x0), int(y0), 0, None)
    app.on_mouse(cv2.EVENT_MOUSEMOVE, int(x0) + 7, int(y0) + 4, cv2.EVENT_FLAG_LBUTTON, None)
    app.on_mouse(cv2.EVENT_LBUTTONUP, int(x0) + 7, int(y0) + 4, 0, None)
    assert np.allclose(app.ed.spots[app.ed.selected].polygon[0], [x0 + 7, y0 + 4])


def test_clicking_empty_space_clears_the_selection():
    app = app_with()
    app.act("a")
    assert app.ed.selected is not None
    app.on_mouse(cv2.EVENT_LBUTTONDOWN, 3, 3, 0, None)
    assert app.ed.selected is None


def test_keys_delete_undo_and_quit():
    app = app_with()
    app.act("a")
    n = len(app.ed.spots)
    app.ed.select(0)
    app.act("d")
    assert len(app.ed.spots) == n - 1
    app.act("z")
    assert len(app.ed.spots) == n
    assert app.act("q") is False, "'q' must ask the app to stop"


def test_key_n_adds_a_bay_at_the_cursor():
    app = app_with()
    app.cursor = (450, 300)
    app.act("n")
    assert len(app.ed.spots) == 1
    c = app.ed.spots[0].polygon.mean(axis=0)
    assert abs(c[0] - 450) < 20 and abs(c[1] - 300) < 20, c


def test_tab_cycles_the_selection():
    app = app_with()
    app.act("a")
    app.ed.select(0)
    app.act("\t")
    assert app.ed.selected == 1
    for _ in range(len(app.ed.spots) - 1):
        app.act("\t")
    assert app.ed.selected == 0


def test_save_key_writes_the_file():
    with tempfile.TemporaryDirectory() as tmp:
        ed = editor()
        app = CalibrationApp(ed, os.path.join(tmp, "spots.json"))
        app.detector_params = DetectorParams()
        app._frame = lot_frame()
        app.act("a")
        app.act("s")
        assert os.path.exists(os.path.join(tmp, "spots.json"))
        reloaded = load_spots(os.path.join(tmp, "spots.json"), (H, W))
        assert len(reloaded) == 12


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
