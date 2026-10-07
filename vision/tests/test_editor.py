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

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
REAL_CONFIG = os.path.join(HERE, "..", "config", "parking_spots.json")
sys.path.insert(0, SRC)

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
    app.detector_params = DetectorParams(orientation="vertical")
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
    report = ed.auto_detect(lot_frame(), DetectorParams(orientation="vertical"))
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
    report = ed.auto_detect(lot_frame(), DetectorParams(orientation="vertical"))
    assert any(s.id == "99" for s in ed.spots), "the hand-made spot disappeared"
    kept = next(s for s in ed.spots if s.id == "99")
    assert np.allclose(kept.polygon, before), "the hand-made spot was modified"
    assert report["skipped_overlap"] >= 0


def test_undo_reverses_a_whole_auto_detect():
    ed = editor()
    ed.add_at(400, 300)
    before = len(ed.spots)
    # the hand-made bay overlaps one synthetic bay, which auto detect must skip
    report = ed.auto_detect(lot_frame(), DetectorParams(orientation="vertical"))
    assert len(ed.spots) == before + report["added"], (len(ed.spots), report)
    assert report["added"] >= 11, report
    assert ed.undo()
    assert len(ed.spots) == before, "one undo should remove everything auto detect added"


def test_auto_detect_on_a_blank_frame_reports_and_survives():
    ed = editor()
    report = ed.auto_detect(np.full((H, W, 3), 128, np.uint8), DetectorParams(orientation="vertical"))
    assert report["found"] == 0
    assert ed.message.level == "warn"
    assert "hand" in ed.message.text or "generate" in ed.message.text


def test_candidates_keep_the_inner_detection_polygon():
    ed = editor()
    ed.auto_detect(lot_frame(), DetectorParams(orientation="vertical"))
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
        app.detector_params = DetectorParams(orientation="vertical")
        app._frame = lot_frame()
        app.act("a")
        app.act("s")
        assert os.path.exists(os.path.join(tmp, "spots.json"))
        reloaded = load_spots(os.path.join(tmp, "spots.json"), (H, W))
        assert len(reloaded) == 12



# ---------------------------------------------------------------------------
# Bays that leave the frame
#
# A drag that leaves the window reports coordinates outside the image, and a bay parked
# outside was saved, after which the viewer refused to start at all ("spot 167:
# detection area is empty after clipping to 800x600"). Three guards now stop that, and
# they are tested separately because each covers a different route into the same broken
# file - plus the one that lets the viewer start anyway, which is what makes it possible
# to get into edit mode and delete the offending bay.
# ---------------------------------------------------------------------------

def test_a_drag_that_leaves_the_window_is_clamped_into_the_frame() -> None:
    ed = SpotEditor((H, W))
    ed.add_at(200, 200)
    ed.select(0)
    ed.begin_drag(*np.asarray(ed.spots[0].polygon[0], dtype=float))
    ed.update_drag(5000, -3000)          # what the backend reports when the pointer exits
    ed.end_drag()
    poly = np.asarray(ed.spots[0].polygon, dtype=float)
    assert poly[:, 0].min() >= 0 and poly[:, 0].max() < W, poly
    assert poly[:, 1].min() >= 0 and poly[:, 1].max() < H, poly


def test_a_nudge_cannot_push_a_bay_out_of_the_frame() -> None:
    ed = SpotEditor((H, W))
    ed.add_at(60, 60)
    ed.select(0)
    for _ in range(40):                  # 40 x 5 px is well past the left edge
        ed.nudge(-5, 0)
    assert np.asarray(ed.spots[0].polygon, dtype=float)[:, 0].min() >= 0


def test_saving_a_bay_outside_the_frame_is_refused() -> None:
    ed = SpotEditor((H, W))
    ed.add_at(200, 200)
    ed.spots[0].polygon = np.asarray(ed.spots[0].polygon, dtype=float) + [4000, 0]
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "spots.json")
        assert ed.save(out) is False
        # The half that matters: nothing written, so the previous good file survives.
        assert not os.path.exists(out)
        assert ed.dirty is True


def test_rows_by_y_groups_a_hand_drawn_two_row_calibration() -> None:
    ed = SpotEditor((H, W))
    for i in range(4):
        ed.add_at(100 + i * 80, 120)
    for i in range(3):
        ed.add_at(100 + i * 80, 320)     # 200 px lower: unambiguously another row
    assert sorted(len(r) for r in ed.rows_by_y()) == [3, 4]


def test_row_scope_moves_only_the_row_of_the_selected_bay() -> None:
    ed = SpotEditor((H, W))
    for i in range(4):
        ed.add_at(100 + i * 80, 120)
    for i in range(3):
        ed.add_at(100 + i * 80, 320)
    rows = ed.rows_by_y()
    assert len(rows) == 2, rows

    app = CalibrationApp(ed, "unused.json")
    app.nudge_scope = "row"
    ed.select(rows[0][0])
    before_row = [np.asarray(ed.spots[i].polygon, dtype=float).copy() for i in rows[0]]
    before_other = [np.asarray(ed.spots[i].polygon, dtype=float).copy() for i in rows[1]]

    app._nudges(10, 0)

    for i, b in zip(rows[0], before_row):
        assert np.allclose(np.asarray(ed.spots[i].polygon, dtype=float) - b, [10, 0])
    for i, b in zip(rows[1], before_other):
        assert np.allclose(np.asarray(ed.spots[i].polygon, dtype=float), b)


def test_out_of_frame_bays_are_skipped_so_the_viewer_still_starts() -> None:
    one = {"id": "1", "polygon": [[10, 10], [50, 10], [50, 60], [10, 60]]}
    stray = {"id": "167", "polygon": [[1306, 384], [1354, 384], [1354, 488], [1306, 488]]}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "spots.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"spots": [one], "image_size": [H, W]}, fh)
        assert len(load_spots(path, (H, W))) == 1

        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"spots": [one, stray], "image_size": [H, W]}, fh)
        try:
            load_spots(path, (H, W))
        except Exception:
            pass
        else:
            raise AssertionError("a stray bay alone should still be an error by default")
        # ...but with skip_bad the viewer starts, which is what lets 'e' fix the file.
        assert len(load_spots(path, (H, W), skip_bad=True)) == 1


# ---------------------------------------------------------------------------
# Renaming a bay
# ---------------------------------------------------------------------------

def a_loaded_editor(tmp: str):
    """A real editor over a copy of the real map, so the ids are the ones a person sees."""
    import shutil, spot_config, spot_editor
    cfg = os.path.join(tmp, "parking_spots.json")
    shutil.copy(REAL_CONFIG, cfg)
    editor = spot_editor.SpotEditor((600, 800))
    assert editor.load(cfg)
    return editor, cfg


def test_renaming_a_bay_works_and_refuses_only_empty_or_duplicate_ids() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        editor, cfg = a_loaded_editor(tmp)
        original = [s.id for s in editor.spots[:3]]
        editor.select(0)
        assert editor.rename_selected("A1") is True
        assert editor.spots[0].id == "A1" and editor.dirty is True
        assert editor.rename_selected("") is False, "an empty id is not a name"
        assert editor.rename_selected(original[1]) is False, "a duplicate id would be ambiguous"
        assert editor.spots[0].id == "A1", "a refused rename must not half-apply"


def test_a_renamed_bay_survives_saving_and_reloading_even_in_chinese() -> None:
    """The user's case: rename a bay and it must still be called that afterwards."""
    import spot_config
    with tempfile.TemporaryDirectory() as tmp:
        editor, cfg = a_loaded_editor(tmp)
        editor.select(0)
        assert editor.rename_selected("东 1") is True
        assert editor.save(cfg) is True
        again = spot_config.load_spots(cfg, skip_bad=True)
        assert again[0].id == "东 1", [s.id for s in again[:3]]


def test_the_window_says_which_key_renames_in_the_window_you_are_looking_at() -> None:
    """The bug behind "renaming did not work": the window's hint told you to press r, while r in
    that window captured the reference frame. Both now agree that r renames, and the capture moved
    to R - checked here so the two cannot drift apart again."""
    editor_src = open(os.path.join(SRC, "calibration.py"), encoding="utf-8").read()
    main_src = open(os.path.join(SRC, "main.py"), encoding="utf-8").read()
    tail = main_src[main_src.index("def run_edit_mode("):main_src.index("def main() -> int:")]
    assert 'self.rename_hint = "r"' in editor_src, "the standalone default"
    assert "{self.rename_hint} rename" in editor_src, "the hint must use it, not a literal"
    assert "app.rename_hint" not in tail, \
        "the edit mode must not override it: r renames there too"
    assert "key == ord('R')" in tail, "the reference capture must be uppercase R"
    assert "key == ord('r')" not in tail, "and r must reach the editor, where it renames"
    assert "r = rename" in tail, "the mode says out loud what each key does"


def test_leaving_the_editor_writes_the_change_even_if_q_was_not_the_last_key() -> None:
    """The second net: the caller reloads the config as soon as the editor returns, so anything
    still marked dirty has to be written before that happens."""
    main_src = open(os.path.join(SRC, "main.py"), encoding="utf-8").read()
    tail = main_src[main_src.index("def run_edit_mode("):main_src.index("def main() -> int:")]
    assert "app._save_all()" in tail, "the edit mode must flush on the way out"
    assert tail.index("app._save_all()") > tail.index("cv2.destroyAllWindows()"), \
        "and after the loop, not before it"


# --------------------------------------------------------------------------- #
# arrow keys: half a pixel at a time
# --------------------------------------------------------------------------- #
# A bay one pixel out cannot be fixed with a mouse. The arrow keys move the selected corner - or
# the whole bay when no corner was clicked - by half a pixel, which is small enough to be
# invisible, so the window answers with the coordinate instead.

UP, DOWN, LEFT, RIGHT = 63232, 63233, 63234, 63235           # the family macOS reports
X_LEFT, X_UP, X_RIGHT, X_DOWN = 65361, 65362, 65363, 65364   # the X11 / GTK / Qt family


def app_with_one_bay():
    ed = SpotEditor((H, W), start_id=38)
    ed.add_at(400, 300)
    app = CalibrationApp(ed, "/tmp/does-not-matter.json")
    app._frame = lot_frame()
    return app


def test_the_arrow_keys_are_handled_and_ordinary_keys_are_not():
    a = app_with_one_bay()
    for code in (UP, DOWN, LEFT, RIGHT, X_LEFT, X_UP, X_RIGHT, X_DOWN):
        assert a.act_code(code) is True, code
    for code in (ord("a"), ord("q"), ord("s"), 13, 9, -1, 255):
        assert a.act_code(code) is False, code


def test_arrow_keys_move_the_selected_bay_by_half_a_pixel():
    a = app_with_one_bay()
    a.ed.select(0)
    a.ed.selected_corner = None                 # a body selection, not a corner
    before = a.ed.spots[0].polygon.copy()
    a.act_code(UP)
    assert np.allclose(a.ed.spots[0].polygon - before, [0, -0.5]), a.ed.spots[0].polygon - before
    a.act_code(RIGHT)
    assert np.allclose(a.ed.spots[0].polygon - before, [0.5, -0.5])
    assert a.ed.dirty


def test_arrow_keys_move_only_the_selected_corner():
    a = app_with_one_bay()
    corner = 2
    x, y = a.ed.spots[0].polygon[corner]
    assert a.ed.begin_drag(x + 1, y + 1)               # a click on the corner selects it
    assert a.ed.selected_corner == corner
    before = a.ed.spots[0].polygon.copy()
    a.act_code(LEFT)
    after = a.ed.spots[0].polygon
    moved = [i for i in range(len(after)) if not np.allclose(after[i], before[i])]
    assert moved == [corner], f"moved {moved}, wanted only corner {corner}"
    assert np.allclose(after[corner] - before[corner], [-0.5, 0.0])


def test_a_click_on_the_body_means_the_whole_bay():
    """No new mode to learn: whatever was clicked last is what the arrows move."""
    a = app_with_one_bay()
    poly = a.ed.spots[0].polygon
    centre = poly.mean(axis=0)
    assert a.ed.begin_drag(float(centre[0]), float(centre[1]))
    assert a.ed.selected_corner is None
    before = poly.copy()
    a.act_code(DOWN)
    assert np.allclose(a.ed.spots[0].polygon - before, [0, 0.5])


def test_twenty_presses_are_exactly_ten_pixels():
    """Half a pixel is only useful if it does not drift."""
    a = app_with_one_bay()
    a.ed.begin_drag(a.ed.spots[0].polygon[0][0] + 1, a.ed.spots[0].polygon[0][1] + 1)
    start = a.ed.spots[0].polygon[a.ed.selected_corner].copy()
    for _ in range(20):
        a.act_code(LEFT)
    moved = a.ed.spots[0].polygon[a.ed.selected_corner] - start
    assert np.allclose(moved, [-10.0, 0.0]), moved


def test_a_run_of_presses_is_one_undo_step():
    """Twenty presses in a row are one undo step; twenty separate corrections would be twenty.

    The run's first press is the one that records the state to come back to - what matters is that
    the other nineteen do not each add their own.
    """
    a = app_with_one_bay()
    a.ed.begin_drag(a.ed.spots[0].polygon[0][0] + 1, a.ed.spots[0].polygon[0][1] + 1)
    depth = len(a.ed._undo)
    for _ in range(20):
        a.act_code(LEFT)
    assert len(a.ed._undo) == depth + 1, \
        f"20 presses pushed {len(a.ed._undo) - depth} undo states, wanted 1"


def test_presses_that_are_not_a_run_are_their_own_undo_steps():
    """The contrast that shows the merging is real rather than the stack never growing."""
    a = app_with_one_bay()
    a.ed.begin_drag(a.ed.spots[0].polygon[0][0] + 1, a.ed.spots[0].polygon[0][1] + 1)
    depth = len(a.ed._undo)
    for _ in range(5):
        a.act_code(LEFT)
        a.act("i")                     # an ordinary key ends the run each time
        a.act("i")
    assert len(a.ed._undo) == depth + 5, len(a.ed._undo) - depth


def test_an_ordinary_key_ends_the_run():
    a = app_with_one_bay()
    a.ed.begin_drag(a.ed.spots[0].polygon[0][0] + 1, a.ed.spots[0].polygon[0][1] + 1)
    a.act_code(LEFT)
    depth = len(a.ed._undo)
    a.act("i")                                   # any ordinary key
    a.act_code(LEFT)
    assert len(a.ed._undo) == depth + 1, "the second run joined the first"


def test_an_unknown_special_code_is_reported_rather_than_swallowed():
    """If a backend reports the arrows as something else, the window has to say so.

    That message is why this can be corrected in one line later instead of guessed at now: the
    first press says which number to add.
    """
    a = app_with_one_bay()
    a.ed.select(0)
    assert a.act_code(70000) is False
    assert "70000" in a.ed.message.text, a.ed.message.text
    before = a.ed.spots[0].polygon.copy()
    a.act_code(70000)
    assert np.allclose(a.ed.spots[0].polygon, before), "an unknown code moved something"


def test_the_roads_layer_moves_a_road_point():
    a = app_with_one_bay()
    a.layer = "roads"
    a.road.net.add_node(300, 500)
    a.road.select(0)
    before = (a.road.net.nodes[0].x, a.road.net.nodes[0].y)
    assert a.act_code(UP) is True
    after = (a.road.net.nodes[0].x, a.road.net.nodes[0].y)
    assert abs(after[0] - before[0]) < 1e-9, (before, after)
    assert abs(after[1] - before[1] + 0.5) < 1e-9, (before, after)
    assert a.road.dirty


def test_nothing_selected_says_what_to_do_instead_of_moving_something():
    a = app_with_one_bay()
    a.ed.select(None)
    before = a.ed.spots[0].polygon.copy()
    assert a.act_code(UP) is True
    assert np.allclose(a.ed.spots[0].polygon, before)
    assert "select" in a.ed.message.text, a.ed.message.text


def test_a_guessed_low_code_says_that_it_was_a_guess():
    """The low codes cannot collide with real keys, but they are a convention, not a measurement.

    So the message names the code. If the direction turns out to be wrong, the window already
    says which number to swap, instead of moving a bay the wrong way and saying nothing.
    """
    a = app_with_one_bay()
    a.ed.select(0)
    a.ed.selected_corner = None
    before = a.ed.spots[0].polygon.copy()
    assert a.act_code(3) is True                     # 3 is 'right' in the guessed table
    assert np.allclose(a.ed.spots[0].polygon - before, [0.5, 0.0])
    assert "arrow code 3" in a.ed.message.text, a.ed.message.text
    # The high families are measured facts and carry no such note.
    a.act_code(65363)
    assert "arrow code" not in a.ed.message.text, a.ed.message.text


def test_a_key_that_cannot_be_a_character_is_reported():
    """The bug reported twice: an unhandled code was swallowed in silence.

    Printable characters arrive as their ASCII value ('2' is 50), so anything outside 32..126
    that is not an arrow is not a key anyone meant to press, and the window has to say so.
    """
    a = app_with_one_bay()
    for code in (4, 7, 25, 200, 70000):
        assert a.act_code(code) is False, code
        assert str(code) in a.ed.message.text, (code, a.ed.message.text)
    # And the codes that are not keys at all stay quiet: the message must not change. (Checked as
    # "unchanged" rather than "does not contain", because the message from the loop above is
    # still sitting there - the window is not cleared between keys.)
    for code in (-1, 255, ord("a"), ord("q"), 13, 9):
        before = a.ed.message.text
        assert a.act_code(code) is False, code
        assert a.ed.message.text == before, (code, a.ed.message.text)


def test_working_keys_are_not_reported_as_unknown():
    """Tab, Enter and Esc arrive as control codes too, and they are not unknown - they are keys.

    The first version of the report included them, which would have put a warning on screen every
    time somebody pressed Tab to cycle to the next bay.
    """
    a = app_with_one_bay()
    a.ed.select(0)
    a.ed._say("")
    for code in (9, 13, 27, 8, 10, -1, 255):
        assert a.act_code(code) is False, code
        assert a.ed.message.text == "", (code, a.ed.message.text)


def test_both_loops_offer_the_raw_code_before_truncating_it():
    """The bug this would otherwise have been: `waitKey() & 0xFF` keeps 63232's low byte, zero.

    Read out of the shipped source, because the failure is invisible in a unit test: either the
    arrow works, or it silently becomes the character '\x00' and does nothing at all.
    """
    for name in ("calibration.py", "main.py"):
        text = open(os.path.join(SRC, name), encoding="utf-8").read()
        assert "raw = cv2.waitKey(" in text, name
        assert text.index("act_code(raw)") < text.index("raw & 0xFF"), \
            f"{name} truncates the key before the arrow codes are looked at"


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
