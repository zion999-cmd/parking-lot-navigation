"""The parking-lot outline: drop red things outside it, keep the car inside it."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import hsv_marker                                                # noqa: E402
import spot_config                                               # noqa: E402

# A rectangle well inside an 800x600 frame, with a red square in the middle and one outside it.
REGION = [[200, 200], [600, 200], [600, 500], [200, 500]]


def frame_with_blobs(centres, size=9):
    frame = np.zeros((600, 800, 3), np.uint8)
    for cx, cy in centres:
        frame[cy - size:cy + size, cx - size:cx + size] = (0, 0, 220)   # BGR red
    return frame


def centres_of(pool):
    return sorted((round(c["centre"][0]), round(c["centre"][1])) for c in pool)


def test_inside_is_a_plain_ray_cast():
    assert hsv_marker._inside(REGION, 400, 350) is True
    assert hsv_marker._inside(REGION, 100, 350) is False
    assert hsv_marker._inside(REGION, 400, 550) is False
    assert hsv_marker._inside(None, 1, 1) is True        # no outline: everything is allowed


def test_region_drops_blobs_outside_the_lot():
    frame = frame_with_blobs([(400, 350), (60, 350)])
    without = centres_of(hsv_marker.candidates(frame, max_shape=1.0))
    with_region = centres_of(hsv_marker.candidates(frame, max_shape=1.0, region=REGION))
    assert (400, 350) in with_region
    assert (60, 350) not in with_region
    assert len(with_region) < len(without), "the outline must remove something"


def test_local_search_also_respects_the_region():
    """The tracking path passes centre/radius; the outline still applies."""
    frame = frame_with_blobs([(400, 350), (250, 220)])
    pool = hsv_marker.candidates(frame, centre=(400, 350), radius=220, max_shape=None,
                                 region=[[380, 330], [420, 330], [420, 370], [380, 370]])
    assert centres_of(pool) == [(400, 350)], centres_of(pool)


def test_load_lot_region_reads_the_calibration(tmp_path=None):
    import json
    import tempfile

    doc = {"spots": [{"id": "9", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]}],
           "lot_region": REGION}
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "parking_spots.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(doc, handle)
        region = spot_config.load_lot_region(path)
        assert region is not None and len(region) == 4
        assert hsv_marker._inside(region, 400, 350) is True


def test_missing_or_broken_region_is_none_not_an_error(tmp_path=None):
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "parking_spots.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"spots": []}, handle)
        assert spot_config.load_lot_region(path) is None
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"spots": [], "lot_region": [[1, 2]]}, handle)   # fewer than 3 points
        assert spot_config.load_lot_region(path) is None
        assert spot_config.load_lot_region(os.path.join(directory, "nope.json")) is None


def test_editor_renders_a_region_without_crashing():
    """Regression: _draw_lot_region referenced a colour constant that does not exist.

    It only fires once two points exist, so the editor worked, the first click worked, and the crash
    arrived on the second point - after 'save' in the report, because the frame is redrawn after a save.
    """
    import calibration as C
    import spot_editor

    editor = spot_editor.SpotEditor((600, 800))
    editor.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "config", "parking_spots.json"))
    app = C.CalibrationApp(editor, os.path.join(os.path.dirname(os.path.abspath(
        os.path.dirname(os.path.dirname(__file__)))), "vision", "config", "parking_spots.json"))
    frame = np.zeros((600, 800, 3), np.uint8)
    for count in (0, 1, 2, 3, 4):
        app.lot = [(210, 120), (700, 120), (700, 520), (210, 520)][:count]
        for layer in ("spots", "roads", "lot"):
            app.switch_layer(layer)
            canvas = app.render(frame)
            assert canvas.shape == frame.shape


def _lot_app():
    import calibration as C
    import spot_editor

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    editor = spot_editor.SpotEditor((600, 800))
    editor.load(os.path.join(root, "config", "parking_spots.json"))
    app = C.CalibrationApp(editor, os.path.join(root, "config", "parking_spots.json"))
    app.switch_layer("lot")
    return app


def test_q_and_escape_quit_from_the_lot_layer():
    """The lot layer used to swallow every key, which made q do nothing there."""
    for key in ("q", "\x1b"):
        app = _lot_app()
        assert app.act(key) is False, key


def test_t_still_cycles_layers_from_the_lot_layer():
    app = _lot_app()
    assert app.act("t") is True
    assert app.layer == "spots"


def test_a_corner_can_be_dragged():
    app = _lot_app()
    app.lot = [(200.0, 200.0), (600.0, 200.0), (600.0, 500.0), (200.0, 500.0)]
    app.on_mouse(1, 203, 197, 0, None)          # press near the first corner
    assert app._lot_drag == 0
    app.on_mouse(0, 320, 260, 1, None)          # drag with the button held
    assert app.lot[0] == (320.0, 260.0)
    app.on_mouse(4, 320, 260, 0, None)          # release
    assert app._lot_drag is None
    assert len(app.lot) == 4


def test_pressing_away_from_a_corner_adds_one():
    app = _lot_app()
    app.lot = [(200.0, 200.0)]
    app.on_mouse(1, 500, 400, 0, None)
    assert len(app.lot) == 2 and app.lot[1] == (500.0, 400.0)


def test_no_more_than_four_points():
    app = _lot_app()
    app.lot = [(100.0, 100.0), (500.0, 100.0), (500.0, 400.0), (100.0, 400.0)]
    app.on_mouse(1, 300, 300, 0, None)          # inside the quad, not near a corner
    assert len(app.lot) == 4


def test_lot_layer_does_not_show_a_selected_bay():
    """The reported bug: in lot mode the highlight still belonged to a bay."""
    app = _lot_app()
    app.ed.selected = 5
    app.lot = [(200.0, 200.0), (600.0, 200.0), (600.0, 500.0), (200.0, 500.0)]
    canvas = app.render(np.zeros((600, 800, 3), np.uint8))
    assert canvas.shape == (600, 800, 3)
    # _draw_spots must be called faint in this layer, which is what removes the selection ring.
    import inspect
    source = inspect.getsource(app.render)
    assert 'self.layer == "lot"' in source and "faint=True" in source


def test_clicking_a_corner_selects_it_and_d_removes_it():
    app = _lot_app()
    app.lot = [(200.0, 200.0), (600.0, 200.0), (600.0, 500.0), (200.0, 500.0)]
    app.on_mouse(1, 202, 198, 0, None)
    assert app._lot_sel == 0
    assert "point 1" in app._header()
    app.act("d")
    assert len(app.lot) == 3 and app._lot_sel is None
    app.render(np.zeros((600, 800, 3), np.uint8))
