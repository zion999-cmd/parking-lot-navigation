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


def test_lot_layer_draws_no_bright_bays_and_no_selection_ring():
    """The purple bay ring was still on screen in the lot layer.

    render() had if roads / else, so the lot layer fell into the else branch and drew the bays at full
    strength with the selection ring. Checked on pixels rather than by eye: COL_SEL is (255,0,255) in BGR
    and COL_BAY is (0,255,255).
    """
    app = _lot_app()
    app.ed.selected = 5
    app.lot = [(200.0, 200.0), (600.0, 200.0), (600.0, 500.0), (200.0, 500.0)]
    app._lot_sel = None
    frame = np.full((600, 800, 3), 30, np.uint8)

    app.switch_layer("lot")
    lot_canvas = app.render(frame)
    bay_yellow = int(((lot_canvas[:, :, 0] == 0) & (lot_canvas[:, :, 1] == 255)
                      & (lot_canvas[:, :, 2] == 255)).sum())
    assert bay_yellow == 0, f"{bay_yellow} bright bay pixels in the lot layer"

    app.switch_layer("spots")
    spots_canvas = app.render(frame)
    assert int(((spots_canvas[:, :, 0] == 0) & (spots_canvas[:, :, 1] == 255)
                & (spots_canvas[:, :, 2] == 255)).sum()) > 1000


def test_arrows_do_not_move_a_bay_from_the_lot_layer():
    app = _lot_app()
    app.ed.selected = 5
    app.lot = [(200.0, 200.0), (600.0, 200.0), (600.0, 500.0), (200.0, 500.0)]
    before = [tuple(p) for p in app.ed.spots[5].polygon]
    for key in ("2", "4", "6", "8"):
        app.act(key)
    assert before == [tuple(p) for p in app.ed.spots[5].polygon]
    assert app.act("t") is True and app.layer == "spots"


def test_arrow_keys_move_the_selected_lot_corner_not_a_bay():
    """Arrows reached the bay nudge code because _fine_nudge had no lot branch."""
    app = _lot_app()
    app.ed.selected = 5
    app.lot = [(200.0, 200.0), (600.0, 200.0), (600.0, 500.0), (200.0, 500.0)]
    bay_before = [tuple(p) for p in app.ed.spots[5].polygon]

    app.act_code(app.ARROW_CODES and list(app.ARROW_CODES)[0])       # no selection yet
    assert app.lot[0] == (200.0, 200.0), "nothing should move without a selected corner"

    app._lot_sel = 0
    up = list(app.ARROW_CODES)[0]
    for _ in range(4):
        app.act_code(up)
    assert app.lot[0][1] < 200.0, app.lot[0]
    assert app.lot[0][0] == 200.0
    assert bay_before == [tuple(p) for p in app.ed.spots[5].polygon]


def test_dragging_skips_the_full_frame_blend():
    """The translucent fill costs a full-frame copy; while dragging it is skipped."""
    import time

    app = _lot_app()
    app.lot = [(200.0, 200.0), (600.0, 200.0), (600.0, 500.0), (200.0, 500.0)]
    frame = np.full((600, 800, 3), 30, np.uint8)

    def cost(n=30):
        start = time.perf_counter()
        for _ in range(n):
            app.render(frame)
        return (time.perf_counter() - start) / n * 1000

    app._lot_drag = None
    still = cost()
    app._lot_drag = 0
    dragging = cost()
    app._lot_drag = None
    assert dragging <= still * 1.1, (still, dragging)


def test_saving_the_bays_does_not_delete_the_lot_region():
    """The reported bug: the outline had to be redrawn every time.

    map_document.build re-emits only the keys in SECTION_ORDER, so a key missing from that tuple is
    deleted by the next save of any layer. lot_region was missing, so saving the bays erased it.
    """
    import json
    import shutil
    import tempfile

    import spot_config
    import spot_editor

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "parking_spots.json")
        shutil.copy(os.path.join(root, "config", "parking_spots.json"), path)
        spot_config.save_lot_region(path, [[210, 120], [700, 120], [700, 520], [210, 520]])

        editor = spot_editor.SpotEditor((600, 800))
        assert editor.load(path)
        assert editor.save(path)                       # saving the bays must not lose the outline

        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        assert document.get("lot_region") == [[210, 120], [700, 120], [700, 520], [210, 520]]
        assert len(document["spots"]) == 48
        assert spot_config.load_lot_region(path) is not None


def test_recovery_candidates_are_also_confined_to_the_lot():
    """The second candidate source used to ignore the outline entirely."""
    import vehicle

    class Pose:
        bbox = (60, 300, 30, 20)          # outside REGION
        centre = (75.0, 310.0)
        area = 600
        aspect = 1.5
        score = 0.9
        heading = None

    import types

    def fake_detect(frame):
        return Pose()

    real_arrow, real_aruco = vehicle.red_arrow.detect, vehicle.aruco_hero.detect
    vehicle.red_arrow.detect = fake_detect
    vehicle.aruco_hero.detect = fake_detect
    try:
        vehicle.set_lot_region(None)
        assert vehicle._recovery_candidates(np.zeros((600, 800, 3), np.uint8)), "baseline: pose accepted"
        vehicle.set_lot_region(REGION)
        assert vehicle._recovery_candidates(np.zeros((600, 800, 3), np.uint8)) == [], "outside dropped"
        Pose.centre = (400.0, 350.0)      # inside REGION
        Pose.bbox = (385, 340, 30, 20)
        assert vehicle._recovery_candidates(np.zeros((600, 800, 3), np.uint8)), "inside kept"
    finally:
        vehicle.red_arrow.detect, vehicle.aruco_hero.detect = real_arrow, real_aruco
        vehicle.set_lot_region(None)


def test_a_low_confidence_position_is_not_published():
    """The headline request: do not invent a detection.

    vehicle_state is where a pose becomes a published position. Below the confidence limit it must
    return None, which makes the state document say visible=false, which is what every consumer
    already handles.
    """
    import main

    class Pose:
        def as_dict(self):
            return {"id": "hero", "x": 400.0, "y": 500.0, "heading": 0.0, "visible": True}

    class Vehicle:
        last_verdict = {"confidence": 0.30, "state": "TRACKING"}

    real = main.vehicle
    main.vehicle = Vehicle()
    try:
        assert main.vehicle_state(Pose(), None, 0.60) is None
        Vehicle.last_verdict = {"confidence": 0.85, "state": "TRACKING"}
        out = main.vehicle_state(Pose(), None, 0.60)
        assert out is not None and out["confidence"] == 0.85
        # No verdict at all (first frames): do not refuse on missing evidence.
        Vehicle.last_verdict = {}
        assert main.vehicle_state(Pose(), None, 0.60) is not None
    finally:
        main.vehicle = real


def test_unusable_confidence_cannot_bypass_the_gate():
    """C3: NaN, infinity and non-numbers must not publish a position.

    NaN compares False against every limit and infinite compares greater than all of them, so both
    slipped past `confidence < limit` and published a position nobody had accepted.
    """
    import math

    import main

    class Pose:
        def as_dict(self):
            return {"id": "hero", "x": 400.0, "y": 500.0, "visible": True}

    class Vehicle:
        last_verdict = {}

    real = main.vehicle
    main.vehicle = Vehicle()
    try:
        for bad in (float("nan"), float("inf"), float("-inf"), "bogus", object()):
            Vehicle.last_verdict = {"confidence": bad, "state": "TRACKING"}
            assert main.vehicle_state(Pose(), None, 0.60) is None, bad
        Vehicle.last_verdict = {}                       # no evidence yet: allowed
        assert main.vehicle_state(Pose(), None, 0.60) is not None
        Vehicle.last_verdict = {"confidence": 0.30}
        assert main.vehicle_state(Pose(), None, 0.60) is None
        Vehicle.last_verdict = {"confidence": 0.85}
        assert main.vehicle_state(Pose(), None, 0.60) is not None
    finally:
        main.vehicle = real
