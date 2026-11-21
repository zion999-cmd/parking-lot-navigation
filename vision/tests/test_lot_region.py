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
