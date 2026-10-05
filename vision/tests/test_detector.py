"""Synthetic car parks: does the geometry detector find the bays?

Every case is drawn, so the expected answer is known exactly. The task says
detection does not have to be perfect, only a good assistant - but "good" should
mean something measurable, so these tests pin the recall on regular layouts
(where it must be exact) and check that hard cases degrade to *fewer* candidates
rather than invented ones.

Run:  python tests/test_detector.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from detector import DetectorParams, detect_bays, sort_reading_order  # noqa: E402


def synth_lot(rows: int, bays: int, bay_w: int = 70, bay_h: int = 140,
              x0: int = 90, y0: int = 80, row_gap: int = 40,
              width: int = 900, height: int = 600,
              slant: float = 0.0, missing=(), noise: int = 0,
              background: int = 120) -> np.ndarray:
    """A regular car park: per row, bays+1 dividers between two horizontal lines."""
    img = np.full((height, width, 3), background, np.uint8)
    for r in range(rows):
        ry0 = int(y0 + r * (bay_h + row_gap))
        ry1 = ry0 + bay_h
        dx = int(slant * (ry1 - ry0))
        xs = [int(x0 + i * bay_w) for i in range(bays + 1)]
        for i, x in enumerate(xs):
            if (r, i) in missing:
                continue
            cv2.line(img, (x, ry0), (x + dx, ry1), (245, 245, 245), 3)
        cv2.line(img, (xs[0], ry0), (xs[-1] + dx, ry0), (245, 245, 245), 2)
        cv2.line(img, (xs[0], ry1), (xs[-1] + dx, ry1), (245, 245, 245), 2)
    if noise:
        rng = np.random.default_rng(1)
        img = np.clip(img.astype(int) + rng.integers(-noise, noise + 1, img.shape),
                      0, 255).astype(np.uint8)
    return img


def detected_bays(img, **kw):
    return detect_bays(img, DetectorParams(**kw))


# --------------------------------------------------------------------------- #
# regular layouts must be found exactly
# --------------------------------------------------------------------------- #

def _check_exact(name, expect, **kw):
    img = synth_lot(**kw)
    res = detected_bays(img)
    assert res.count == expect, (
        f"{name}: expected {expect} bays, detector found {res.count}; "
        f"diagnostics={res.diagnostics}")


def test_single_row_of_four():
    _check_exact("1x4", 4, rows=1, bays=4)


def test_single_row_of_eight():
    _check_exact("1x8", 8, rows=1, bays=8)


def test_two_rows_of_six():
    _check_exact("2x6", 12, rows=2, bays=6)


def test_three_rows_of_eight():
    _check_exact("3x8", 24, rows=3, bays=8)


def test_narrow_bays():
    _check_exact("narrow 2x10", 20, rows=2, bays=10, bay_w=44)


def test_narrow_and_tall_bays():
    _check_exact("narrow+tall 2x8", 16, rows=2, bays=8, bay_w=38, bay_h=180)


def test_short_bays():
    _check_exact("short 2x6", 12, rows=2, bays=6, bay_h=60, row_gap=30)


def test_wide_bays():
    _check_exact("wide 2x4", 8, rows=2, bays=4, bay_w=150)


def test_slanted_row():
    """A row seen at an angle: bay edges are not vertical."""
    _check_exact("slanted 2x5", 10, rows=2, bays=5, slant=0.18)


def test_noisy_image():
    _check_exact("noisy 2x6", 12, rows=2, bays=6, noise=14)


def test_bays_are_where_they_should_be():
    """Not just the right count - the polygons must sit on the drawn lines."""
    img = synth_lot(rows=2, bays=6, bay_w=70, bay_h=140, x0=90, y0=80, row_gap=40)
    res = detected_bays(img)
    polys = sort_reading_order(res.polygons)
    row1 = [p for p in polys if np.asarray(p)[:, 1].mean() < 250]
    assert len(row1) == 6, len(row1)
    xs = sorted(np.asarray(p)[:, 0].min() for p in row1)
    for i, x in enumerate(xs):
        assert abs(x - (90 + i * 70)) <= 4, (i, x, xs)


# --------------------------------------------------------------------------- #
# missing dividers: the even spacing must let the row be reconstructed
# --------------------------------------------------------------------------- #

def test_missing_divider_is_reconstructed():
    """The pitch is known, so a bay whose divider was not detected is still found."""
    img = synth_lot(rows=2, bays=6, missing=((0, 3), (1, 5)))
    res = detected_bays(img)
    assert res.count == 12, f"expected 12 (reconstructed), got {res.count}"
    filled = sum(r.get("grid_filled", 0) for r in res.diagnostics["rows"])
    assert filled >= 1, res.diagnostics


# --------------------------------------------------------------------------- #
# failure modes must be quiet, not destructive
# --------------------------------------------------------------------------- #

def test_blank_image_finds_nothing_and_does_not_raise():
    img = np.full((600, 800, 3), 128, np.uint8)
    res = detected_bays(img)
    assert res.count == 0
    assert "error" not in res.diagnostics or res.count == 0


def test_noise_only_image_finds_nothing_plausible():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 255, (600, 800, 3), dtype=np.uint8)
    res = detected_bays(img)
    # Random noise has no regular dividers; a few candidates would be acceptable,
    # a screenful would mean the plausibility filters are not working.
    assert res.count <= 6, f"noise produced {res.count} candidates"


def test_single_line_finds_no_bays():
    img = np.full((600, 800, 3), 120, np.uint8)
    cv2.line(img, (400, 100), (400, 500), (255, 255, 255), 3)
    assert detected_bays(img).count == 0


def test_orientation_is_reported():
    img = synth_lot(rows=2, bays=6)
    res = detected_bays(img)
    assert res.diagnostics.get("orientation") in ("vertical", "horizontal")
    assert res.diagnostics.get("method") in ("profile", "segments")


def test_horizontal_bays_are_found_too():
    """Bays whose dividers run horizontally must be found by the same code path."""
    img = synth_lot(rows=6, bays=2, bay_w=140, bay_h=70, x0=80, y0=60,
                    row_gap=0, width=600, height=900)
    rot = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    res = detect_bays(rot, DetectorParams(orientation="auto"))
    assert res.count >= 6, f"horizontal bays not found: {res.count}"


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
