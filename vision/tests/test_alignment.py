"""Image alignment: does it actually undo a known displacement?

Sign conventions are the classic way to get registration wrong, so these tests do
not check matrices against hand-derived expectations. They synthesise a shift or a
rotation, ask the aligner to undo it, and then assert the *result is aligned*:
warping the displaced frame must bring it back onto the reference. That is true or
false regardless of how the matrix is written.

Run:  python tests/test_alignment.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from alignment import Aligner, _to_gray, transform_points     # noqa: E402

W, H = 800, 600


def lot_image(shift=(0, 0), angle=0.0, scale=1.0, width=W, height=H) -> np.ndarray:
    """A regular car park with some texture so ECC has gradients to work with."""
    img = np.full((height, width, 3), 110, np.uint8)
    for r in range(3):
        y0, y1 = 60 + r * 180, 200 + r * 180
        for i in range(10):
            x = 60 + i * 70
            cv2.line(img, (x, y0), (x, y1), (235, 235, 235), 3)
        cv2.line(img, (60, y0), (690, y0), (235, 235, 235), 2)
        cv2.line(img, (60, y1), (690, y1), (235, 235, 235), 2)
    # a couple of asymmetric landmarks, so a shift cannot be confused with anything
    cv2.rectangle(img, (20, 20), (60, 50), (60, 150, 200), -1)
    cv2.circle(img, (740, 560), 18, (200, 120, 60), -1)
    rng = np.random.default_rng(4)
    img = np.clip(img.astype(int) + rng.integers(-6, 7, img.shape), 0, 255).astype(np.uint8)

    if angle or scale != 1.0 or shift != (0, 0):
        m = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
        m[0, 2] += shift[0]
        m[1, 2] += shift[1]
        img = cv2.warpAffine(img, m, (width, height), flags=cv2.INTER_LINEAR)
    return img


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return float(np.abs(ga - gb).mean())


# --------------------------------------------------------------------------- #
# translation
# --------------------------------------------------------------------------- #

def test_small_shift_is_recovered_and_undone():
    ref = lot_image()
    for dx, dy in ((7, -4), (-11, 9), (3, 3), (0, 15)):
        cur = lot_image(shift=(dx, dy))
        al = Aligner(ref, motion="translation")
        res = al.estimate(cur)
        assert res.ok, (dx, dy, res.describe())
        aligned = al.warp_frame(cur, res.matrix)
        before = mean_abs_diff(ref, cur)
        after = mean_abs_diff(ref, aligned)
        assert after < before * 0.35, (
            f"shift ({dx},{dy}): misaligned, mean abs diff {before:.2f} -> {after:.2f} "
            f"({res.describe()})")
        # res.dx is the correction that maps the current frame onto the reference,
        # so it has the opposite sign to the displacement that was applied
        assert abs(res.dx + dx) <= 1.5 and abs(res.dy + dy) <= 1.5, (dx, dy, res.describe())


def test_no_shift_is_near_identity():
    ref = lot_image()
    al = Aligner(ref, motion="translation")
    res = al.estimate(ref.copy())
    assert res.ok
    assert abs(res.dx) < 1.0 and abs(res.dy) < 1.0, res.describe()


def test_phase_correlation_fallback_recovers_a_shift():
    """The fallback must work on its own, without ECC."""
    ref = lot_image()
    cur = lot_image(shift=(12, -8))
    al = Aligner(ref, motion="translation")
    matrix, response, source = al._phase_correlate(al._ref_small,
                                                   _to_gray(cur, al._work_size), "test")
    assert matrix is not None, source
    matrix = al._rescale(matrix)
    aligned = al.warp_frame(cur, matrix)
    assert mean_abs_diff(ref, aligned) < mean_abs_diff(ref, cur) * 0.5, \
        f"fallback did not align (source={source})"


# --------------------------------------------------------------------------- #
# moving the other way: polygons onto the current frame
# --------------------------------------------------------------------------- #

def test_inverse_matrix_moves_a_point_onto_the_current_frame():
    """M brings the current frame onto the reference; M^-1 brings a bay onto it."""
    ref = lot_image()
    dx, dy = 10, -6
    cur = lot_image(shift=(dx, dy))
    al = Aligner(ref, motion="translation")
    res = al.estimate(cur)
    assert res.ok, res.describe()

    inv = al.matrix_to_current(res.matrix)
    # a point that sits at p in the reference must appear at p + (dx, dy) in the
    # current frame, because the frame was created by shifting it that way
    p = np.array([[300.0, 200.0]])
    moved = transform_points(p, inv)[0]
    assert abs(moved[0] - (300 + dx)) <= 2.0, (moved, dx)
    assert abs(moved[1] - (200 + dy)) <= 2.0, (moved, dy)


def test_transform_points_handles_rotation():
    # getRotationMatrix2D with +90 turns clockwise on screen (y grows downwards),
    # so (200, 100) about (100, 100) lands on (100, 0).
    m = cv2.getRotationMatrix2D((100, 100), 90.0, 1.0)
    out = transform_points([[200.0, 100.0]], m)[0]
    assert abs(out[0] - 100) < 1e-6 and abs(out[1] - 0) < 1e-6, out


# --------------------------------------------------------------------------- #
# the guards
# --------------------------------------------------------------------------- #

def test_huge_shift_is_rejected_not_applied():
    ref = lot_image()
    cur = lot_image(shift=(150, 0))          # ~19% of the width
    al = Aligner(ref, motion="translation")
    res = al.estimate(cur)
    assert not res.ok, f"a 150 px shift should be refused, got {res.describe()}"
    # either gate is a legitimate refusal: the magnitude test, or the correlation
    # test when ECC diverges and the fallback finds too little overlap
    assert ("exceeds" in res.reason) or ("correlation" in res.reason), res.reason


def test_unrelated_image_is_rejected():
    ref = lot_image()
    rng = np.random.default_rng(9)
    cur = rng.integers(0, 255, ref.shape, dtype=np.uint8)
    al = Aligner(ref, motion="translation")
    res = al.estimate(cur)
    assert not res.ok, res.describe()


def test_whole_bay_shift_is_never_silently_applied():
    """On an evenly spaced grid a whole-bay shift is indistinguishable from none.

    The aligner must therefore refuse to pretend: either it declines to align, or -
    if it does move - the move must be flagged, because it cannot know whether it
    corrected a real displacement or jumped to the neighbouring bay.
    """
    ref = lot_image()
    cur = lot_image(shift=(70, 0))           # exactly one bay pitch
    al = Aligner(ref, motion="translation", bay_pitch=70.0, max_shift_frac=0.2)
    res = al.estimate(cur)
    # The property that matters is about what gets APPLIED, not about the verdict:
    # either it declines, or it moves by less than half a bay, or it says out loud
    # that it is moving about a whole bay.
    applied = abs(res.dx) if res.ok else 0.0
    assert applied < 0.5 * 70.0 or res.warnings, \
        f"a whole-bay move must not be applied silently: {res.describe()}"
    if applied >= 0.5 * 70.0:
        assert any("bay" in w for w in res.warnings), res.describe()


def test_perfectly_periodic_scene_cannot_be_disambiguated():
    """A documented limitation, asserted so it cannot regress into a wrong move.

    On an idealised evenly spaced grid a whole-bay shift is indistinguishable from
    no shift, and "do nothing" scores higher than the correct one-bay correction.
    The aligner may therefore answer "aligned, no correction needed" and leave the
    ROIs a bay out - harmless, because it applies nothing - but it must never
    invent a sub-bay correction it cannot justify.
    """
    ref = lot_image()
    cur = lot_image(shift=(70, 0))
    al = Aligner(ref, motion="translation", bay_pitch=70.0)
    res = al.estimate(cur)
    if res.ok:
        assert abs(res.dx) < 35.0, res.describe()


def test_alignment_never_makes_the_frame_worse():
    """Property: a rejected estimate is fine, a wrong applied one is not."""
    ref = lot_image()
    for shift in ((0, 0), (4, 0), (-6, 3), (12, -9), (70, 0), (150, 0)):
        cur = lot_image(shift=shift)
        al = Aligner(ref, motion="translation", bay_pitch=70.0)
        before = mean_abs_diff(ref, cur)
        res = al.estimate(cur)
        if not res.ok:
            continue
        after = mean_abs_diff(ref, al.warp_frame(cur, res.matrix))
        assert after <= before + 0.01, (
            f"shift {shift}: alignment made it worse ({before:.2f} -> {after:.2f}) "
            f"{res.describe()}")


def test_small_shift_is_not_flagged_as_aliasing():
    ref = lot_image()
    cur = lot_image(shift=(6, -3))
    al = Aligner(ref, motion="translation", bay_pitch=70.0)
    res = al.estimate(cur)
    assert res.ok
    assert not res.warnings, res.describe()


# --------------------------------------------------------------------------- #
# rotation and scale need a richer model
# --------------------------------------------------------------------------- #

def test_rotation_needs_the_euclidean_model_and_then_works():
    ref = lot_image()
    cur = lot_image(angle=1.8)
    al = Aligner(ref, motion="euclidean")
    res = al.estimate(cur)
    assert res.ok, res.describe()
    assert abs(res.angle_deg - 1.8) < 0.6, res.describe()
    aligned = al.warp_frame(cur, res.matrix)
    assert mean_abs_diff(ref, aligned) < mean_abs_diff(ref, cur) * 0.6, res.describe()


def test_translation_model_reports_no_angle():
    ref = lot_image()
    cur = lot_image(shift=(5, 5))
    al = Aligner(ref, motion="translation")
    res = al.estimate(cur)
    assert res.ok
    assert abs(res.angle_deg) < 1e-6 and abs(res.scale - 1.0) < 1e-9, res.describe()


def test_large_rotation_is_rejected():
    ref = lot_image()
    cur = lot_image(angle=25.0)
    al = Aligner(ref, motion="euclidean")
    res = al.estimate(cur)
    assert not res.ok, res.describe()


def test_frame_size_mismatch_is_refused():
    ref = lot_image()
    al = Aligner(ref, motion="translation")
    res = al.estimate(cv2.resize(ref, (400, 300)))
    assert not res.ok
    assert "size" in res.reason


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
