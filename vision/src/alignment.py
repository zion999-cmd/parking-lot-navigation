"""Align the live image to the frame the bays were calibrated on.

The camera is meant to be fixed, but a nudge of the tripod or of the sandbox moves
the picture by a few pixels, and then every ROI is slightly off its bay. This
module estimates that displacement and undoes it, so the polygons never have to
change.

    reference frame  (bays were drawn on this)
          +
    current frame
          |
    cv2.findTransformECC  ->  M  (maps the current frame into reference coordinates)
          |
    warp the current frame with M  ->  ROIs apply unchanged
    (or invert M to move the polygons onto the current frame)

Two things matter for correctness here:

* **Translation is the default.** A tripod nudge is mostly translation, and the
  cheaper model is harder to get wrong.
* **Every estimate is checked.** This scene is a grid of identical, evenly spaced
  bays, so a registration error of exactly one bay pitch looks like a perfect
  match. A shift that lands near a multiple of the bay pitch is reported as
  suspicious, and an implausibly large shift is rejected outright - because
  silently moving all the ROIs one bay sideways is far worse than doing nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

MOTION_MODELS = {
    "translation": cv2.MOTION_TRANSLATION,
    "euclidean": cv2.MOTION_EUCLIDEAN,
    "affine": cv2.MOTION_AFFINE,
}

# A shift this large is not a nudge any more; something else is wrong.
DEFAULT_MAX_SHIFT_FRAC = 0.12
# Threshold on the *structure* correlation (see structure_correlation below).
#
# Measured on the real calibrated frame (data/debug/frame.jpg), shifting it by a
# known amount and letting ECC correct it:
#
#   shift      no-move   after ECC   ECC's dx vs truth
#   3, 2        0.522      0.973       +3.0  correct
#   7, -4       0.190      0.974       +7.0  correct
#   15, 8       0.050      0.957      +15.3  correct
#   40, 0       0.504      0.545      +27.7  wrong
#   70, 0       0.449      0.636      +25.0  wrong
#   150, 0      0.361      0.623     +105.0  wrong
#
# So a genuine alignment lands at 0.95-0.97 and every failed one at 0.54-0.64.
# 0.8 sits 0.16 away from both, which is why a nudge up to ~15 px (a third of a bay)
# is corrected and anything larger is reported as "cannot align" rather than
# applied as a wrong correction.
#
# Two things these numbers also show. First, a plain brightness correlation is
# useless here: the picture is mostly flat floor, and a 150 px displacement still
# scored 0.96 on brightness. Second, the correlation is *periodic* on an evenly
# spaced grid - a whole-bay shift scores higher than a 7 px one - so it is only
# ever used to validate a candidate, never to choose between candidates.
DEFAULT_MIN_CORRELATION = 0.8
# A shift within this fraction of the bay pitch is probably one bay of aliasing.
PITCH_ALIAS_TOLERANCE = 0.25


@dataclass
class AlignResult:
    ok: bool
    matrix: np.ndarray = field(default_factory=lambda: np.eye(2, 3, dtype=np.float32))
    correlation: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    angle_deg: float = 0.0
    scale: float = 1.0
    reason: str = ""
    # Actionable: something the operator should look at.
    warnings: List[str] = field(default_factory=list)
    # Informational: which estimator ran and what it scored.
    notes: List[str] = field(default_factory=list)

    def describe(self) -> str:
        status = "ok" if self.ok else f"rejected ({self.reason})"
        text = (f"{status}  shift dx={self.dx:+.1f} dy={self.dy:+.1f}px  "
                f"rot={self.angle_deg:+.2f}deg  scale={self.scale:.4f}  "
                f"corr={self.correlation:.3f}")
        if self.warnings:
            text += "  [" + "; ".join(self.warnings) + "]"
        if self.notes:
            text += "  (" + ", ".join(self.notes) + ")"
        return text


def _to_gray(frame: np.ndarray, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if size is not None:
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32) / 255.0


def structure_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation of the two images' gradient magnitude, in [-1, 1].

    Deep breath: on a car park the picture is mostly flat floor, so a plain
    brightness correlation barely changes when the image shifts - a 150 px
    displacement still scored 0.96, which is useless as a check. Comparing edge
    energy instead asks the question that matters, "do the lines still fall on the
    lines", and separates cleanly.
    """
    def mag(x: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    ma, mb = mag(a), mag(b)
    if ma.std() < 1e-6 or mb.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(ma.ravel(), mb.ravel())[0, 1])


def _decompose(matrix: np.ndarray) -> Tuple[float, float, float, float]:
    """dx, dy, rotation in degrees, uniform scale."""
    m = np.asarray(matrix, dtype=np.float64)
    dx, dy = float(m[0, 2]), float(m[1, 2])
    a, b = float(m[0, 0]), float(m[1, 0])
    scale = float(np.hypot(a, b))
    angle = float(np.degrees(np.arctan2(b, a)))
    return dx, dy, angle, scale


class Aligner:
    """Estimate and apply the alignment between a reference and the live frame."""

    def __init__(self, reference: np.ndarray, motion: str = "translation",
                 max_shift_frac: float = DEFAULT_MAX_SHIFT_FRAC,
                 min_correlation: float = DEFAULT_MIN_CORRELATION,
                 working_width: int = 400,
                 bay_pitch: Optional[float] = None):
        if motion not in MOTION_MODELS:
            raise ValueError(f"unknown motion model {motion!r}; "
                             f"pick one of {sorted(MOTION_MODELS)}")
        self.motion = motion
        self.max_shift_frac = max_shift_frac
        self.min_correlation = min_correlation
        self.working_width = working_width
        self.bay_pitch = bay_pitch
        self.reference_frame = reference
        self._shape = reference.shape[:2]
        scale = working_width / float(reference.shape[1])
        self._work_size = (working_width, max(int(round(reference.shape[0] * scale)), 8))
        self._ref_small = _to_gray(reference, self._work_size)
        self._hann = cv2.createHanningWindow(
            (self._work_size[0], self._work_size[1]), cv2.CV_32F)

    # -- estimation ---------------------------------------------------------

    def _score(self, cur_small: np.ndarray, matrix: Optional[np.ndarray]) -> float:
        """Structure correlation after applying ``matrix`` (None = do not move)."""
        m = np.eye(2, 3, dtype=np.float32) if matrix is None else matrix
        aligned = cv2.warpAffine(cur_small, m,
                                 (self._work_size[0], self._work_size[1]),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        return structure_correlation(self._ref_small, aligned)

    def estimate(self, frame: np.ndarray) -> AlignResult:
        """Estimate the transform that maps ``frame`` into reference coordinates.

        An explicit policy, not a beauty contest between estimators:

        1. ECC first. Its basin is only a few pixels, which is exactly the problem
           being solved (a camera nudge), and that small basin is what guarantees it
           cannot jump a bay. Accepted if the result genuinely lines up.
        2. Only if ECC did not line up, try the global phase-correlation estimate -
           and require it to beat *doing nothing* as well. On an evenly spaced grid a
           whole-bay shift is visually identical to no shift, and "do nothing"
           actually scores higher than the correct one-bay correction (0.956 against
           0.925). A global estimate that does not improve on doing nothing is about
           to jump to the neighbouring bay, so it is refused rather than applied.

        The consequence is deliberate: a displacement of about one whole bay is
        reported as "cannot align" instead of being silently corrected into
        something that merely looks right.
        """
        if frame.shape[:2] != self._shape:
            return AlignResult(ok=False, reason=f"frame size {frame.shape[:2]} != "
                                                f"reference {self._shape}")
        cur_small = _to_gray(frame, self._work_size)
        baseline = self._score(cur_small, None)
        notes = [f"no-move={baseline:.3f}"]

        chosen: Optional[Tuple[str, np.ndarray, float]] = None

        m_ecc, cc_ecc, src_ecc = self._ecc(self._ref_small, cur_small)
        if m_ecc is not None:
            score = self._score(cur_small, m_ecc)
            notes.append(f"ecc={score:.3f}")
            if score >= self.min_correlation:
                chosen = (src_ecc, m_ecc, score)

        if chosen is None:
            m_pc, r_pc, src_pc = self._phase_correlate(self._ref_small, cur_small,
                                                      "ecc did not line up")
            if m_pc is not None:
                score = self._score(cur_small, m_pc)
                notes.append(f"phase-corr={score:.3f}")
                if score >= self.min_correlation and score > baseline:
                    chosen = (src_pc, m_pc, score)

        if chosen is None:
            return AlignResult(
                ok=False, correlation=baseline,
                reason="no candidate aligned the frame",
                notes=notes)

        source, matrix, correlation = chosen
        matrix = self._rescale(matrix)
        dx, dy, angle, scale = _decompose(matrix)
        result = AlignResult(ok=True, matrix=matrix, correlation=correlation,
                             dx=dx, dy=dy, angle_deg=angle, scale=scale)
        result.notes.append(f"estimator {source}; " + ", ".join(notes))
        self._check(result, source)
        return result

    def _ecc(self, ref: np.ndarray, cur: np.ndarray) -> Tuple[Optional[np.ndarray], float, str]:
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
        try:
            # Argument order matters and is easy to get backwards. Measured on a
            # synthetic frame shifted by (+7, -4) px, with the estimate scaled back
            # to full resolution and applied as warpAffine(cur, M):
            #   findTransformECC(ref, cur, M) -> tx=+7.0  mean|ref-cur| 20.5 -> 22.0
            #   findTransformECC(cur, ref, M) -> tx=-6.9  mean|ref-cur| 20.5 ->  0.5
            # so the current frame must come FIRST for M to map cur onto ref.
            correlation, warp = cv2.findTransformECC(
                cur, ref, warp, MOTION_MODELS[self.motion], criteria, None, 5)
        except cv2.error as exc:
            return None, 0.0, f"ECC failed ({str(exc).splitlines()[0][:60]})"
        return warp, float(correlation), f"ECC/{self.motion}"

    def _phase_correlate(self, ref: np.ndarray, cur: np.ndarray, why: str
                         ) -> Tuple[Optional[np.ndarray], float, str]:
        """Global translation estimate. Jumps far better than ECC, and can overshoot."""
        try:
            (dx, dy), response = cv2.phaseCorrelate(ref, cur, self._hann)
        except cv2.error as exc:
            return None, 0.0, f"phase correlation failed ({str(exc)[:60]})"
        # phaseCorrelate reports how far cur is shifted from ref, so mapping cur
        # onto ref needs the opposite sign.
        warp = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
        return warp, float(response), f"phase-correlation fallback ({why})"

    def _rescale(self, matrix: np.ndarray) -> np.ndarray:
        """Scale a matrix estimated on the downscaled images back to full size.

        M_full = S * M_small * S^-1, with S = scaling from small to full.
        """
        s = self._shape[1] / float(self._work_size[0])
        if abs(s - 1.0) < 1e-9:
            return np.asarray(matrix, dtype=np.float32)
        sinv = np.diag([1.0 / s, 1.0 / s, 1.0])
        sdiag = np.diag([s, s, 1.0])
        full = sdiag @ np.vstack([matrix, [0, 0, 1]]) @ sinv
        return full[:2, :].astype(np.float32)

    def _check(self, res: AlignResult, source: str) -> None:
        h, w = self._shape
        max_shift = self.max_shift_frac * max(h, w)

        # ``correlation`` is the structure correlation between the reference and the
        # frame after applying this transform, so it measures alignment directly
        # rather than trusting either estimator's own score.
        if res.correlation < self.min_correlation:
            res.ok = False
            res.reason = (f"correlation {res.correlation:.3f} < {self.min_correlation} "
                          f"after applying the transform ({source})")
            return

        if abs(res.dx) > max_shift or abs(res.dy) > max_shift:
            res.ok = False
            res.reason = (f"shift ({res.dx:+.0f}, {res.dy:+.0f}) px exceeds "
                          f"{self.max_shift_frac * 100:.0f}% of the frame")
            return

        if self.motion in ("euclidean", "affine"):
            if abs(res.angle_deg) > 8.0:
                res.ok = False
                res.reason = f"rotation {res.angle_deg:+.1f} deg is too large"
                return
            if not 0.94 <= res.scale <= 1.06:
                res.ok = False
                res.reason = f"scale {res.scale:.3f} is too far from 1"
                return

        # The one failure that would look perfect: locking onto the next bay.
        if self.bay_pitch:
            for sign, comp in ((res.dx, "x"), (res.dy, "y")):
                ratio = abs(sign) / self.bay_pitch
                nearest = round(ratio)
                if nearest >= 1 and abs(ratio - nearest) <= PITCH_ALIAS_TOLERANCE:
                    res.warnings.append(
                        f"shift {comp}={sign:+.1f}px is about {nearest} bay "
                        f"pitch(es) ({self.bay_pitch:.0f}px) - check the overlay, "
                        f"this may be locked onto the neighbouring bay")

    # -- application --------------------------------------------------------

    def warp_frame(self, frame: np.ndarray, matrix: Optional[np.ndarray] = None
                   ) -> np.ndarray:
        """Bring ``frame`` into reference coordinates, so the ROIs apply as drawn."""
        m = matrix if matrix is not None else np.eye(2, 3, dtype=np.float32)
        return cv2.warpAffine(frame, m, (self._shape[1], self._shape[0]),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def matrix_to_current(self, matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """Inverse of the estimated matrix.

        A point that sits at ``p`` in the reference appears at ``M^-1 . p`` in the
        current frame. Use this to move the polygons onto the live image instead of
        moving the image onto the polygons.
        """
        m = np.asarray(matrix if matrix is not None
                       else np.eye(2, 3, dtype=np.float32), dtype=np.float64)
        inv = cv2.invertAffineTransform(m)
        return inv.astype(np.float32)


def transform_points(points: Sequence[Sequence[float]], matrix: np.ndarray
                     ) -> np.ndarray:
    """Apply an affine matrix to a list of [x, y] points."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    m = np.asarray(matrix, dtype=np.float64)
    out = pts @ m[:, :2].T + m[:, 2]
    return out
