"""Occupancy detection: how a bay looks now versus how it looked empty.

Deliberately the simplest thing that can work, and nothing more:

    empty reference  ->  current ROI
                             |
                        absdiff (grayscale, blurred)
                             |
                        threshold  ->  morphology
                             |
                        changed_ratio  ->  EMPTY / OCCUPIED

No AI, no feature matching, no background modelling library. ``changed_ratio`` is
the fraction of the spot's *inner* detection area whose pixels differ from the
reference by more than ``pixel_threshold``; the spot is OCCUPIED when that
fraction reaches ``changed_ratio_threshold``.

The threshold is not supposed to be guessed. ``tests/fit_threshold.py`` reads the
recorded acceptance cases and reports the value that separates empty from
occupied with the largest margin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from spot_config import Spot

EMPTY = "EMPTY"
OCCUPIED = "OCCUPIED"


@dataclass
class OccupancyParams:
    """Everything that can be tuned, in one place, with no hidden constants."""

    # Spatial pre-filter. Small blur suppresses JPEG noise without smearing a car
    # away; must be odd.
    blur_ksize: int = 5
    # A pixel counts as "changed" when absdiff exceeds this (0-255).
    pixel_threshold: int = 25
    # Fraction of the inner area that must change for the bay to count occupied.
    changed_ratio_threshold: float = 0.15
    # Morphology: remove speckle, then join nearby changed regions.
    open_ksize: int = 3
    close_ksize: int = 5
    # Match the frame's brightness and contrast to the reference before differencing. Measured:
    # without it, a global shift of +-26 grey levels (or a 20% gain change - the camera's own
    # auto-exposure moving is enough) put EVERY bay over the per-pixel threshold at once, so all
    # 48 read OCCUPIED. The matching is a robust gain+offset over the whole frame, which a car
    # (a small part of the picture) cannot move, and it leaves local changes intact.
    match_lighting: bool = True
    # Below this per-bay median residual the bay gets a local offset correction as well, for a
    # shadow that covers one bay and not the others. Above it the median is the car, not the
    # light, so it is left alone - otherwise a big dark car would be "explained away".
    local_offset_max: float = 26.0

    def normalized(self) -> "OccupancyParams":
        blur = self.blur_ksize if self.blur_ksize % 2 == 1 else self.blur_ksize + 1
        return OccupancyParams(
            blur_ksize=max(blur, 1),
            pixel_threshold=int(np.clip(self.pixel_threshold, 1, 255)),
            changed_ratio_threshold=float(np.clip(self.changed_ratio_threshold, 0.0, 1.0)),
            open_ksize=max(self.open_ksize, 0),
            close_ksize=max(self.close_ksize, 0),
            match_lighting=bool(self.match_lighting),
            local_offset_max=float(self.local_offset_max),
        )


@dataclass
class SpotResult:
    spot_id: str
    state: str
    changed_ratio: float
    changed_pixels: int = 0
    inner_area: int = 0
    # Cropped diff mask, only filled in when a debug image is requested.
    diff_mask: Optional[np.ndarray] = field(default=None, repr=False)

    def __str__(self) -> str:
        return f"{self.spot_id} {self.state} ({self.changed_ratio:.3f})"


class OccupancyDetector:
    def __init__(self, params: Optional[OccupancyParams] = None):
        self.params = (params or OccupancyParams()).normalized()
        self._ref_roi: Dict[str, np.ndarray] = {}
        self._spots: List[Spot] = []
        self._ref_shape: Optional[Tuple[int, int]] = None
        # The full reference frame as grey, for matching a frame's lighting before differencing.
        self._ref_gray: Optional[np.ndarray] = None
        self._ref_levels: Tuple[float, float] = (0.0, 1.0)     # (median, inter-quartile spread)

    # -- reference ----------------------------------------------------------

    @property
    def has_reference(self) -> bool:
        return bool(self._ref_roi)

    def set_reference(self, frame: np.ndarray, spots: List[Spot]) -> None:
        """Cache the empty-bay appearance for every spot, cropped to its ROI."""
        self._ref_roi.clear()
        self._spots = spots
        self._ref_shape = frame.shape[:2]
        self._ref_gray = self._prep(frame)
        self._ref_levels = self._levels(self._ref_gray)
        for s in spots:
            s.prepare(frame.shape[:2])
            x0, y0, x1, y1 = s.roi
            self._ref_roi[s.id] = self._prep(frame[y0:y1, x0:x1])

    def _prep(self, bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.params.blur_ksize > 1:
            gray = cv2.GaussianBlur(gray, (self.params.blur_ksize,) * 2, 0)
        return gray

    @staticmethod
    def _levels(gray: np.ndarray) -> Tuple[float, float]:
        """Robust brightness and contrast of a frame: median and inter-quartile spread.

        Both are medians of large samples, so a car - a small part of the picture, and only ever
        part of one bay - cannot move them, while a change in the light moves everything.
        """
        flat = gray.reshape(-1)
        q1, mid, q3 = np.percentile(flat, (25, 50, 75))
        return float(mid), float(max(q3 - q1, 1e-3))

    def match_lighting(self, gray: np.ndarray) -> np.ndarray:
        """Bring a frame to the reference's brightness and contrast.

        One global gain and offset per frame, not per bay: the light changes the whole picture,
        a car changes one bay, and fitting per bay would let a large car fit itself away.
        """
        if not self.params.match_lighting or self._ref_gray is None:
            return gray
        if gray.shape != self._ref_gray.shape:
            return gray
        mid_c, spread_c = self._levels(gray)
        mid_r, spread_r = self._ref_levels
        out = (gray.astype(np.float32) - mid_c) * (spread_r / spread_c) + mid_r
        return np.clip(out, 0, 255).astype(np.uint8)

    # -- per-frame ----------------------------------------------------------

    def evaluate_spot(self, frame: np.ndarray, spot: Spot,
                      want_debug: bool = False) -> SpotResult:
        if spot.id not in self._ref_roi:
            raise RuntimeError(f"spot {spot.id}: no reference captured for it")
        if spot.mask_roi is None:
            spot.prepare(frame.shape[:2])

        x0, y0, x1, y1 = spot.roi
        # Match the whole frame to the reference first, then take this bay's window from it.
        matched = self.match_lighting(self._prep(frame))
        cur = matched[y0:y1, x0:x1]
        ref = self._ref_roi[spot.id]
        if cur.shape != ref.shape:
            # Camera resolution changed (e.g. someone edited the firmware config).
            raise RuntimeError(
                f"frame geometry changed: reference ROI is {ref.shape}, current is {cur.shape}. "
                f"Re-capture the reference with 'r'.")

        # A shadow over one bay only: correct a local offset, but only while it is small enough to
        # be light rather than a car sitting there.
        if self.params.match_lighting:
            local = float(np.median(cur.astype(np.float32) - ref.astype(np.float32)))
            if 0.0 < abs(local) <= self.params.local_offset_max:
                cur = np.clip(cur.astype(np.float32) - local, 0, 255).astype(np.uint8)

        diff = cv2.absdiff(ref, cur)
        _, mask = cv2.threshold(diff, self.params.pixel_threshold, 255, cv2.THRESH_BINARY)

        if self.params.open_ksize > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (self.params.open_ksize,) * 2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        if self.params.close_ksize > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (self.params.close_ksize,) * 2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        inside = cv2.countNonZero(cv2.bitwise_and(mask, spot.mask_roi))
        area = spot.inner_area()
        ratio = inside / area if area > 0 else 0.0

        state = OCCUPIED if ratio >= self.params.changed_ratio_threshold else EMPTY
        return SpotResult(
            spot_id=spot.id, state=state, changed_ratio=ratio,
            changed_pixels=inside, inner_area=area,
            diff_mask=mask if want_debug else None,
        )

    def evaluate(self, frame: np.ndarray, spots: List[Spot],
                 want_debug: bool = False) -> Dict[str, SpotResult]:
        return {s.id: self.evaluate_spot(frame, s, want_debug) for s in spots}

    # -- diagnostics --------------------------------------------------------

    @staticmethod
    def describe(frame: np.ndarray) -> str:
        """Frame quality numbers, for diagnosing 'why is detection unstable'."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        return (f"{frame.shape[1]}x{frame.shape[0]} "
                f"brightness={gray.mean():.1f} contrast={gray.std():.1f} "
                f"sharpness(lapvar)={lap:.1f} "
                f"overexposed={np.mean(gray > 245) * 100:.2f}% "
                f"underexposed={np.mean(gray < 10) * 100:.2f}%")
