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
    #
    # Raised from 0.15 to 0.40 on request: 0.15 was firing on bays that were only marginally
    # different, and the strip showed them occupied.
    #
    # Measured on 5 live frames against the production reference (10 October, still good inside the
    # bays - the whole-frame difference of 36% is the road and the lights, not the bays):
    #
    #     threshold 0.15   occupied  3/48   ratio median 0.009   max 0.293
    #     threshold 0.40   occupied  0/48
    #
    # so the three bays reading 0.15-0.29 (32, 43, 44 on the day) now read EMPTY. If those are
    # genuinely occupied, this threshold hides them; if they are the flicker that prompted the
    # change, it is the right trade. A reference captured at 11:53 measured *worse* inside the bays
    # (median 0.071, 15 bays over 0.15) and must not be promoted - a fresher reference is not
    # automatically a better one, and the whole-frame difference says nothing about the bays.
    changed_ratio_threshold: float = 0.40
    # Morphology: remove speckle, then join nearby changed regions.
    open_ksize: int = 3
    close_ksize: int = 5

    def normalized(self) -> "OccupancyParams":
        blur = self.blur_ksize if self.blur_ksize % 2 == 1 else self.blur_ksize + 1
        return OccupancyParams(
            blur_ksize=max(blur, 1),
            pixel_threshold=int(np.clip(self.pixel_threshold, 1, 255)),
            changed_ratio_threshold=float(np.clip(self.changed_ratio_threshold, 0.0, 1.0)),
            open_ksize=max(self.open_ksize, 0),
            close_ksize=max(self.close_ksize, 0),
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

    # -- reference ----------------------------------------------------------

    @property
    def has_reference(self) -> bool:
        return bool(self._ref_roi)

    def set_reference(self, frame: np.ndarray, spots: List[Spot]) -> None:
        """Cache the empty-bay appearance for every spot, cropped to its ROI."""
        self._ref_roi.clear()
        self._spots = spots
        self._ref_shape = frame.shape[:2]
        for s in spots:
            s.prepare(frame.shape[:2])
            x0, y0, x1, y1 = s.roi
            self._ref_roi[s.id] = self._prep(frame[y0:y1, x0:x1])

    def _prep(self, bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.params.blur_ksize > 1:
            gray = cv2.GaussianBlur(gray, (self.params.blur_ksize,) * 2, 0)
        return gray

    # -- per-frame ----------------------------------------------------------

    def evaluate_spot(self, frame: np.ndarray, spot: Spot,
                      want_debug: bool = False) -> SpotResult:
        if spot.id not in self._ref_roi:
            raise RuntimeError(f"spot {spot.id}: no reference captured for it")
        if spot.mask_roi is None:
            spot.prepare(frame.shape[:2])

        x0, y0, x1, y1 = spot.roi
        cur = self._prep(frame[y0:y1, x0:x1])
        ref = self._ref_roi[spot.id]
        if cur.shape != ref.shape:
            # Camera resolution changed (e.g. someone edited the firmware config).
            raise RuntimeError(
                f"frame geometry changed: reference ROI is {ref.shape}, current is {cur.shape}. "
                f"Re-capture the reference with 'r'.")

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
