"""Where the hero car is, and which way its axis runs.

Why colour and shape rather than a marker: the job is to know the car's position and
heading. A marker gives that precisely, and also tells the front from the back - but at
800x600 across a whole sandbox a marker has to be about half the car's width and flat towards
the camera to be readable at all, and on this car that is physically unreasonable. Colour and
shape need no hardware and no reference frame, which also makes them immune to the two things
that were wrecking measurements here: lighting changes and sub-pixel shifts of the painted bay
lines.

Measured on the real thing, with the car parked and the camera still:

    position   x +- 0.25 px, y +- 0.11 px      (a navigator needs +- 5 px)
    heading    +- 0.8 deg                      (a navigator needs +- 15 deg)

And it agrees with something it was never told: with the car left on a road, the heading
reads 0.0 deg while that road segment runs at -0.17 deg.

The honest limit is in the name of the field: `heading` is the *axis* of the car, in 0..180,
with no front. A car pointing east and the same car pointing west both read 0.0. That is what
a marker would buy, and nothing here can replace it.

The thresholds below were measured from a frame rather than chosen: the red car sits at hue
0..2 and 176..179 with S >= 90 and V >= 60, while the orange-red trim along the top of the
enclosure sits at hue 11..15. The first attempt used hue <= 12 and turned that trim into the
largest "red object" in the picture.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# The sign template, sampled from the real car: tests/capture_vehicle_profile.py builds it.
# 24 brightness bins read rear -> front (bin 0 at the boot, bin 23 at the nose).
PROFILE_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "vehicle_profile.json")

# Red wraps around the hue circle, so it takes two ranges. Tight on purpose: see above.
RED_HUE_RANGES: Tuple[Tuple[Tuple[int, int, int], Tuple[int, int, int]], ...] = (
    ((0, 90, 60), (3, 255, 255)),
    ((176, 90, 60), (180, 255, 255)),
)

# ---------------------------------------------------------------------------
# Which end of the car is the front.
#
# Decided by the user looking at an annotated frame: the end marked green was the front, and
# the green marker was on the *wider* end. Nothing here can work that out on its own - it is a
# fact about a particular toy car, not about pixels - so it is stated once and used thereafter.
#
# It cross-checks with the other asymmetric cue: the dark band (the windscreen, presumably)
# sits near the wider end in both orientations, which is why there is really only one cue here
# rather than two independent ones.
FRONT_IS_THE_WIDER_END = True

# How much wider one end has to be before the front is claimed at all. Measured on this car
# either side of a 180 degree turn: +0.17 and -0.15, with a frame-to-frame spread of +-0.05.
# Below this the honest answer is "front unknown" rather than a coin flip.
MIN_TAPER = 0.04

# Below this a red blob is a sticker or a reflection, not a car.
DEFAULT_MIN_AREA = 80
# Two pieces of the same car are joined if they are this close. The body breaks into pieces
# wherever a dark band crosses it, and a centroid that jumps between two halves is not a
# position.
CLOSE_KERNEL = 21
# A blob that runs into the edge of the picture is the enclosure, not a car.
BORDER_MARGIN = 2


@dataclass
class VehiclePose:
    """One sighting of the car.

    `heading` is the long axis in degrees 0..180 and has no front. `front_heading` is the
    direction the car actually faces in 0..360, and is only filled in when the shape is
    asymmetric enough to say which end is which - `front_is_known` says whether to trust it.
    """

    x: float
    y: float
    heading: float
    area: int
    elongation: float
    fill: float
    bbox: Tuple[int, int, int, int]
    # Second opinion on the heading: the two agree to about half a degree when the mask is
    # clean, so a large disagreement is a sign the mask is not a car.
    heading_pca: float = 0.0
    # How much wider one end of the shape is than the other. Positive means the end in the
    # +heading direction is the wider one.
    taper: float = 0.0
    # Where the darkest stretch sits along the axis, as a fraction from the -axis end, and how
    # much darker it is than the rest. Both are diagnostic; they feed front_evidence.
    band_pos: Optional[float] = None
    band_contrast: float = 0.0
    # Grey levels between the two ends after removing the lighting trend. Positive means the
    # +axis end is brighter, which by the car's paint pattern means the front is that way.
    ends_grey: float = 0.0
    # Full-profile sign, the replacement for ends_grey: how well the whole 24-bin profile
    # matches the sampled template read rear->front, and the same read the other way; margin is
    # the gap between them.
    forward_score: Optional[float] = None
    reverse_score: Optional[float] = None
    score_margin: Optional[float] = None
    # Signed and normalised: +1 means the front is at the +axis end.
    front_evidence: Optional[float] = None
    front_heading: Optional[float] = None
    front_is_known: bool = False

    @property
    def centre(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def as_dict(self) -> dict:
        out = {"id": "hero", "x": round(self.x, 1), "y": round(self.y, 1),
               "heading": round(self.heading, 1), "visible": True,
               "heading_is_axis_only": True,
               "front_is_known": bool(self.front_is_known)}
        if self.front_is_known and self.front_heading is not None:
            # The axis, plus the one fact a person had to tell us: which end is the front.
            out["front_heading"] = round(self.front_heading, 1)
        else:
            out["front_heading"] = None
        return out


def red_mask(hsv: np.ndarray, close_kernel: int = CLOSE_KERNEL) -> np.ndarray:
    """The red regions, with the car's holes filled in.

    Filling matters: windows, the shadowed side and dark trim drop out of a colour mask and
    leave holes inside the blob, and PCA weighs every pixel, so holes that come and go make
    the axis jump between two orientations. Flood filling from the border and inverting keeps
    the outline and loses the holes.
    """
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in RED_HUE_RANGES:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small)
    if close_kernel > 1:
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)))
    holes = mask.copy()
    cv2.floodFill(holes, np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8),
                  (0, 0), 255)
    return mask | cv2.bitwise_not(holes)


def touches_border(bbox: Sequence[int], shape: Sequence[int],
                   margin: int = BORDER_MARGIN) -> bool:
    x, y, w, h = bbox
    return (x <= margin or y <= margin
            or x + w >= shape[1] - margin or y + h >= shape[0] - margin)


# ---------------------------------------------------------------------------
# Which end is the front.
#
# Two cues, and the weaker one turned out to be the one that was used first.
#
#   the dark band    the windscreen, presumably. Measured on this car at 65% along the axis
#                    with 45 grey levels of contrast and a spread of 2%. On the other car it
#                    sat at 30% or 67% depending on which way round it was.
#   the taper        how much wider one end of the silhouette is. -0.10 to -0.17 on the first
#                    car, but +0.012 on the car measured now, where the two ends differ by one
#                    pixel. Silhouettes are easy to make symmetric by accident: a shadow across
#                    one end, or simply looking at the car from a slightly different angle.
#
# So the band leads and the taper is the fallback. Both are expressed as a signed "evidence"
# in -1..1 pointing along the axis, normalised by how strong that cue usually gets, so one set
# of thresholds works for either.
# The ends of the car, which is the cue that works.
#
# The car is red, black, red, black from front to rear - bonnet, front glass, roof, rear glass.
# That is why the previous cue failed: such a pattern has *two* dark bands, so "where is the
# darkest band" picks whichever of the two happens to be darker at that moment. Tilting the car
# or pushing it into different light swaps which one that is, the band position jumps from one
# end to the other, and the front flips - although the car's front has not moved at all.
#
# The two ends do not swap: red at the front against black at the rear. Measured on the real
# car, 13.8 grey levels apart with 0.3 to 0.6 of frame-to-frame noise, and the brighter end is
# the front.
#
# A lighting gradient across the car would also make one end look brighter than the other, so
# the profile's linear trend is removed first. What is compared is each end's tone *relative to
# that trend* - the pattern rather than the lighting.
ENDS_MIN_GREY = 4.0          # grey levels; below this there is no usable end-to-end difference
ENDS_SCALE = 12.0            # ...and what counts as fully convincing
BAND_MIN_CONTRAST = 15.0     # only used to report the band now; it no longer decides anything
BAND_SCALE = 0.5
TAPER_SCALE = 0.20           # fallback cue, for a car with no usable pattern
KEEP_EVIDENCE = 0.25         # below this, do not claim a front at all
FLIP_EVIDENCE = 0.60         # to change an established front, this much the other way...

# To turn the front round, the evidence for the other end has to be much stronger than the
# evidence needed to keep the one already known, and it has to hold for several frames.
#
# The reason is the one physical fact in this file: a car cannot swap ends in a few frames.
# The smoothed taper's noise is about 0.02, so it can never reach 0.10 by accident, while a real
# 180 degree turn crosses it immediately and stays there. This is a Schmitt trigger: the gate
# for *keeping* a front is MIN_TAPER, the gate for *changing* one is FLIP_TAPER.
FLIP_TAPER = 0.10
FLIP_FRAMES = 3
# A brief loss - an occlusion, motion blur, a hand across the lens - must not erase what is
# known. Only a run of misses means it is a different situation.
MAX_MISSES = 5


class VehicleTracker:
    """Smooth the readings, and refuse to believe the car has turned round without cause.

    Two separate problems, and it took measuring the car driving to see the second one.

    Noise. The taper sits at -0.12 and wobbles by +-0.05 frame to frame, because the colour
    mask moves a pixel at its edges and JPEG adds its own. Gating on a single frame dropped the
    front reading about one frame in six while the car was stationary. A short average fixes
    that: a car's shape changes only when the car does.

    Resetting too eagerly. The first version treated any jump of more than 40 px between frames
    as "the car was moved by hand" and cleared everything. At ten frames a second a toy car
    crossing the sandbox in two seconds moves 40 px per frame - so driving the car wiped the
    average and the front state on almost every frame, which is exactly when the readings were
    seen to flicker. A jump now re-starts only the position; shape and front are properties of
    the car and do not depend on where it is. A run of missed frames is what resets everything.

    Flipping. Even with the average, a moving car's shape is less steady, and the derived front
    could cross the gate and swap. Since a car cannot swap ends in a few frames, a swap has to
    earn it: FLIP_TAPER for FLIP_FRAMES consecutive frames. When the car really is turned
    around the crossing is immediate and sustained, so it is followed within those few frames.
    """

    def __init__(self, alpha: float = 0.3, position_alpha: float = 0.5,
                 keep_evidence: float = KEEP_EVIDENCE, flip_evidence: float = FLIP_EVIDENCE,
                 flip_frames: int = FLIP_FRAMES, max_misses: int = MAX_MISSES,
                 position_reset_px: float = 40.0):
        self.alpha = alpha
        self.position_alpha = position_alpha
        self.keep_evidence = keep_evidence
        self.flip_evidence = flip_evidence
        self.flip_frames = flip_frames
        self.max_misses = max_misses
        self.position_reset_px = position_reset_px
        self._evidence: Optional[float] = None
        self._ax = 0.0
        self._ay = 0.0
        self._x: Optional[float] = None
        self._y: Optional[float] = None
        # Which end is the front: True means the +axis direction, and it only changes when the
        # evidence says so strongly enough, for long enough.
        self._front_is_plus: Optional[bool] = None
        self._opposite_streak = 0
        self._misses = 0

    def reset(self) -> None:
        """Start again: the car has been out of sight long enough to be a new situation."""
        self._evidence = None
        self._x = self._y = None
        self._front_is_plus = None
        self._opposite_streak = 0
        self._misses = 0

    def update(self, pose: Optional[VehiclePose]) -> Optional[VehiclePose]:
        """One sighting in, a smoothed sighting out. None in, None out."""
        if pose is None:
            self._misses += 1
            if self._misses >= self.max_misses:
                self.reset()
            return None
        self._misses = 0

        # A jump too big to be driving means the car was moved by hand. That invalidates where
        # it is, but not what shape it is - so only the position starts again.
        if (self._x is not None
                and math.hypot(pose.x - self._x, pose.y - self._y) > self.position_reset_px):
            self._x = self._y = None

        a = self.alpha
        # The evidence is smoothed, not the taper: it is what decides, and it has the same
        # meaning whichever cue produced it. A frame with no usable cue leaves it alone rather
        # than dragging it towards zero.
        if pose.front_evidence is not None:
            self._evidence = (pose.front_evidence if self._evidence is None
                              else a * pose.front_evidence + (1 - a) * self._evidence)
        # The axis is an angle in 0..180, where 0 and 180 are the same; doubling it makes the
        # average well defined instead of cancelling itself out.
        doubled = math.radians(2.0 * pose.heading)
        if self._ax == 0.0 and self._ay == 0.0:
            self._ax, self._ay = math.cos(doubled), math.sin(doubled)
        else:
            self._ax = a * math.cos(doubled) + (1 - a) * self._ax
            self._ay = a * math.sin(doubled) + (1 - a) * self._ay
        heading = (math.degrees(math.atan2(self._ay, self._ax)) / 2.0) % 180.0

        pa = self.position_alpha
        self._x = pose.x if self._x is None else pa * pose.x + (1 - pa) * self._x
        self._y = pose.y if self._y is None else pa * pose.y + (1 - pa) * self._y

        # Which end the car says is the front, and whether to believe it.
        #
        # The *raw* evidence drives the flip and the smoothed one only holds it. Using the
        # smoothed value for both made a real turn take about a second to register, because an
        # average needs several frames to cross zero - too slow for a car that can be turned
        # round in one movement. The raw value is safe for the flip because the trigger demands
        # strong evidence the other way for several consecutive frames, which noise does not
        # produce.
        raw = pose.front_evidence
        if raw is not None:
            wants_plus = raw > 0
            # Establishing the front needs solid evidence; keeping it does not need to be
            # re-justified every frame. Which end is the front is a property of this car, not of
            # the last frame, so a frame with a weak cue - or the smoothed evidence dipping
            # while one contrary frame is being ignored - must not make the front disappear.
            # Only sustained contrary evidence changes it, which is what the trigger below is.
            if self._front_is_plus is None and abs(raw) >= self.keep_evidence:
                self._front_is_plus = wants_plus
            elif wants_plus != self._front_is_plus and abs(raw) >= self.flip_evidence:
                self._opposite_streak += 1
                if self._opposite_streak >= self.flip_frames:
                    self._front_is_plus = not self._front_is_plus
                    self._opposite_streak = 0
            else:
                self._opposite_streak = 0

        known = self._front_is_plus is not None and pose.elongation >= 1.6
        front = ((heading if self._front_is_plus else (heading + 180.0) % 360.0)
                 if self._front_is_plus is not None else None)
        return replace(pose, x=self._x, y=self._y, heading=heading,
                       front_heading=front if known else None, front_is_known=known)


_TEMPLATE_CACHE = "unloaded"
# Measured, not chosen: the correct-vs-reverse margin on 60 frames of the real car is 0.875, so a
# confident frame maps to evidence near 1.0 and an uncertain one scales down. Re-measured any time
# by tests/capture_vehicle_profile.py --build; the number belongs to this car, not to the code.
MARGIN_SCALE = 0.9


def vehicle_template(path: str = PROFILE_TEMPLATE_PATH):
    """The sampled sign template, loaded once. None when it has not been finalized.

    Two channels chosen by measuring both orientations: saturation (glass low, paint high;
    separation 0.94) and red_fraction (where the enclosed windows are; 0.81). The cache is a dict
    of arrays, not one vector. The sentinel uses isinstance: comparing an ndarray to the string
    "unloaded" gives an element-wise boolean array rather than one bool.
    """
    global _TEMPLATE_CACHE
    if isinstance(_TEMPLATE_CACHE, str):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                values = json.load(fh).get("template", {})
            _TEMPLATE_CACHE = {k: np.asarray(v, dtype=np.float64)
                               for k, v in values.items()}
        except (OSError, ValueError):
            _TEMPLATE_CACHE = None
    return None if isinstance(_TEMPLATE_CACHE, str) else _TEMPLATE_CACHE


def _shape(values: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-spread: compare the pattern shape, not the lighting level."""
    v = np.asarray(values, dtype=np.float64)
    v = v - v.mean()
    spread = v.std()
    return v / spread if spread > 1e-9 else v


def _correlate(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two equal-length profile shapes, in -1..1."""
    return float(np.dot(_shape(a), _shape(b)) / len(a))


def binned(index, values, bins: int = 24) -> np.ndarray:
    """Mean value per axis bin; NaN where a bin has no pixels (profile_of aborts instead)."""
    out = np.full(bins, np.nan)
    values = np.asarray(values, dtype=np.float64)
    for b in range(bins):
        m = index == b
        if m.any():
            out[b] = values[m].mean()
    return out


def _z_finite(values):
    """Zero-mean unit-spread over the finite bins, zero elsewhere, plus the finite mask."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    v = values[finite] - values[finite].mean()
    spread = v.std()
    out = np.zeros_like(values)
    out[finite] = v / spread if spread > 1e-9 else v
    return out, finite


def _corr_finite(a, b) -> float:
    """Pearson over the bins finite in both, in -1..1 (0 if nothing overlaps)."""
    za, fa = _z_finite(a); zb, fb = _z_finite(b); m = fa & fb
    den = math.sqrt((za[m] ** 2).sum() * (zb[m] ** 2).sum())
    return float((za[m] * zb[m]).sum() / den) if den > 1e-9 else 0.0


def score_sign(sat_profile, redfraction_profile, template):
    """Match the two robust channels against the template both directions.

    The template reads rear->front. A forward (rear->front along +axis) win means the front is at
    the high-projection end: front_is_plus. The gray brightness that failed is not used.
    """
    ts, tr = template["saturation"], template["red_fraction"]
    forward = (_corr_finite(sat_profile, ts)
               + _corr_finite(redfraction_profile, tr)) / 2.0
    reverse = (_corr_finite(sat_profile, ts[::-1])
               + _corr_finite(redfraction_profile, tr[::-1])) / 2.0
    margin = forward - reverse
    return forward, reverse, margin, margin > 0


def profile_of(along: np.ndarray, tones: np.ndarray, bins: int = 24
               ) -> Optional[np.ndarray]:
    """The brightness profile along the car's axis: one mean tone per slice.

    No rotation needed - the pixels are projected onto the axis and their grey values averaged
    per slice, which is the same as measuring an aligned crop for a fraction of the cost.
    """
    if len(along) < 30:
        return None
    lo, hi = float(along.min()), float(along.max())
    if hi - lo < 12:
        return None
    index = np.clip(((along - lo) / (hi - lo) * bins).astype(int), 0, bins - 1)
    sums = np.bincount(index, weights=tones, minlength=bins)
    counts = np.bincount(index, minlength=bins)
    if not counts.all():
        return None
    return sums / counts


def ends_of(profile: np.ndarray, fraction: int = 4
            ) -> Tuple[float, float]:
    """The tone at the two ends, each relative to the profile's linear trend.

    Returns (plus_end - minus_end, magnitude) in grey levels. Positive means the end in the
    +axis direction is brighter. Removing the trend is what makes this a measurement of the
    car's pattern rather than of how the light happens to fall across it.
    """
    bins = len(profile)
    trend = np.polyval(np.polyfit(np.arange(bins, dtype=float), profile, 1),
                       np.arange(bins, dtype=float))
    detrended = profile - trend
    difference = float(detrended[-fraction:].mean() - detrended[:fraction].mean())
    return difference, abs(difference)


def band_of(profile: np.ndarray) -> Tuple[float, float]:
    """Where the darkest stretch sits along the axis, and its contrast. Reported, not used."""
    return float((int(np.argmin(profile)) + 0.5) / len(profile)), \
        float(profile.max() - profile.min())


def taper_of(points: np.ndarray, heading: float, bins: int = 12) -> float:
    """How much wider one end of a blob is than the other, along its own axis.

    Positive means the end in the +heading direction is wider. This is the cue that says
    which end of an otherwise symmetric car is the front, and it needs no rotation: the
    pixels are projected onto the axis and onto the perpendicular, and the width is the
    perpendicular extent within each slice along the axis.
    """
    if len(points) < 20:
        return 0.0
    radians = np.radians(heading)
    along = points[:, 0] * np.cos(radians) + points[:, 1] * np.sin(radians)
    across = -points[:, 0] * np.sin(radians) + points[:, 1] * np.cos(radians)
    lo, hi = float(along.min()), float(along.max())
    if hi - lo < 8:
        return 0.0
    widths = []
    for i in range(bins):
        a = lo + (hi - lo) * i / bins
        b = lo + (hi - lo) * (i + 1) / bins
        band = across[(along >= a) & (along < b)]
        widths.append(float(band.max() - band.min()) if len(band) else 0.0)
    n = max(bins // 4, 1)
    first, last = float(np.mean(widths[:n])), float(np.mean(widths[-n:]))
    return (last - first) / max(last + first, 1e-6)


def locate_all(frame: np.ndarray, min_area: int = DEFAULT_MIN_AREA,
               keep_border: bool = False,
               close_kernel: int = CLOSE_KERNEL) -> List[VehiclePose]:
    """Every red thing big enough to be the car, largest first."""
    if frame is None or frame.size == 0:
        return []
    hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = red_mask(hsv_full, close_kernel)
    # Paint without the hole fill, so the windows stay visible as gaps: red_fraction per bin says
    # where the enclosed glass is. (One inRange + open per frame.)
    raw_red = np.zeros(frame.shape[:2], np.uint8)
    for _lo, _hi in RED_HUE_RANGES:
        raw_red |= cv2.inRange(hsv_full, np.array(_lo, np.uint8), np.array(_hi, np.uint8))
    raw_red = cv2.morphologyEx(
        raw_red, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out: List[VehiclePose] = []
    for i in range(1, count):
        area = int(stats[i, 4])
        if area < min_area:
            continue
        x, y, w, h = (int(stats[i, 0]), int(stats[i, 1]),
                      int(stats[i, 2]), int(stats[i, 3]))
        if not keep_border and touches_border((x, y, w, h), frame.shape):
            continue
        ys, xs = np.where(labels == i)
        points = np.column_stack([xs, ys]).astype(np.float32)
        gray_values = gray_image[ys, xs].astype(np.float64)
        sat_values = hsv_full[ys, xs, 1].astype(np.float64)
        is_raw_red = (raw_red[ys, xs] > 0).astype(np.float64)
        centre = points.mean(axis=0)
        values, vectors = np.linalg.eigh(np.cov((points - centre).T))
        major = vectors[:, int(np.argmax(values))]
        heading_pca = float(np.degrees(np.arctan2(major[1], major[0]))) % 180.0
        elongation = float(np.sqrt(max(values) / max(min(values), 1e-6)))
        rect = cv2.minAreaRect(points)
        (_, _), (rw, rh), angle = rect
        # Long axis of the fitted rectangle. Steadier than PCA on a blob whose outline
        # flickers, because the rectangle depends on the extremes rather than every pixel.
        heading = (angle if rw >= rh else angle + 90.0) % 180.0
        taper = taper_of(points, heading)
        radians = np.radians(heading)
        along = points[:, 0] * np.cos(radians) + points[:, 1] * np.sin(radians)
        profile = profile_of(along, gray_values)
        band_pos: Optional[float] = None
        band_contrast = 0.0
        if profile is not None:
            band_pos, band_contrast = band_of(profile)
        # Bins by position along the axis (0..1), so the profiles align even when the blob is a
        # pixel longer or shorter frame to frame.
        _lo, _hi = float(along.min()), float(along.max())
        axis_index = None
        if _hi - _lo >= 12:
            axis_index = np.clip((((along - _lo) / (_hi - _lo)) * 24).astype(int), 0, 23)
        sat_profile = redfraction_profile = None
        if axis_index is not None:
            sat_profile = binned(axis_index, sat_values)
            redfraction_profile = binned(axis_index, is_raw_red)

        # Which end the front is on, as signed evidence along the axis: +1 means the +axis end.
        # The car's pattern is the first choice - the brighter end is the front - and the taper
        # of the silhouette is the fallback for a car with no usable pattern.
        evidence: Optional[float] = None
        ends_grey = 0.0
        forward_score = reverse_score = score_margin = None
        if profile is not None:
            difference, _ = ends_of(profile)
            ends_grey = difference                  # kept for the HUD, no longer the sign
        template = vehicle_template()
        if template is not None and sat_profile is not None:
            forward_score, reverse_score, score_margin, _plus = \
                score_sign(sat_profile, redfraction_profile, template)
            # Margin ~0 is the template abstaining - leave evidence None so the silhouette
            # taper below still gets used, instead of a dead 0 that disables the fallback.
            if abs(score_margin) > 1e-9:
                evidence = float(np.clip(score_margin / MARGIN_SCALE, -1.0, 1.0))
        if evidence is None and abs(taper) >= MIN_TAPER:
            signed = taper if FRONT_IS_THE_WIDER_END else -taper
            evidence = float(np.clip(signed / TAPER_SCALE, -1.0, 1.0))
        if elongation < 1.6:
            evidence = None          # a blob with no axis cannot have a front either

        front_heading = None
        front_known = False
        if evidence is not None and abs(evidence) >= KEEP_EVIDENCE:
            front_heading = heading if evidence > 0 else (heading + 180.0) % 360.0
            front_known = True
        out.append(VehiclePose(x=float(centre[0]), y=float(centre[1]), heading=heading,
                               area=area, elongation=elongation,
                               fill=area / max(w * h, 1), bbox=(x, y, w, h),
                               heading_pca=heading_pca, taper=taper,
                               band_pos=band_pos, band_contrast=band_contrast,
                               ends_grey=ends_grey,
                               forward_score=forward_score,
                               reverse_score=reverse_score,
                               score_margin=score_margin,
                               front_evidence=evidence,
                               front_heading=front_heading, front_is_known=front_known))
    out.sort(key=lambda p: -p.area)
    return out


def locate(frame: np.ndarray, **kwargs) -> Optional[VehiclePose]:
    """The car, or None when it cannot be seen. None is a real answer, not a failure."""
    found = locate_all(frame, **kwargs)
    return found[0] if found else None


def location_of(pose: VehiclePose, spots=None, roads=None,
                on_road_px: float = 35.0) -> dict:
    """Turn a pose into "which bay, or on which road", using the map rather than pixels.

    This is where the two halves of P0002 meet: the car is found by colour, and *where it is*
    is answered by the bay polygons and the road network a person drew. Neither half knows
    about the other's internals.
    """
    if spots:
        for spot in spots:
            if cv2.pointPolygonTest(np.asarray(spot.polygon, np.float32),
                                    (pose.x, pose.y), False) >= 0:
                return {"kind": "spot", "id": spot.id, "distance_to_road": None}
    if roads is not None and getattr(roads, "nodes", None):
        near = roads.closest_point(pose.x, pose.y, skip_stubs=True)
        if near is not None:
            kind = "road" if near.distance <= on_road_px else "floor"
            return {"kind": kind, "id": None, "distance_to_road": round(near.distance, 1)}
    return {"kind": "unknown", "id": None, "distance_to_road": None}
