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

import math
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

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
BAND_MIN_CONTRAST = 15.0     # grey levels; below this there is no band to speak of
BAND_SCALE = 0.5             # a band this far from the middle is fully convincing
TAPER_SCALE = 0.20           # likewise for the taper
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


def band_of(along: np.ndarray, tones: np.ndarray, bins: int = 12
            ) -> Tuple[Optional[float], float]:
    """Where the dark stretch sits along the axis, and how dark it is against the rest.

    No rotation is needed: the pixels are projected onto the axis, and the grey values are
    averaged per slice. Same quantity as measuring the brightness profile of an aligned crop,
    for a fraction of the cost.
    """
    if len(along) < 20:
        return None, 0.0
    lo, hi = float(along.min()), float(along.max())
    if hi - lo < 8:
        return None, 0.0
    index = np.clip(((along - lo) / (hi - lo) * bins).astype(int), 0, bins - 1)
    sums = np.bincount(index, weights=tones, minlength=bins)
    counts = np.bincount(index, minlength=bins)
    if not counts.all():
        return None, 0.0
    profile = sums / counts
    contrast = float(profile.max() - profile.min())
    return float((int(np.argmin(profile)) + 0.5) / bins), contrast


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
    mask = red_mask(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), close_kernel)
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
        band_pos, band_contrast = band_of(along, gray_values)

        # Which end the front is on, as a signed evidence along the axis.
        evidence: Optional[float] = None
        if band_pos is not None and band_contrast >= BAND_MIN_CONTRAST:
            evidence = float(np.clip((band_pos - 0.5) * 2.0 / BAND_SCALE, -1.0, 1.0))
        elif abs(taper) >= MIN_TAPER:
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
