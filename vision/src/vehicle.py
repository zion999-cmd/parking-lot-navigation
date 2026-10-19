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
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

import arrow            # the roof arrow; see src/arrow.py
# Car-specific but data-driven, so the hero car can be swapped without editing source. The file
# holds the HSV target colour plus the morphology sizes; built-in values below are the fallback.
VEHICLE_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "vehicle.json")
_VEHICLE_CONFIG_CACHE: Optional[dict] = "unloaded"

# Fallback if config/vehicle.json is absent: the values measured for the current red car.
_DEFAULT_VEHICLE_CONFIG = {
    "hue_ranges": [
        [[0, 90, 60], [3, 255, 255]],
        [[176, 90, 60], [180, 255, 255]],
    ],
    "open_kernel": 5,
    "close_kernel": 21,      # CLOSE_KERNEL, the value measured for the red car
    "min_area": 80,          # DEFAULT_MIN_AREA
    # Attach the car's dark part to its painted part. A car painted at both ends (the red one:
    # bonnet, windscreen, roof, rear glass, boot) is covered by the colour mask alone. A car
    # painted at one end only - a yellow nose and a black cabin - is not: the mask covers half
    # the car, which leaves no long axis, and `elongation < 1.6` then declines to name a front.
    # 0 disables it, which is what a both-ends-painted car wants.
    # Which end the taper means. A fact about a particular toy car, not about pixels, so it is
    # data. True = the wider end is the front (measured for the red car: +0.17 / -0.15 either
    # side of a 180 turn). The yellow-nosed car measures the opposite way round, and with the
    # wrong value the taper fallback reports the front backwards - confidently (evidence 0.79).
    "front_is_the_wider_end": True,
    # The paint is one end of the car and the cabin the other, so look for the dark side of the
    # paint rather than for dark pixels in general.
    "attach_paint_side": False,
    "paint_side_gap": 6,
    "paint_side_reach": 0,   # 0 = as far as the car could be: 2.2x the paint
    "paint_side_min_drop": 40,
    # And the end that is painted is the front. A fact about this car from the person who owns it,
    # not something the pixels can know - same category as front_is_the_wider_end.
    "front_is_the_painted_end": False,
    "attach_dark_margin_v": 0,
    "attach_dark_max_s": 160,
    "attach_dark_max_px": 60,
    # An outline-based attachment for a car painted at one end was tried and removed - see
    # ATTACH_MAX_GROWTH below for the measurements. Nothing here is a substitute for painting
    # both ends, which is the only thing measured to work on the real car.
}


def vehicle_config(path: str = VEHICLE_CONFIG_PATH) -> dict:
    """HSV ranges and morphology for the current hero car, loaded once.

    Missing or unreadable file: the measured red defaults. Kept as a config rather than source
    so a different coloured / sized car is a data change and a re-recorded profile, not a code
    change.
    """
    global _VEHICLE_CONFIG_CACHE
    if _VEHICLE_CONFIG_CACHE == "unloaded":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            cfg = dict(_DEFAULT_VEHICLE_CONFIG)
            for key in ("hue_ranges", "open_kernel", "close_kernel", "min_area",
                        "attach_dark_margin_v", "attach_dark_max_s",
                        "attach_dark_max_px", "front_is_the_wider_end",
                        "attach_paint_side", "paint_side_gap", "paint_side_reach",
                        "paint_side_min_drop", "front_is_the_painted_end"):
                if key in raw:
                    cfg[key] = raw[key]
            _VEHICLE_CONFIG_CACHE = cfg
        except (OSError, ValueError):
            _VEHICLE_CONFIG_CACHE = dict(_DEFAULT_VEHICLE_CONFIG)
    return _VEHICLE_CONFIG_CACHE

# Red wraps around the hue circle, so it takes two ranges. Tight on purpose: see above.
RED_HUE_RANGES: Tuple[Tuple[Tuple[int, int, int], Tuple[int, int, int]], ...] = (
    ((0, 90, 60), (3, 255, 255)),
    ((176, 90, 60), (180, 255, 255)),
)

# Below this a red blob is a sticker or a reflection, not a car.
DEFAULT_MIN_AREA = 80

# Below this elongation there is no axis to measure: an angle from a square blob is noise
# wearing a number. Used both to refuse the taper-derived front and to report the heading as
# unknown, on the same principle as front_is_known and the UNKNOWN bay state - a reader that
# cannot tell "no measurement" from "a measurement of zero" will act on the wrong one.
AXIS_MIN_ELONGATION = 1.6
# Two pieces of the same car are joined if they are this close. The body breaks into pieces
# wherever a dark band crosses it, and a centroid that jumps between two halves is not a
# position.
CLOSE_KERNEL = 21


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
    # Motion guard diagnostic: summed axis rotation over the recent window, degrees; 0 at rest.
    observed_turn: float = 0.0
    # Signed and normalised: +1 means the front is at the +axis end.
    front_evidence: Optional[float] = None
    front_heading: Optional[float] = None
    front_is_known: bool = False
    # True when the heading came from the roof arrow rather than from the shape's long
    # axis. The arrow states the direction, so this heading has a front and is not an
    # axis with two indistinguishable ends.
    heading_from_arrow: bool = False

    @property
    def centre(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def as_dict(self) -> dict:
        # heading is null when the shape has no axis, rather than an angle from a square. The
        # number is kept on the dataclass because the overlay and the diagnostic tools print it
        # next to "NO CLEAR AXIS", but the contract is what a consumer acts on, and a consumer
        # cannot tell a meaningless 87.6 from a real one.
        axis_known = self.heading_from_arrow or self.elongation >= AXIS_MIN_ELONGATION
        out = {"id": "hero", "x": round(self.x, 1), "y": round(self.y, 1),
               "heading": round(self.heading, 1) if axis_known else None,
               "visible": True,
                 "heading_is_axis_only": bool(axis_known and not self.heading_from_arrow),
               "front_is_known": bool(self.front_is_known)}
        if self.front_is_known and self.front_heading is not None:
            # The axis, plus the one fact a person had to tell us: which end is the front.
            out["front_heading"] = round(self.front_heading, 1)
        else:
            out["front_heading"] = None
        return out


def merge_near_components(mask: np.ndarray, gap_fraction: float = 0.9,
                         min_piece: int = 8) -> np.ndarray:
    """Join red pieces that belong to one car, by measured distance rather than morphology.

    A normal CLOSE is the wrong tool here: the car's front and rear red halves are separated by
    the windscreen (~24 px), tenuously joined only by sparse edge pixels, so opening disconnects
    them and closing's erosion reopens the bridge. Instead: connected components are unioned when
    their centroid gap is less than `gap_fraction` of the largest piece's bbox diagonal - the two
    halves of one car (gap ~0.5 diagonal) merge, a separate red object does not.
    """
    count, labels, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    pieces = []
    for i in range(1, count):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_piece:   # keep tiny pillar strips - the bridge
            pieces.append((int(stats[i, 0]), int(stats[i, 1]),
                           int(stats[i, 2]), int(stats[i, 3]),
                           float(cent[i][0]), float(cent[i][1])))
    if not pieces:
        return mask
    parent = list(range(len(pieces)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Two pieces belong to one car when they read as stacked, aligned body halves rather than
    # separate objects: their across-axis extents overlap, and the gap is within ~2x a typical
    # piece. The threshold scales with the *median* piece (not the largest) and pairs chain
    # transitively, so a thin pillar strip in the middle connects the two big halves - this is
    # why the OPEN must be small enough to keep that strip. A red object off to the side does
    # not overlap across-axis and stays its own component.
    median_size = float(np.median([max(w, h) for _, _, w, h, _, _ in pieces]))
    for a in range(len(pieces)):
        for b in range(a + 1, len(pieces)):
            xa, ya, wa, ha, _, _ = pieces[a]
            xb, yb, wb, hb, _, _ = pieces[b]
            gap_x = max(0, max(xa, xb) - min(xa + wa, xb + wb))
            gap_y = max(0, max(ya, yb) - min(ya + ha, yb + hb))
            edge_gap = math.hypot(gap_x, gap_y)
            if edge_gap > 2.0 * median_size:
                continue
            # Two pieces of one car are close and their union stays compact. An earlier version
            # also demanded that the facing edges overlap by half - which the real pieces broke:
            # measured live, the car's three fragments sat diagonally offset (x overlap 6 px of 23)
            # and were refused, so the car was reported as its largest fragment (box 23x26, long
            # side 26 against 63 for the whole car). The compactness guard is what keeps a distant
            # red object out, not the overlap.
            ux = max(xa + wa, xb + wb) - min(xa, xb)
            uy = max(ya + ha, yb + hb) - min(ya, yb)
            if math.hypot(ux, uy) <= 4.0 * max(max(wa, ha), max(wb, hb)):
                union(a, b)

    out = np.zeros(mask.shape, np.uint8)
    groups = {}
    for i in range(len(pieces)):
        groups.setdefault(find(i), []).append(i)
    for ids in groups.values():
        for i in ids:
            x, y, w, h, _, _ = pieces[i]
            out[y:y + h, x:x + w] |= mask[y:y + h, x:x + w]
        # The pieces of one group are spatially disjoint (the windscreen sits between them), so
        # just painting them leaves several components. Connect consecutive pieces with a short
        # thick segment: order by centroid position and join each to its nearest neighbour.
        group_pieces = [pieces[i] for i in ids]
        remaining = list(group_pieces)
        _,_,_,_,x,y = remaining.pop(0)
        chain=[(x,y)]
        while remaining:
            x,y=chain[-1]
            nearest=min(remaining, key=lambda p: math.hypot(p[4]-x,p[5]-y))
            remaining.remove(nearest)
            chain.append((nearest[4],nearest[5]))
        for (ax,ay),(bx,by) in zip(chain,chain[1:]):
            cv2.line(out,(int(ax),int(ay)),(int(bx),int(by)),255,3,cv2.LINE_AA)
    return out


def red_mask(hsv: np.ndarray, close_kernel: Optional[int] = None,
             open_kernel: Optional[int] = None) -> np.ndarray:
    """The car-colour regions as one connected shape, with enclosed holes filled.

    The car's red paint is two large halves separated by the full-width windscreen, so a colour
    mask does not contain one connected car and a morphological close cannot hold the bridge.
    merge_near_components joins the halves by their measured distance (they are ~half a body
    diagonal apart along the axis), after which flood fill closes any genuinely enclosed hole -
    a side window - but not the windscreen, which is correctly part of the silhouette.

    Colours and kernel sizes come from config/vehicle.json, so swapping the hero car is a config
    change plus a re-recorded profile, not a source edit.
    """
    cfg = vehicle_config()
    if close_kernel is None:
        close_kernel = int(cfg["close_kernel"])
    if open_kernel is None:
        open_kernel = int(cfg["open_kernel"])

    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in cfg["hue_ranges"]:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    if open_kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(open_kernel // 2))

    # What the paint alone found, kept as the fallback and as the scale for the plausibility
    # test at the end of this function.
    paint_area = float(mask.sum())
    paint_only = mask.copy()

    # The car's other half, when the paint is only one end of it: which side of the paint is
    # dark. See paint_side() for why this works where four earlier attempts did not.
    if cfg.get("attach_paint_side"):
        side = paint_side(mask, hsv[:, :, 2],
                          int(cfg.get("paint_side_gap", 6)),
                          int(cfg.get("paint_side_reach", 0)),
                          int(cfg.get("paint_side_min_drop", 40)))
        if side is not None:
            mask = attach_paint_side(mask, hsv[:, :, 2], side,
                                     int(cfg.get("paint_side_gap", 6)),
                                     int(cfg.get("paint_side_reach", 0)),
                                     int(cfg.get("paint_side_min_drop", 40)))

    # The car's dark part, if this car has one that the paint does not cover (see the config).
    mask = attach_dark(mask, hsv,
                       int(cfg.get("attach_dark_margin_v", 0)),
                       int(cfg.get("attach_dark_max_s", 160)),
                       int(cfg.get("attach_dark_max_px", 60)))

    mask = merge_near_components(mask, gap_fraction=0.9)

    holes = mask.copy()
    cv2.floodFill(holes,
                   np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8),
                   (0, 0), 255)
    full = mask | cv2.bitwise_not(holes)

    # Plausibility, checked here rather than inside attach_dark, because that is not where the
    # damage happens: measured stage by stage, the attachment added 6 px, the merge added 1,400,
    # and the hole fill added 9,000 - the fill treats background as an enclosed hole when the
    # mask cuts the frame's corner region off from (0, 0).
    #
    # The test is growth relative to the paint. Measured on the same car: a good mask is 6.6x the
    # paint (433 px of nose to 2,848 px of car); the leaking one was 22.7x (492 to 10,925). Area
    # is what separates them - the leaked blob's bounding box was only 1.25:1, so no shape test
    # would have caught it.
    #
    # Refusing gives the paint alone: no long axis, so no heading, and the pose reports the nose
    # instead of a centre that belongs to the road.
    if full.sum() > ATTACH_MAX_GROWTH * paint_area:
        return paint_only
    return full
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

# Motion guard (phase 2). A sign flip is accepted only when the axis really rotated about half
# a turn: a car cannot reverse its facing without turning. The raw axis is 0..180, so consecutive
# changes are unwrapped and accumulated over a window covering a real turn.
TURN_MIN_DEG = 120.0       # summed unwrapped axis (motion guard) needed to accept a flip
# Simple anti-jitter: the physical car cannot swap ends in a few frames, so a sign change has
# to be a clear majority of the recent frames, not just 3 strong ones.
SIGN_VOTE_WINDOW = 7        # frames remembered
SIGN_VOTE_MIN = 5           # ..of which must agree to flip (tolerates 2 jitter frames)
TURN_WINDOW_FRAMES = 120
TURN_WINDOW_SECONDS = 5.0
# A flipped first frame cannot be motion-checked (no history): commit the sign only after this
# many frames agreeing, so one bad frame cannot set it.
ESTABLISH_FRAMES = 3


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
        # (axis heading, time) of recent raw poses, for the accumulated-rotation check.
        self._heading_history: "deque[Tuple[float, float]]" = deque()
        self._fx = self._fy = 0.0
        self._contrary = 0
        # Sign candidate while ESTABLISH_FRAMES votes accumulate, plus the vote count.
        self._establish_plus: Optional[bool] = None
        self._establish_votes = 0
        # Recent frames' raw sign (True=plus) for the 180 anti-jitter majority.
        self._sign_history: "deque[bool]" = deque()

    def reset(self) -> None:
        """Start again: the car has been out of sight long enough to be a new situation."""
        self._evidence = None
        self._x = self._y = None
        self._front_is_plus = None
        self._opposite_streak = 0
        self._misses = 0
        self._heading_history.clear()
        self._fx = self._fy = 0.0
        self._contrary = 0
        self._establish_plus = None
        self._establish_votes = 0
        self._sign_history.clear()

    def update(self, pose: Optional[VehiclePose]) -> Optional[VehiclePose]:
        """One sighting in, a smoothed sighting out. None in, None out."""
        if pose is None:
            self._misses += 1
            if self._misses >= self.max_misses:
                self.reset()
            return None
        self._misses = 0

        # The arrow already states the direction, and its heading is 0..360 rather than the
        # 0..180 axis this tracker was built around. Handing it to the axis maths below
        # would be wrong twice over: the doubled-angle average cannot tell 5 degrees from
        # 185, and the front vote would be deciding something that is no longer a guess.
        # The detector is already steady frame to frame (0.993 score at the same pixel on
        # four consecutive frames), so there is nothing here for a filter to do.
        if getattr(pose, "heading_from_arrow", False):
            self._x, self._y = pose.x, pose.y
            return pose

        # Motion guard: remember the raw axis heading, trim the window by count and by age.
        self._heading_history.append((float(pose.heading), time.monotonic()))
        while (len(self._heading_history) > TURN_WINDOW_FRAMES
                or (self._heading_history
                    and time.monotonic() - self._heading_history[0][1]
                        > TURN_WINDOW_SECONDS)):
            self._heading_history.popleft()

        # A pose may carry the evidence without an explicit front - the shape the API had before
        # the direction was worked out per frame. Fill it in from the RAW heading, unwrapped by
        # the turn the motion guard has already accumulated: the axis is only defined in 0..180,
        # so after a 180-degree turn the raw heading reads the same as before and only the
        # accumulated turn says the car has been round. Without that, a real about-turn comes out
        # as "the nose did not move", which is the very fault this change removes.
        front = pose.front_heading
        if front is None and pose.front_evidence is not None:
            axis_unwrapped = (pose.heading + self._accumulated_turn()) % 360.0
            front = (axis_unwrapped if pose.front_evidence > 0
                     else (axis_unwrapped + 180.0) % 360.0)

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

        # The front is smoothed as a direction of its own: the same EMA, on the unit vector
        # rather than on the angle, because averaging directions as numbers breaks at the wrap.
        #
        # This is what the axis cannot do. `front = smoothed_axis + 180` used to reconstruct the
        # front, and it flipped by 180 whenever the smoothed axis crossed 0/180 - the same axis,
        # opposite answer. Forty live frames happened to read correctly, so it was latent.
        #
        # A window of the last N frames was the first attempt and a test rejected it: against a
        # deliberate U-turn it averaged the new direction away with the old one, the way any
        # window would. An EMA is what x, y, the evidence and the axis all use, and it follows a
        # real turn in about 1/alpha frames while still smoothing the jitter. Plain EMA was the
        # second attempt and a different test rejected that: three strong contrary frames moved
        # the reported front to nearly perpendicular. So the EMA only runs on frames that agree
        # with the committed direction, and a contrary frame counts towards the same vote the
        # sign already uses.
        if front is not None:
            if self._fx == 0.0 and self._fy == 0.0:
                self._fx, self._fy = math.cos(math.radians(front)), math.sin(math.radians(front))
            else:
                committed = math.degrees(math.atan2(self._fy, self._fx))
                if abs((front - committed + 180.0) % 360.0 - 180.0) > 90.0:
                    # A reversal candidate. A few of these are the jitter the vote already
                    # debounces, so they do not move the front; a sustained run is a real pivot
                    # and re-seeds it.
                    self._contrary += 1
                    if self._contrary >= SIGN_VOTE_MIN:
                        self._fx = math.cos(math.radians(front))
                        self._fy = math.sin(math.radians(front))
                        self._contrary = 0
                else:
                    self._contrary = 0
                    radians = math.radians(front)
                    self._fx = a * math.cos(radians) + (1 - a) * self._fx
                    self._fy = a * math.sin(radians) + (1 - a) * self._fy
        front_smoothed = (math.degrees(math.atan2(self._fy, self._fx)) % 360.0
                          if (self._fx or self._fy) else None)

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
            self._sign_history.append(wants_plus)
            while len(self._sign_history) > SIGN_VOTE_WINDOW:
                self._sign_history.popleft()
            # Establishing the front needs solid evidence; keeping it does not need to be
            # re-justified every frame. Which end is the front is a property of this car, not of
            # the last frame, so a frame with a weak cue - or the smoothed evidence dipping
            # while one contrary frame is being ignored - must not make the front disappear.
            # Only sustained contrary evidence changes it, which is what the trigger below is.
            if self._front_is_plus is None and abs(raw) >= self.keep_evidence:
                # No motion history exists at start-up, so one frame is not trusted: require
                # ESTABLISH_FRAMES agreeing, restarting if they contradict.
                if self._establish_plus == wants_plus:
                    self._establish_votes += 1
                else:
                    self._establish_plus = wants_plus
                    self._establish_votes = 1
                if self._establish_votes >= ESTABLISH_FRAMES:
                    self._front_is_plus = wants_plus
            elif wants_plus != self._front_is_plus and abs(raw) >= self.flip_evidence:
                self._opposite_streak += 1
                if self._opposite_streak >= self.flip_frames:
                    # The veto: strong contrary frames are believed only when the axis really
                    # turned ~half a circle. A sign flip with ~0 rotation is a perception error
                    # and the old facing is kept.
                    if abs(self._accumulated_turn()) >= TURN_MIN_DEG:
                        self._front_is_plus = not self._front_is_plus
                        self._opposite_streak = 0
                    # Not enough real rotation: hold. The streak stays so a genuine slow turn
                    # completes when the summed rotation crosses the gate.
            else:
                self._opposite_streak = 0

        # Simple anti-jitter, the last word on the sign. The motion guard can still pass a
        # short burst (recent driving rotation left history), so do not let the committed sign
        # disagree with a clear majority of recent frames. Three jitter frames in a 7-frame
        # window cannot outvote; a real U-turn fills the window and flips (lag <= window).
        votes = list(self._sign_history)
        if self._front_is_plus is not None and len(votes) >= SIGN_VOTE_WINDOW:
            plus_votes = sum(votes)
            majority_plus = plus_votes > len(votes) - plus_votes
            if majority_plus != self._front_is_plus:
                if max(plus_votes, len(votes) - plus_votes) >= SIGN_VOTE_MIN:
                    self._front_is_plus = majority_plus

        known = self._front_is_plus is not None and pose.elongation >= 1.6
        front = front_smoothed
        return replace(pose, x=self._x, y=self._y, heading=heading,
                       observed_turn=self._accumulated_turn(),
                       front_heading=front if known else None, front_is_known=known)

    def _accumulated_turn(self) -> float:
        """Sum the minimal signed frame-to-frame axis changes over the history.

        The axis wraps at 180, so each step is the smaller of `d` and `d - 180`, in [-90,90].
        A real turn adds same-sign steps (total ~+180); stationary noise is equally signed and
        cancels to ~0. Degrees.
        """
        history = list(self._heading_history)
        total = 0.0
        for (a, _ta), (b, _tb) in zip(history, history[1:]):
            total += (b - a + 90.0) % 180.0 - 90.0
        return total


# SCORE_USE_MIN: the profile score (forward-reverse margin) must clear this to be believed.
# Below it the match is noise (a flat/uniform blob) and the silhouette taper is the fallback.
# Real car frames measure ~0.9, flat synthetic blobs ~0.
SCORE_USE_MIN = 0.2


def _shape(values: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-spread: compare the pattern shape, not the lighting level."""
    v = np.asarray(values, dtype=np.float64)
    v = v - v.mean()
    spread = v.std()
    return v / spread if spread > 1e-9 else v


_ARROW_TEMPLATE = None


def locate_all(frame: np.ndarray, min_area: int = DEFAULT_MIN_AREA,
               keep_border: bool = False,
               close_kernel: int = CLOSE_KERNEL) -> List[VehiclePose]:
    """The car, found by the arrow taped to its roof.

    `min_area`, `keep_border` and `close_kernel` are kept so that callers and the command
    line flags still work; the arrow detector does not use them, because it is not looking
    for a colour region of any particular size.

    The yellow paint is no longer what identifies the car, but it is still the cheapest
    "the car is around here" hint, and it is what keeps a frame at 27 ms instead of the
    921 ms a whole-frame sweep costs. If the paint hint is absent the search falls back to
    the whole frame, which is slow but correct.

    See src/arrow.py for why this replaced the paint pipeline and for the measurements.
    """
    if frame is None or frame.size == 0:
        return []

    global _ARROW_TEMPLATE
    if _ARROW_TEMPLATE is None:
        try:
            _ARROW_TEMPLATE = arrow.load()
        except FileNotFoundError as exc:
            print("  arrow template missing, hero detection is off: %s" % exc, file=sys.stderr)
            return []

    hint = None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    paint = red_mask(hsv, close_kernel)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(paint, 8)
    if count > 1:
        biggest = max(range(1, count), key=lambda k: stats[k, cv2.CC_STAT_AREA])
        hint = (float(centroids[biggest][0]), float(centroids[biggest][1]))

    match = arrow.locate(frame, _ARROW_TEMPLATE, roi_hint=hint)
    if match is None:
        return []

    xs = [c[0] for c in match.corners]
    ys = [c[1] for c in match.corners]
    w = float(max(xs) - min(xs))
    h = float(max(ys) - min(ys))
    long_side = max(w, h, 1.0)
    short_side = max(min(w, h), 1.0)
    heading = float(match.heading % 360.0)
    return [VehiclePose(
        x=float(match.centre[0]),
        y=float(match.centre[1]),
        heading=heading,
        area=int(round(w * h)),
        elongation=long_side / short_side,
        fill=1.0,
        bbox=(int(min(xs)), int(min(ys)), int(w), int(h)),
        heading_pca=heading,
        # The match score is the only evidence there is now, so it goes where the old
        # pipeline kept its confidence: forward_score.
        forward_score=float(match.score),
        heading_from_arrow=True,
        front_heading=heading,
        front_is_known=True,
    )]



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
