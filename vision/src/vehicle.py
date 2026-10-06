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


class VehicleTracker:
    """Smooth the readings a little, because a car does not turn instantly.

    Measured on the real car, 30 frames: the axis is steady to 0.0 degrees, and the taper sits
    at -0.12 with a spread of 0.05 - so it dips below the 0.04 gate in about one frame in six.
    Those dips are not the car changing shape; they are the colour mask shifting a pixel at its
    edges and the JPEG wobbling. Gating on a single frame therefore made the front appear and
    disappear several times in thirty frames while the car never moved.

    Averaging is safe here for the same reason it is dangerous in general: the quantity being
    averaged cannot change quickly. A car's shape, and so its taper, changes only when the car
    does. With alpha 0.3 the smoothed taper comes out at -0.12 +- 0.02, six times the gate away
    from crossing, while a genuine 180 degree turn still moves it within a few frames.
    """

    def __init__(self, alpha: float = 0.3, position_alpha: float = 0.5,
                 min_taper: float = MIN_TAPER, reset_px: float = 40.0):
        self.alpha = alpha
        self.position_alpha = position_alpha
        self.min_taper = min_taper
        self.reset_px = reset_px
        self._taper: Optional[float] = None
        self._ax = 0.0
        self._ay = 0.0
        self._x: Optional[float] = None
        self._y: Optional[float] = None

    def reset(self) -> None:
        """Start again: used when the car is out of sight, or has been picked up."""
        self._taper = None
        self._x = self._y = None

    def update(self, pose: Optional[VehiclePose]) -> Optional[VehiclePose]:
        """One sighting in, a smoothed sighting out. None in, None out."""
        if pose is None:
            self.reset()
            return None

        # A jump too big to be driving means the car was moved by hand, so the old shape says
        # nothing about the new one.
        if (self._x is not None
                and math.hypot(pose.x - self._x, pose.y - self._y) > self.reset_px):
            self.reset()

        a = self.alpha
        self._taper = (pose.taper if self._taper is None
                       else a * pose.taper + (1 - a) * self._taper)
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

        out = replace(pose, x=self._x, y=self._y, heading=heading, taper=self._taper)
        if pose.elongation >= 1.6 and abs(self._taper) >= self.min_taper:
            wider_is_plus = self._taper > 0
            plus_is_front = wider_is_plus == FRONT_IS_THE_WIDER_END
            out.front_heading = heading if plus_is_front else (heading + 180.0) % 360.0
            out.front_is_known = True
        else:
            out.front_heading = None
            out.front_is_known = False
        return out


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
        front_heading = None
        front_known = False
        if elongation >= 1.6 and abs(taper) >= MIN_TAPER:
            # `taper` is positive when the end in the +heading direction is wider, and
            # FRONT_IS_THE_WIDER_END records which end of this car that is.
            wider_is_plus = taper > 0
            plus_is_front = wider_is_plus == FRONT_IS_THE_WIDER_END
            front_heading = heading if plus_is_front else (heading + 180.0) % 360.0
            front_known = True
        out.append(VehiclePose(x=float(centre[0]), y=float(centre[1]), heading=heading,
                               area=area, elongation=elongation,
                               fill=area / max(w * h, 1), bbox=(x, y, w, h),
                               heading_pca=heading_pca, taper=taper,
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
