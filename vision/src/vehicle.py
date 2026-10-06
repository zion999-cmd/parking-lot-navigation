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

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Red wraps around the hue circle, so it takes two ranges. Tight on purpose: see above.
RED_HUE_RANGES: Tuple[Tuple[Tuple[int, int, int], Tuple[int, int, int]], ...] = (
    ((0, 90, 60), (3, 255, 255)),
    ((176, 90, 60), (180, 255, 255)),
)

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
    """One sighting of the car. `heading` is the long axis in degrees, 0..180, no front."""

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

    @property
    def centre(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def as_dict(self) -> dict:
        return {"id": "hero", "x": round(self.x, 1), "y": round(self.y, 1),
                "heading": round(self.heading, 1), "visible": True,
                "heading_is_axis_only": True}


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
        out.append(VehiclePose(x=float(centre[0]), y=float(centre[1]), heading=heading,
                               area=area, elongation=elongation,
                               fill=area / max(w * h, 1), bbox=(x, y, w, h),
                               heading_pca=heading_pca))
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
