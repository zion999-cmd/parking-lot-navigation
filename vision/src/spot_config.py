"""Parking spot configuration: polygons on the camera image.

The spots are calibrated by hand (see ``calibration.py``) and stored in
``config/parking_spots.json``. Nothing in this project tries to find parking
spots automatically.

Each spot has two polygons:

* ``polygon``       - the whole bay, used for drawing and for the reference crop.
* ``detect_polygon``- the smaller area actually used for occupancy. By default it
  is ``polygon`` shrunk towards its centroid by ``inner_scale``, which is how
  white bay lines, red/white markers, bay numbers and the kerb edge are kept out
  of the difference calculation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

DEFAULT_INNER_SCALE = 0.62
MIN_AREA_PX = 200


class SpotConfigError(ValueError):
    pass


def _as_polygon(points: Sequence[Sequence[float]], what: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise SpotConfigError(f"{what}: expected a list of [x, y] pairs, got {points!r}")
    if len(arr) < 3:
        raise SpotConfigError(f"{what}: a polygon needs at least 3 points, got {len(arr)}")
    return arr


@dataclass
class Spot:
    id: str
    polygon: np.ndarray
    inner_scale: float = DEFAULT_INNER_SCALE
    detect_polygon: Optional[np.ndarray] = None
    note: str = ""

    # derived, filled by prepare()
    roi: Tuple[int, int, int, int] = field(default=(0, 0, 0, 0), repr=False)
    mask_roi: Optional[np.ndarray] = field(default=None, repr=False)
    inner_points: Optional[np.ndarray] = field(default=None, repr=False)

    # -- geometry -----------------------------------------------------------

    def centroid(self) -> np.ndarray:
        return self.polygon.mean(axis=0)

    def area(self) -> float:
        return abs(cv2.contourArea(self.polygon.astype(np.float32)))

    def inner(self) -> np.ndarray:
        """Detection polygon: explicit override, else the shrunk bay polygon."""
        if self.detect_polygon is not None:
            return self.detect_polygon
        c = self.centroid()
        return (self.polygon - c) * self.inner_scale + c

    def crop_box(self, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        x0, y0 = self.polygon.min(axis=0).astype(int)
        x1, y1 = self.polygon.max(axis=0).astype(int)
        return (max(x0, 0), max(y0, 0),
                min(x1, shape[1]), min(y1, shape[0]))

    # -- preparation --------------------------------------------------------

    def prepare(self, shape: Tuple[int, int]) -> None:
        """Precompute the inner polygon, its bounding box and its mask."""
        inner = self.inner()
        self.inner_points = inner

        x0, y0 = np.floor(inner.min(axis=0)).astype(int)
        x1, y1 = np.ceil(inner.max(axis=0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, shape[1]), min(y1, shape[0])
        if x1 - x0 < 4 or y1 - y0 < 4:
            raise SpotConfigError(f"spot {self.id}: detection area is empty after clipping")

        self.roi = (x0, y0, x1, y1)
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(inner - [x0, y0]).astype(np.int32)], 255)
        self.mask_roi = mask

        if self.area() < MIN_AREA_PX:
            raise SpotConfigError(
                f"spot {self.id}: bay polygon is only {self.area():.0f} px^2, "
                f"that is too small to be a parking bay")

    def inner_area(self) -> int:
        return int(cv2.countNonZero(self.mask_roi)) if self.mask_roi is not None else 0

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> Dict:
        d: Dict = {
            "id": self.id,
            "polygon": [[round(float(x), 1), round(float(y), 1)] for x, y in self.polygon],
        }
        if abs(self.inner_scale - DEFAULT_INNER_SCALE) > 1e-6:
            d["inner_scale"] = round(float(self.inner_scale), 4)
        if self.detect_polygon is not None:
            d["detect_polygon"] = [[round(float(x), 1), round(float(y), 1)]
                                   for x, y in self.detect_polygon]
        if self.note:
            d["note"] = self.note
        return d

    @staticmethod
    def from_dict(d: Dict, index: int) -> "Spot":
        if "id" not in d or "polygon" not in d:
            raise SpotConfigError(f"spot #{index}: needs both 'id' and 'polygon'")
        sid = d["id"]
        detect = d.get("detect_polygon")
        return Spot(
            id=str(sid),
            polygon=_as_polygon(d["polygon"], f"spot {sid} polygon"),
            inner_scale=float(d.get("inner_scale", DEFAULT_INNER_SCALE)),
            detect_polygon=_as_polygon(detect, f"spot {sid} detect_polygon") if detect else None,
            note=str(d.get("note", "")),
        )


def validate(spots: List[Spot], shape: Tuple[int, int], allow_overlap: bool = True) -> List[str]:
    """Return a list of human-readable problems (empty list means fine)."""
    h, w = shape
    problems: List[str] = []
    for s in spots:
        for x, y in s.polygon:
            if not (0 <= x <= w and 0 <= y <= h):
                problems.append(f"spot {s.id}: point ({x:.0f}, {y:.0f}) is outside the {w}x{h} frame")
                break
    if not allow_overlap:
        for i, a in enumerate(spots):
            for b in spots[i + 1:]:
                if cv2.intersectConvexConvex(
                        a.polygon.astype(np.float32), b.polygon.astype(np.float32))[0] > 0:
                    problems.append(f"spots {a.id} and {b.id} overlap")
    return problems


def load_spots(path: str, shape: Optional[Tuple[int, int]] = None) -> List[Spot]:
    if not os.path.exists(path):
        raise SpotConfigError(
            f"{path} does not exist yet - calibrate the bays first:\n"
            f"    python src/calibration.py --source <stream-url>")
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, list):
        entries = raw
    else:
        entries = raw.get("spots", [])
    if not entries:
        raise SpotConfigError(f"{path} contains no spots")

    spots = [Spot.from_dict(e, i) for i, e in enumerate(entries)]
    if shape is not None:
        for s in spots:
            s.prepare(shape)
    return spots


def _fmt_polygon(points) -> str:
    """One line per polygon, so the file stays editable by hand."""
    return "[" + ", ".join(f"[{float(x):g}, {float(y):g}]" for x, y in points) + "]"


def save_spots(path: str, spots: List[Spot], shape: Optional[Tuple[int, int]] = None) -> None:
    """Write the config in a hand-editable form.

    json.dump(indent=2) puts every number on its own line, which makes a file full
    of polygons almost impossible to read or fix in an editor. Each spot is written
    as one line instead, and the result is still plain JSON.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    lines = [
        "{",
        '  "comment": "Fixed parking bay polygons on the camera image. '
        'Seeded by Auto Detect and confirmed by hand with src/calibration.py; '
        'edit with the same tool.",',
        f'  "image_size": {f"[{shape[0]}, {shape[1]}]" if shape else "null"},',
        '  "spots": [',
    ]
    for i, spot in enumerate(spots):
        parts = [f'"id": {json.dumps(spot.id)}',
                 f'"polygon": {_fmt_polygon(spot.polygon)}']
        if abs(spot.inner_scale - DEFAULT_INNER_SCALE) > 1e-6:
            parts.append(f'"inner_scale": {spot.inner_scale:g}')
        if spot.detect_polygon is not None:
            parts.append(f'"detect_polygon": {_fmt_polygon(spot.detect_polygon)}')
        if spot.note:
            parts.append(f'"note": {json.dumps(spot.note)}')
        comma = "," if i < len(spots) - 1 else ""
        lines.append("    { " + ", ".join(parts) + " }" + comma)
    lines += ["  ]", "}"]

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(tmp, path)
