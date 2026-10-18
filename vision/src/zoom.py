"""Digital zoom on the PC side: crop the middle, scale back up, keep the coordinates honest.

Why here and not in the camera: the sensor-side zoom needs set_res_raw(), whose window and scaling
registers are different for every sensor, and getting them wrong does not raise an error - it just
shows a strange picture. This is deterministic, testable on saved frames, and needs no firmware or
OTA. What it cannot do is add resolution: the camera has already scaled the scene to 800x600, so a
1.2x zoom makes the car bigger and slightly softer rather than sharper. That trade is worth knowing
before turning it on.

The two halves have to move together. Zooming the frame without moving the bays and roads would
silently misplace every polygon, and the bays would then all read OCCUPIED. So:

    frame   -> crop the central W/z x H/z and resize back to W x H
    points  -> x_z = (x_o - off_x) * z          (the inverse of the crop)

with off_x = W(1 - 1/z)/2. At z = 1 both are the identity, which is what the default is.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


def offsets(shape: Sequence[int], zoom: float) -> Tuple[float, float]:
    height, width = int(shape[0]), int(shape[1])
    if zoom <= 0:
        raise ValueError("zoom must be positive")
    return (width - width / zoom) / 2.0, (height - height / zoom) / 2.0


def zoom_frame(frame: np.ndarray, zoom: float) -> np.ndarray:
    """The central 1/zoom of the frame, scaled back to the frame's own size."""
    if zoom == 1.0 or frame is None or frame.size == 0:
        return frame
    height, width = frame.shape[:2]
    off_x, off_y = offsets(frame.shape, zoom)
    win_w, win_h = width / zoom, height / zoom
    x0, y0 = int(round(off_x)), int(round(off_y))
    x1, y1 = min(width, int(round(off_x + win_w))), min(height, int(round(off_y + win_h)))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return frame
    interp = cv2.INTER_AREA if zoom > 1.0 else cv2.INTER_LINEAR
    return cv2.resize(crop, (width, height), interpolation=interp)


def transform_points(points: Iterable[Sequence[float]], zoom: float,
                     shape: Sequence[int]) -> List[Tuple[float, float]]:
    """Move original-frame coordinates into the zoomed frame's coordinate system."""
    if zoom == 1.0:
        return [(float(p[0]), float(p[1])) for p in points]
    off_x, off_y = offsets(shape, zoom)
    return [((float(p[0]) - off_x) * zoom, (float(p[1]) - off_y) * zoom) for p in points]


def transform_spots(spots: List, zoom: float, shape: Sequence[int]) -> List:
    """Rewrite each bay's polygon into the zoomed frame, dropping the ones now out of view.

    Zooming in is not free: at 1.2x the outer eighth of the picture is gone, and a bay drawn there
    has no pixels left to watch. Keeping it would make prepare() raise "detection area is empty
    after clipping" - which is how this was found - and pretending it is still visible would be
    worse. So those bays are dropped and the caller reports how many.

    The Spot keeps its geometry in `polygon`; `inner`/`detect_polygon`/the prepared crop are derived
    from it, so transforming the polygon and calling prepare() again is enough.
    """
    if zoom == 1.0 or not spots:
        return spots
    height, width = int(shape[0]), int(shape[1])
    kept = []
    for spot in spots:
        poly = np.asarray(transform_points(spot.polygon, zoom, shape), np.float32)
        x0, y0 = float(poly[:, 0].min()), float(poly[:, 1].min())
        x1, y1 = float(poly[:, 0].max()), float(poly[:, 1].max())
        if min(x1, width) - max(x0, 0.0) < 4 or min(y1, height) - max(y0, 0.0) < 4:
            continue
        spot.polygon = poly
        if hasattr(spot, "prepare"):
            spot.prepare((height, width))
        kept.append(spot)
    return kept


def transform_network(network, zoom: float, shape: Sequence[int]):
    """Move every road node, then re-derive anything cached from the geometry."""
    if zoom == 1.0 or network is None or not getattr(network, "nodes", None):
        return network
    for node in network.nodes:
        node.x, node.y = transform_points([(node.x, node.y)], zoom, shape)[0]
    return network
