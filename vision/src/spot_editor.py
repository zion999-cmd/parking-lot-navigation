"""Editing model for parking bay polygons - no GUI in here.

Keeping the state machine separate from the window means the interesting part
(drag a corner, add, delete, rename, generate a row, undo, merge auto-detected
candidates) can be tested without a display, which is exactly what
``tests/test_editor.py`` does. ``calibration.py`` is then just the window, the
mouse and the key map.

Design rules from the task, enforced here:

* Auto Detect is an assistant: it **adds** candidates and never removes or
  overwrites anything that is already there (requirement 11).
* A detection that fails for a bay must not fail the tool: a candidate that
  cannot be prepared is skipped and reported, not raised.
* Undo exists so one Auto Detect cannot destroy hand work (requirement 12).
* The saved JSON keeps the outer ``polygon`` + inner detection polygon scheme
  unchanged (requirement 9).
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from detector import DetectorParams, detect_bays, sort_reading_order
from spot_config import DEFAULT_INNER_SCALE, Spot, SpotConfigError, load_spots, save_spots

# How much of a candidate may overlap an existing spot before it is considered
# already present.
OVERLAP_SKIP = 0.25
# Default size for a hand-added bay, as a fraction of the frame.
NEW_SPOT_W_FRAC = 0.09
NEW_SPOT_H_FRAC = 0.22
MAX_UNDO = 25


@dataclass
class EditMessage:
    text: str = ""
    level: str = "info"          # info | warn | error


class SpotEditor:
    def __init__(self, shape: Tuple[int, int], start_id: int = 38,
                 inner_scale: float = DEFAULT_INNER_SCALE):
        self.shape = shape
        self.start_id = start_id
        self.inner_scale = inner_scale
        self.spots: List[Spot] = []
        self.selected: Optional[int] = None
        self.message = EditMessage()
        self.detection: Optional[object] = None      # last DetectorResult
        self._undo: List[Tuple[List[Spot], Optional[int]]] = []
        self._drag: Optional[Dict] = None
        self._counter = start_id
        self.dirty = False

    # -- housekeeping -------------------------------------------------------

    def _say(self, text: str, level: str = "info") -> None:
        self.message = EditMessage(text, level)

    def _push_undo(self) -> None:
        self._undo.append((copy.deepcopy(self.spots), self.selected))
        if len(self._undo) > MAX_UNDO:
            self._undo.pop(0)

    def _prepare(self, spot: Spot) -> bool:
        try:
            spot.prepare(self.shape)
            return True
        except SpotConfigError as exc:
            self._say(str(exc), "warn")
            return False

    def next_id(self) -> int:
        """Monotonically increasing, skipping ids that are already taken.

        A "lowest free number" policy would hand out 39 again after the user
        renamed 38 to 40, which is surprising when the numbers are bay labels.
        """
        used = set()
        for s in self.spots:
            try:
                used.add(int(s.id))
            except ValueError:
                pass
        while self._counter in used:
            self._counter += 1
        result = self._counter
        self._counter += 1
        return result

    def select(self, index: Optional[int]) -> None:
        self.selected = index if index is not None and 0 <= index < len(self.spots) else None

    def selected_spot(self) -> Optional[Spot]:
        if self.selected is None or not (0 <= self.selected < len(self.spots)):
            return None
        return self.spots[self.selected]

    def select_next(self, step: int = 1) -> None:
        if not self.spots:
            self.selected = None
            return
        if self.selected is None:
            self.selected = 0
        else:
            self.selected = (self.selected + step) % len(self.spots)

    # -- hit testing and dragging -------------------------------------------

    def hit_test(self, x: float, y: float, corner_tol: float = 12.0
                 ) -> Tuple[Optional[int], Optional[int]]:
        """Return (spot index, corner index) - corner None means inside the body."""
        # Corners first, and the selected spot wins ties so its handles stay usable.
        order = list(range(len(self.spots)))
        if self.selected in order:
            order.remove(self.selected)
            order.insert(0, self.selected)
        for i in order:
            poly = self.spots[i].polygon
            for c, (px, py) in enumerate(poly):
                if (px - x) ** 2 + (py - y) ** 2 <= corner_tol ** 2:
                    return i, c
        for i in order:
            if cv2.pointPolygonTest(self.spots[i].polygon.astype(np.float32),
                                    (float(x), float(y)), False) >= 0:
                return i, None
        return None, None

    def begin_drag(self, x: float, y: float) -> bool:
        index, corner = self.hit_test(x, y)
        if index is None:
            self.selected = None
            return False
        self.selected = index
        self._push_undo()
        self._drag = {
            "index": index,
            "corner": corner,
            "last": (x, y),
            "original": self.spots[index].polygon.copy(),
        }
        return True

    def update_drag(self, x: float, y: float) -> None:
        if self._drag is None:
            return
        idx = self._drag["index"]
        if not (0 <= idx < len(self.spots)):
            return
        spot = self.spots[idx]
        corner = self._drag["corner"]
        if corner is None:
            dx = x - self._drag["last"][0]
            dy = y - self._drag["last"][1]
            spot.polygon = spot.polygon + np.array([dx, dy], dtype=np.float64)
            self._drag["last"] = (x, y)
        else:
            spot.polygon[corner] = [x, y]
        self._prepare(spot)
        self.dirty = True

    def end_drag(self) -> None:
        if self._drag is None:
            return
        idx = self._drag["index"]
        if 0 <= idx < len(self.spots):
            spot = self.spots[idx]
            if spot.area() < 100:            # a stray click should not leave a dot
                self._say(f"spot {spot.id} collapsed, removed", "warn")
                self.spots.pop(idx)
                self.selected = None
        # begin_drag() already pushed the undo state for this drag; pushing again
        # here would make one 'z' look like it did nothing.
        self._drag = None
        self.dirty = True

    def nudge(self, dx: float, dy: float) -> None:
        spot = self.selected_spot()
        if spot is None:
            return
        self._push_undo()
        spot.polygon = spot.polygon + np.array([dx, dy], dtype=np.float64)
        self._prepare(spot)
        self.dirty = True

    # -- structural edits ---------------------------------------------------

    def add_at(self, x: Optional[float] = None, y: Optional[float] = None) -> Optional[Spot]:
        h, w = self.shape
        bw = max(int(w * NEW_SPOT_W_FRAC), 24)
        bh = max(int(h * NEW_SPOT_H_FRAC), 40)
        if x is None:
            x = w / 2
        if y is None:
            y = h / 2
        x0, y0 = x - bw / 2, y - bh / 2
        poly = np.array([[x0, y0], [x0 + bw, y0], [x0 + bw, y0 + bh], [x0, y0 + bh]],
                        dtype=np.float64)
        # keep it inside the frame
        poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
        spot = Spot(id=str(self.next_id()), polygon=poly, inner_scale=self.inner_scale)
        if not self._prepare(spot):
            return None
        self._push_undo()
        self.spots.append(spot)
        self.selected = len(self.spots) - 1
        self.dirty = True
        self._say(f"added spot {spot.id} - drag its corners into place")
        return spot

    def delete_selected(self) -> bool:
        if self.selected is None or not self.spots:
            self._say("nothing selected", "warn")
            return False
        self._push_undo()
        spot = self.spots.pop(self.selected)
        self.selected = min(self.selected, len(self.spots) - 1) if self.spots else None
        self.dirty = True
        self._say(f"deleted spot {spot.id}")
        return True

    def rename_selected(self, new_id: str) -> bool:
        spot = self.selected_spot()
        if spot is None:
            self._say("nothing selected", "warn")
            return False
        new_id = new_id.strip()
        if not new_id:
            return False
        if any(s is not spot and s.id == new_id for s in self.spots):
            self._say(f"id {new_id} is already used", "warn")
            return False
        self._push_undo()
        old, spot.id = spot.id, new_id
        self.dirty = True
        self._say(f"renamed {old} -> {new_id}")
        return True

    def set_inner_scale(self, value: float) -> bool:
        spot = self.selected_spot()
        if spot is None:
            self._say("nothing selected", "warn")
            return False
        if not 0.2 <= value <= 0.95:
            self._say("inner scale must be within 0.2-0.95", "warn")
            return False
        self._push_undo()
        spot.inner_scale = value
        spot.detect_polygon = None
        self._prepare(spot)
        self.dirty = True
        self._say(f"spot {spot.id} inner scale {value:.2f} "
                  f"(detect area {spot.inner_area()} px^2)")
        return True

    def generate_row(self, direction: int = +1, count: int = 1) -> int:
        """Replicate the selected bay along its own top edge.

        The top edge vector gives both the pitch and the direction, so this works
        for a slanted row too. This is the "calibrate one bay, get the row"
        helper, which on a regular car park is more reliable than asking the
        detector to find every bay.
        """
        spot = self.selected_spot()
        if spot is None:
            self._say("select a spot first, then generate the row from it", "warn")
            return 0
        vec = spot.polygon[1] - spot.polygon[0]        # along the row
        if np.linalg.norm(vec) < 4:
            self._say("selected spot is too small to replicate", "warn")
            return 0

        self._push_undo()
        added = 0
        base = spot
        for k in range(1, count + 1):
            offset = vec * (direction * k)
            poly = base.polygon + offset
            if poly[:, 0].min() < 0 or poly[:, 0].max() >= self.shape[1]:
                self._say("stopped: the row would leave the frame", "warn")
                break
            cand = Spot(id=str(self.next_id()), polygon=poly, inner_scale=base.inner_scale)
            if not self._prepare(cand):
                continue
            if self._overlaps(cand) is not None:
                self._say("stopped: next bay would overlap an existing one", "warn")
                break
            self.spots.append(cand)
            self.selected = len(self.spots) - 1
            added += 1
        self.dirty = True
        if added:
            self._say(f"generated {added} more bay(s) from spot {base.id}")
        return added

    def clear(self) -> None:
        if not self.spots:
            return
        self._push_undo()
        n = len(self.spots)
        self.spots = []
        self.selected = None
        self.dirty = True
        self._say(f"cleared {n} spot(s) - undo with z")

    def undo(self) -> bool:
        if not self._undo:
            self._say("nothing to undo", "warn")
            return False
        self.spots, self.selected = self._undo.pop()
        self.dirty = True
        self._say(f"undo ({len(self._undo)} step(s) left)")
        return True

    # -- auto detect --------------------------------------------------------

    def _overlaps(self, cand: Spot) -> Optional[Spot]:
        for s in self.spots:
            inter, _ = cv2.intersectConvexConvex(
                cand.polygon.astype(np.float32), s.polygon.astype(np.float32))
            if inter > 0 and inter / max(cand.area(), 1.0) > OVERLAP_SKIP:
                return s
        return None

    def auto_detect(self, frame: np.ndarray, params: Optional[DetectorParams] = None) -> Dict:
        """Add detected candidates. Existing spots are never modified or removed."""
        self._push_undo()
        result = detect_bays(frame, params)
        self.detection = result

        added, skipped_overlap, failed = 0, 0, 0
        for poly in sort_reading_order(result.polygons):
            cand = Spot(id=str(self.next_id()),
                        polygon=np.asarray(poly, dtype=np.float64),
                        inner_scale=self.inner_scale)
            try:
                cand.prepare(self.shape)
            except SpotConfigError:
                failed += 1
                continue
            if self._overlaps(cand) is not None:
                skipped_overlap += 1
                continue
            self.spots.append(cand)
            added += 1

        if added:
            self.selected = len(self.spots) - 1
            self.dirty = True

        rows = result.diagnostics.get("rows", [])
        if result.count == 0:
            level, text = "warn", ("auto detect found nothing - add bays by hand "
                                   "(n) or calibrate one and press g to generate a row")
        else:
            level = "info"
            text = (f"auto detect: +{added} added, {skipped_overlap} already present, "
                    f"{failed} unusable, {len(rows)} row(s)")
            spacings = [r.get("grid_spacing") for r in rows if r.get("grid_spacing")]
            if spacings:
                text += f", pitch {min(spacings):.0f}-{max(spacings):.0f} px"
        self._say(text, level)
        return {"added": added, "skipped_overlap": skipped_overlap,
                "failed": failed, "found": result.count, "diagnostics": result.diagnostics}

    # -- persistence --------------------------------------------------------

    def load(self, path: str) -> bool:
        try:
            spots = load_spots(path, self.shape)
        except SpotConfigError as exc:
            self._say(str(exc), "warn")
            return False
        self._push_undo()
        self.spots = spots
        self.selected = 0 if spots else None
        self.dirty = False
        self._say(f"loaded {len(spots)} spot(s) from {path}")
        return True

    def save(self, path: str) -> bool:
        if not self.spots:
            self._say("nothing to save yet", "warn")
            return False
        save_spots(path, self.spots, self.shape)
        self.dirty = False
        self._say(f"saved {len(self.spots)} spot(s) to {os.path.basename(path)}")
        return True
