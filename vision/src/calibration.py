"""Parking bay calibration assistant (P0002.1).

Auto Detect proposes, a human decides. The window shows the live camera image,
OpenCV proposes bay candidates from the regular divider lines, and every corner
is then draggable.

    python src/calibration.py --source http://192.168.3.118/stream
    python src/calibration.py --frame data/debug/frame.jpg      # calibrate offline

Window
------
    [Auto Detect] [Gen Row] [Save] [Undo] [Reset]     <- clickable, or use keys

     left click on a bay      select it
     left click on a corner   drag that corner
     left drag inside a bay   move the whole bay
     right click on a corner  snap it to the nearest strong edge

Keys
----
Two layers over one map document, switched with 't' or by clicking the tabs along the
top: **[ Parking Spots ] [ Road Network ]**. Both are always drawn - the inactive one
faintly - because the bay outlines are what a bay connection point has to land on.

Parking Spots layer (what the camera answers: is this bay free?)
    A  align all bays to the current frame (camera drift; adds/deletes nothing)
    a  Auto Detect (only ADDS candidates; existing work is never touched)
    g  generate the next bay in the row from the selected one (right)
    G  ... same, to the left
    n  add a new empty bay at the mouse position
    d  delete the selected bay
    r  rename the selected bay (type the id in the terminal)
    i  set the inner detection scale of the selected bay (terminal)
    j  send the selected bay's entrance back to automatic (see below)
    s  save the map      o  load it
    z  undo              c  reset (clear all bays)
    l  toggle the detected lines overlay      v  toggle inner detection polygons
    m  cycle what 2/4/6/8 move: this bay -> its row -> every bay
    Tab cycle selection      2/4/6/8 nudge by 5 px
    arrow keys nudge by 0.5 px - the whole bay, or the corner you last clicked, which is the only
    way to place a point that is one pixel out, and they say the new coordinate in the window
    because half a pixel cannot be seen

Where a car drives into a bay is worked out from the roads - the side facing the nearest
one - which is right almost everywhere. At the end of a row or on a corner two roads are
about equally close and the choice is a judgement, so those can be corrected: in the Road
layer, drag a bay's orange entrance dot onto the side cars actually come in from. 'j' here
sends one back to being worked out. The correction is saved with the bay, and the
connection point and the graph follow from it with no further steps.

Road Network layer (what the navigator needs: how does a car get there?)
    click            place a node; it connects to the one before it, so a road is
                     drawn by clicking along the path
    drag a node      move it
    right click      on a node: join the chain head to it (that is how a branch starts)
                     on empty space: start a new road
    k                cycle the node kind: entrance -> junction -> waypoint -> spot
    b                bind the selected node to the next bay without a connection point
    n                break the chain, so the next click starts a new road
    d                delete the selected node and its edges
    p                report what is wrong (half-drawn is normal, unreachable bays are not)
    z undo   c clear   s save the map   Tab next node   2/4/6/8 move the node

Nothing about the road is auto-detected: its visual edges do not matter to navigation.
The centre line a car should follow is a human decision, and that is the ground truth.

    q / ESC  save and quit

The status line reports what Auto Detect did: how many candidates were added,
how many were already present, row count and the measured bay pitch. A detection
that finds nothing is not a failure - add one bay by hand and press 'g' to
generate the rest of the row.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Callable, List, Optional, Tuple
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera import CameraError, open_camera                    # noqa: E402
from alignment import Aligner, transform_points                 # noqa: E402
from detector import (DetectorParams, detect_bays, draw_debug,  # noqa: E402
                      render_profile_ascii, roi_offset_from_bay_lines,
                      sort_reading_order)
from road_editor import RoadEditor                             # noqa: E402
from road_network import (KIND_ENTRANCE, KIND_JUNCTION,        # noqa: E402
                          KIND_SPOT, KIND_WAYPOINT, nearest_edge)
from spot_config import DEFAULT_INNER_SCALE                    # noqa: E402
from spot_editor import SpotEditor                             # noqa: E402

WINDOW = "Parking bay calibration"

COL_BAY = (0, 255, 255)
COL_SEL = (255, 0, 255)
COL_INNER = (0, 190, 0)
COL_OK = (120, 255, 120)
COL_WARN = (0, 210, 255)
COL_ERR = (80, 80, 255)
COL_TEXT = (255, 255, 255)
COL_BTN = (70, 70, 70)
COL_BTN_HOT = (120, 120, 120)

# Road layer. The two layers are drawn in different colours on purpose: a bay outline and
# a road are different kinds of thing and should never be mistaken for each other.
COL_ROAD = (60, 200, 255)
COL_ROAD_FAINT = (95, 95, 95)
COL_NODE = (205, 205, 205)
COL_NODE_ENTRANCE = (0, 255, 0)
COL_NODE_JUNCTION = (255, 255, 0)
COL_NODE_SPOT = (0, 140, 255)
COL_BAY_FAINT = (105, 105, 105)
COL_CHAIN = (255, 0, 255)
# Derived, not drawn: bay entrances and the short lines joining them to a road.
COL_DERIVED = (0, 165, 255)


def _dashed(canvas: np.ndarray, p0, p1, colour, thickness: int = 1,
            dash: int = 6) -> None:
    """A dashed line, so derived things cannot be mistaken for drawn ones."""
    x0, y0 = p0
    x1, y1 = p1
    length = float(np.hypot(x1 - x0, y1 - y0))
    if length < 1e-6:
        return
    steps = int(length // dash) + 1
    for i in range(0, steps, 2):
        t0 = min(i * dash / length, 1.0)
        t1 = min((i + 1) * dash / length, 1.0)
        cv2.line(canvas,
                 (int(x0 + (x1 - x0) * t0), int(y0 + (y1 - y0) * t0)),
                 (int(x0 + (x1 - x0) * t1), int(y0 + (y1 - y0) * t1)),
                 colour, thickness)

BAR_H = 34
NUDGE = 5


class CalibrationApp:
    """Window + input handling on top of SpotEditor (which holds the state)."""

    def __init__(self, editor: SpotEditor, out_path: str,
                 template: Optional[np.ndarray] = None,
                 align_motion: str = "translation"):
        self.ed = editor
        # Which key renames a bay. Standalone this app owns its window, so it is 'r'. main.py
        # binds 'r' to capturing the reference frame, and a window that tells you to press a key
        # that does something else is worse than one that says nothing - so it sets this to 'R'
        # and the hint below tells the truth about the window you are actually looking at.
        self.rename_hint = "r"
        self.out_path = out_path
        self.template = template          # the frame the bays were drawn on
        self.align_motion = align_motion
        self.last_align = None
        self.show_inner = True
        self.show_overlay = False
        self.cursor: Tuple[int, int] = (0, 0)
        # What 2/4/6/8 move: the selected bay, its whole row, or everything.
        self.nudge_scope = "bay"
        # Two layers over one document: the camera's questions and the navigator's.
        self.layer = "spots"
        # Set while a bay's derived entrance is being dragged, so that drag is not taken for
        # moving a road point or for drawing a new one.
        self._access_drag: Optional[str] = None
        self.road = RoadEditor(editor.shape)
        # The roads live in the same document as the bays, so they have to be opened with
        # it. Without this the road layer started empty every time and saving would have
        # wiped roads that were already drawn.
        self.road.load(out_path)
        if self.road.net.stale_connections:
            print(f"  map: restored {self.road.net.stale_connections} stored bay connection "
                  f"point(s) to plain road points - bay entrances are derived now, not "
                  f"saved")
            self.road.dirty = True
        self._buttons: List[Tuple[int, int, int, int, str, str]] = []
        self._pending_prompt: Optional[str] = None
        # True while a run of arrow-key nudges is in progress, so the run is one undo step.
        self._fine_run = False
        # PARKING_LOG_KEYS=1 prints every key code the window receives. The arrow keys are the
        # only keys whose numbers differ between backends, and this is how they get measured on a
        # machine nobody can press keys on: run with the variable set, press the four arrows, and
        # the four numbers are in the terminal.
        self.log_keys = bool(os.environ.get("PARKING_LOG_KEYS"))

    # -- drawing ------------------------------------------------------------

    def render(self, frame: np.ndarray) -> np.ndarray:
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        if self.show_overlay and self.ed.detection is not None:
            canvas = draw_debug(canvas, self.ed.detection)

        # Both layers are always drawn: the inactive one faintly, as context. In road mode
        # the bays are what the connection points have to land on, so they stay visible.
        if self.layer == "roads":
            self._draw_spots(canvas, faint=True)
            self._draw_roads(canvas, faint=False)
        else:
            self._draw_roads(canvas, faint=True)
            self._draw_spots(canvas, faint=False)

        # Layer tabs along the top: choosing the layer is the first decision.
        self._buttons = []
        x = 8
        for label, key, active in (("Parking Spots", "@spots", self.layer == "spots"),
                                   ("Roads", "@roads", self.layer == "roads")):
            bw = 14 + len(label) * 9
            self._buttons.append((x, 6, bw, 24, label, key, active))
            x += bw + 6

        header = self._header()
        cv2.putText(canvas, header, (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(canvas, header, (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_TEXT, 1)

        actions = (("Auto Detect", "a"), ("Gen Row", "g"), ("Save", "s"), ("Undo", "z"),
                   ("Reset", "c")) if self.layer == "spots" else \
                  (("New Road", "n"), ("Entrance", "k"), ("Save", "s"),
                   ("Undo", "z"), ("Check", "p"))
        x = 8
        for label, key in actions:
            bw = 12 + len(label) * 9
            self._buttons.append((x, h - BAR_H - 8, bw, BAR_H - 8, label, key, False))
            x += bw + 8

        msg = self.ed.message if self.layer == "spots" else self.road.message
        colour = {"info": COL_OK, "warn": COL_WARN, "error": COL_ERR}.get(msg.level, COL_TEXT)
        if msg.text:
            cv2.putText(canvas, msg.text[:118], (8, h - BAR_H - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 4)
            cv2.putText(canvas, msg.text[:118], (8, h - BAR_H - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 1)

        if self.layer == "spots":
            hint = ("drag | A align | a auto | g row | n add | d del | "
                    f"{self.rename_hint} rename | i inner | "
                    "j ent | s save | z undo | c reset | 2468 (m) | arrows 0.5px | Tab | t roads | q")
        else:
            hint = ("click = draw road | drag orange dot = fix a bay's way in | "
                    "right-click = join roads | arrows 0.5px | k = way in | d del | z undo | "
                    "s save | q")
        cv2.putText(canvas, hint[:135], (8, h - BAR_H - 28),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 3)
        cv2.putText(canvas, hint[:135], (8, h - BAR_H - 28),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 200, 200), 1)

        for (bx, by, bw, bh, label, _key, active) in self._buttons:
            hover = bx <= self.cursor[0] <= bx + bw and by <= self.cursor[1] <= by + bh
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh),
                          COL_BTN_HOT if (hover or active) else COL_BTN, -1)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (200, 200, 200), 1)
            cv2.putText(canvas, label, (bx + 6, by + bh - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_TEXT, 1)
        return canvas

    # -- layer drawing ------------------------------------------------------

    def _draw_spots(self, canvas: np.ndarray, faint: bool) -> None:
        for i, spot in enumerate(self.ed.spots):
            sel = (i == self.ed.selected) and not faint
            poly = np.round(spot.polygon).astype(np.int32)
            cv2.polylines(canvas, [poly], True,
                          COL_BAY_FAINT if faint else (COL_SEL if sel else COL_BAY),
                          1 if faint else (3 if sel else 2))
            if faint:
                continue
            if self.show_inner and spot.inner_points is not None:
                cv2.polylines(canvas, [np.round(spot.inner_points).astype(np.int32)],
                              True, COL_INNER, 1)
            cx, cy = spot.polygon.mean(axis=0).astype(int)
            label = f"Spot {spot.id}"
            cv2.putText(canvas, label, (cx - 34, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 4)
            cv2.putText(canvas, label, (cx - 34, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        COL_SEL if sel else COL_BAY, 1)
            if sel:
                for px, py in poly:
                    cv2.circle(canvas, (int(px), int(py)), 6, COL_SEL, -1)
                    cv2.circle(canvas, (int(px), int(py)), 6, (0, 0, 0), 1)

    def _node_colour(self, kind: str) -> Tuple[int, int, int]:
        return {KIND_ENTRANCE: COL_NODE_ENTRANCE, KIND_JUNCTION: COL_NODE_JUNCTION,
                KIND_SPOT: COL_NODE_SPOT}.get(kind, COL_NODE)

    def _draw_roads(self, canvas: np.ndarray, faint: bool) -> None:
        """Draw what was drawn, plus what was worked out from it.

        The two are styled differently on purpose. Solid lines and big circles are the roads
        a person drew; thin lines and small dots are the bay entrances derived from them, so
        it is obvious at a glance which parts are theirs to edit and which are automatic.
        """
        roads = self.road.net
        # While a corner is being dragged this layer is only context underneath the bays, and
        # working out the derived entrances costs 20.3 ms on the real map - for 48 bays, every
        # frame. That was the difference between a drag that follows the pointer and one that
        # lurches. The roads as drawn are enough to draw context; the entrances come back the
        # moment the drag ends (and are always there in the roads layer, where they are edited).
        graph = roads if (faint and self.ed.dragging()) else self.road.derived(self.ed.spots)
        road_colour = COL_ROAD_FAINT if faint else COL_ROAD
        # Indexed once. The lookups below used to be RoadNetwork.node(), a linear scan per call,
        # and the loop asked for the same two nodes twice per edge.
        by_id = {n.id: n for n in graph.nodes}
        drawn_ids = {n.id for n in roads.nodes}

        def is_stub(edge) -> bool:
            a, b = by_id.get(edge.a), by_id.get(edge.b)
            return bool(a and b and (a.kind == KIND_SPOT or b.kind == KIND_SPOT))

        for edge in graph.edges:
            a, b = by_id.get(edge.a), by_id.get(edge.b)
            if a is None or b is None:
                continue
            pa, pb = (int(a.x), int(a.y)), (int(b.x), int(b.y))
            if is_stub(edge):
                _dashed(canvas, pa, pb, COL_DERIVED if not faint else COL_ROAD_FAINT,
                        1 if faint else 1)
                continue
            if edge.oneway and not faint:
                mid = ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)
                cv2.arrowedLine(canvas, pa, (int(mid[0]), int(mid[1])), road_colour, 2,
                                tipLength=0.2)
                cv2.line(canvas, (int(mid[0]), int(mid[1])), pb, road_colour, 1)
            else:
                cv2.line(canvas, pa, pb, road_colour, 1 if faint else 3)

        for node in graph.nodes:
            drawn = node.id in drawn_ids            # the person's own point
            centre = (int(node.x), int(node.y))
            if node.kind == KIND_SPOT:
                if faint:
                    continue
                cv2.circle(canvas, centre, 4, COL_DERIVED, -1)
                cv2.circle(canvas, centre, 4, (0, 0, 0), 1)
                if any(s.id == node.spot and s.access_side is not None
                       for s in self.ed.spots):
                    # A ring marks an entrance whose side was chosen by hand, so it is clear
                    # which of these were worked out and which were decided.
                    cv2.circle(canvas, centre, 7, COL_CHAIN, 1)
                continue
            if faint:
                cv2.circle(canvas, centre, 3, COL_ROAD_FAINT, -1)
                continue
            sel = drawn and self.road.selected is not None and \
                roads.nodes[self.road.selected].id == node.id
            colour = self._node_colour(node.kind)
            r = 8 if sel else 6
            cv2.circle(canvas, centre, r, colour, -1)
            cv2.circle(canvas, centre, r, (0, 0, 0), 1)
            if self.road.chain is not None and node.id == self.road.chain:
                cv2.circle(canvas, centre, r + 4, COL_CHAIN, 1)
            if node.kind == KIND_ENTRANCE:
                cv2.putText(canvas, "IN", (centre[0] + r + 3, centre[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
                cv2.putText(canvas, "IN", (centre[0] + r + 3, centre[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

    def _header(self) -> str:
        if self.layer == "roads":
            return self.road.stats(self.ed.spots)
        selected = (self.ed.spots[self.ed.selected].id
                    if self.ed.selected is not None else '-')
        return (f"PARKING SPOTS   spots {len(self.ed.spots)}   selected {selected}   "
                f"inner {self.ed.inner_scale:.2f}   "
                f"{'*unsaved*' if self.ed.dirty else 'saved'}")

    # -- input --------------------------------------------------------------

    def _button_at(self, x: int, y: int) -> Optional[str]:
        for (bx, by, bw, bh, _label, key, _active) in self._buttons:
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return key
        return None

    def on_mouse(self, event, x, y, flags, param) -> None:
        self.cursor = (x, y)
        # A click ends a run of arrow-key nudges: the next one starts a new undo step.
        self._fine_run = False

        if self.layer == "roads":
            self._mouse_roads(event, x, y, flags)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            key = self._button_at(x, y)
            if key is not None:
                self.act(key)
                return
            self.ed.begin_drag(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            self.ed.update_drag(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.ed.end_drag()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.snap_corner(x, y):
                pass

    def _mouse_roads(self, event, x, y, flags) -> None:
        """Draw a road with clicks; grab a bay's entrance to move it."""
        road = self.road
        if event == cv2.EVENT_LBUTTONDOWN:
            key = self._button_at(x, y)
            if key is not None:
                self.act(key)
                return
            grabbed = self._access_under(x, y)
            if grabbed is not None:
                # Grabbing a bay's entrance means "that is the wrong side": whichever side it
                # is dropped on becomes the side cars drive in from.
                self._access_drag = grabbed
                road._say(f"drop it on the side cars drive in from (bay {grabbed})")
                return
            if not road.begin_drag(x, y):
                road.add_at(x, y)          # empty space: draw the road along here
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            if self._access_drag is None:
                road.update_drag(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self._access_drag is not None:
                self._drop_access_side(self._access_drag, x, y)
                self._access_drag = None
            else:
                road.end_drag()
        elif event == cv2.EVENT_RBUTTONDOWN:
            idx = road.hit_test(x, y)
            if idx is not None:
                road.connect_to(idx)       # join the road being drawn to this one
            else:
                road.break_chain()         # right click on nothing: start another road

    def _access_under(self, x: float, y: float, tol: float = 14.0) -> Optional[str]:
        """The bay whose derived entrance sits under the pointer, if any."""
        derived = self.road.derived(self.ed.spots)
        best: Optional[str] = None
        best_d = tol
        for node in derived.nodes:
            if node.kind != KIND_SPOT or not node.spot:
                continue
            d = ((node.x - x) ** 2 + (node.y - y) ** 2) ** 0.5
            if d <= best_d:
                best, best_d = node.spot, d
        return best

    def _drop_access_side(self, bay_id: str, x: float, y: float) -> None:
        """Record which side of that bay the entrance was dropped nearest to."""
        for i, spot in enumerate(self.ed.spots):
            if spot.id == bay_id:
                self.ed.set_access_side(i, nearest_edge(spot.polygon, x, y))
                return

    def bay_pitch(self) -> Optional[float]:
        """Median bay width - the scale for the "is this shift a whole bay" check."""
        widths = []
        for sp in self.ed.spots:
            pts = np.asarray(sp.polygon, dtype=np.float64)
            widths.append(min(np.linalg.norm(pts[1] - pts[0]),
                              np.linalg.norm(pts[3] - pts[0])))
        return float(np.median(widths)) if widths else None

    def align_to_template(self, save: bool = False) -> bool:
        """Move every polygon onto the current frame, cancelling camera drift.

        This is the "align, do not add or delete bays" action: the polygons keep
        their ids and their inner scale, only their coordinates move by the
        estimated drift. On success the template becomes the current frame, because
        the polygons now describe *this* image.
        """
        if self.template is None:
            self.ed._say("no template frame to align against "
                         "(pass --frame, or capture a reference first)", "warn")
            return False
        frame = getattr(self, "_frame", None)
        if frame is None:
            self.ed._say("no live frame yet", "warn")
            return False
        if frame.shape[:2] != self.template.shape[:2]:
            self.ed._say(f"frame size {frame.shape[:2]} does not match the template "
                         f"{self.template.shape[:2]}", "warn")
            return False

        aligner = Aligner(self.template, motion=self.align_motion,
                          bay_pitch=self.bay_pitch())
        res = aligner.estimate(frame)
        self.last_align = res
        print(f"  alignment vs template: {res.describe()}")
        if not res.ok:
            self.ed._say(f"alignment refused: {res.reason}", "warn")
            print("  nothing moved")
            return False

        before_off, n_edges = roi_offset_from_bay_lines(frame, self.ed.spots)

        inv = aligner.matrix_to_current(res.matrix)
        self.ed._push_undo()
        for spot in self.ed.spots:
            spot.polygon = transform_points(spot.polygon, inv)
            spot.detect_polygon = None
            self.ed._prepare(spot)
        self.ed.dirty = True
        self.template = frame.copy()

        after_off, _ = roi_offset_from_bay_lines(frame, self.ed.spots)
        text = (f"aligned {len(self.ed.spots)} bay(s) by "
                f"({res.dx:+.1f}, {res.dy:+.1f}) px, corr {res.correlation:.2f}")
        if n_edges:
            print(f"  bays vs detected lines: {before_off:.1f} px off -> "
                  f"{after_off:.1f} px off ({n_edges} edges measured)")
            text += f", bays now {after_off:.1f} px off the bay lines"
        for warning in res.warnings:
            text += f" | {warning}"
        self.ed._say(text + (" - saved" if save else " - press s to save"),
                     "warn" if res.warnings else "info")
        if save:
            self.ed.save(self.out_path)
        return True

    def snap_corner(self, x: float, y: float, search: int = 9) -> bool:
        """Pull the nearest corner of the selected bay onto the strongest edge."""
        spot = self.ed.selected_spot()
        if spot is None or self.ed.detection is None or self.ed.detection.edges is None:
            return False
        edges = self.ed.detection.edges
        h, w = edges.shape[:2]
        best = None
        for c, (px, py) in enumerate(spot.polygon):
            d = (px - x) ** 2 + (py - y) ** 2
            if best is None or d < best[0]:
                best = (d, c)
        if best is None or best[0] > 20 ** 2:
            return False
        c = best[1]
        px, py = spot.polygon[c]
        x0, x1 = int(max(px - search, 0)), int(min(px + search + 1, w))
        y0, y1 = int(max(py - search, 0)), int(min(py + search + 1, h))
        sub = edges[y0:y1, x0:x1]
        if sub.size == 0 or sub.max() == 0:
            self.ed._say("no strong edge nearby to snap to", "warn")
            return False
        dy, dx = np.unravel_index(int(np.argmax(sub)), sub.shape)
        self.ed._push_undo()
        spot.polygon[c] = [x0 + dx, y0 + dy]
        self.ed._prepare(spot)
        self.ed.dirty = True
        self.ed._say(f"snapped corner {c + 1} of spot {spot.id} to the nearest edge")
        return True

    FINE_PX = 0.5

    # Arrow keys are not characters. Which numbers a backend reports them as differs, so every
    # family OpenCV is known to produce is accepted.
    #
    # The three low codes are the interesting case and are marked as such: printable characters
    # arrive as their ASCII value ('2' is 50), so 0x00..0x03 cannot be a normal key and are free
    # to use. The assignment below is a convention, not something measured on this machine - so
    # when one arrives the window says which number it saw and what it decided it meant. A wrong
    # guess then reads as "arrow code 2 read as left", which is one line to correct, instead of
    # moving a bay the wrong way in silence.
    ARROW_CODES = {
        63232: (0.0, -1.0), 63233: (0.0, 1.0), 63234: (-1.0, 0.0), 63235: (1.0, 0.0),
        65361: (-1.0, 0.0), 65362: (0.0, -1.0), 65363: (1.0, 0.0), 65364: (0.0, 1.0),
        0: (0.0, -1.0), 1: (0.0, 1.0), 2: (-1.0, 0.0), 3: (1.0, 0.0),
    }
    LOW_ARROW_CODES = frozenset((0, 1, 2, 3))
    # Codes that are not arrows and cannot be typed, but are still real keys the loops handle as
    # characters: Tab cycles the selection, Enter is a stray, Esc quits, -1/255 mean no key at
    # all. Reporting these would put a warning on screen every time somebody presses Tab.
    QUIET_CODES = frozenset((-1, 255, 8, 9, 10, 13, 27))
    ARROW_NAMES = {(0.0, -1.0): "up", (0.0, 1.0): "down", (-1.0, 0.0): "left",
                   (1.0, 0.0): "right"}

    def act_code(self, code: int) -> bool:
        """Handle a key that is not a plain character. True when it was handled.

        Called before the caller slices the key with `& 0xFF`, which would keep the low byte of
        63232 - zero - and throw the direction away.

        Anything else that cannot be a printable character is reported, in the window and in the
        terminal. That report is the only way this can be fixed for a backend nobody has seen yet,
        and it is deliberately loud: the first arrow press either moves something or says its own
        number.
        """
        step = self.ARROW_CODES.get(code)
        if step is None:
            if code not in self.QUIET_CODES and not (32 <= code < 127):
                note = (f"key code {code} does nothing here - if that was an arrow key, "
                        f"add it to ARROW_CODES")
                self.ed._say(note, "warn")
                print(f"  {note}", file=sys.stderr)
            return False
        self._fine_nudge(step[0], step[1], code if code in self.LOW_ARROW_CODES else None)
        return True

    def _fine_nudge(self, sx: float, sy: float, arrow_code: Optional[int] = None) -> None:
        """Move the selected corner - or the whole bay - by half a pixel.

        Half a pixel cannot be seen, so what moved is said out loud as a coordinate: that number
        is the only feedback that tells twenty presses from nineteen. When the key arrived as one
        of the guessed low codes, the message says so too, so a wrong guess is visible at once.
        """
        ed = self.ed
        guessed = f"  [arrow code {arrow_code}, guessed]" if arrow_code is not None else ""
        if self.layer == "roads":
            if self.road.nudge(sx * self.FINE_PX, sy * self.FINE_PX, coalesce=self._fine_run):
                node = self.road.selected_node()
                where = f"({node.x:.1f}, {node.y:.1f})" if node is not None else ""
                ed._say(f"road point {self.ARROW_NAMES[(sx, sy)]} to {where}{guessed}")
                self._fine_run = True
                return
            ed._say("select a road point first - click one", "warn")
            return

        dx, dy = sx * self.FINE_PX, sy * self.FINE_PX
        # One undo step for a whole run of presses, not one per press: a 10 px correction is
        # twenty presses, and twenty entries would bury everything else in the stack.
        if ed.nudge_corner(dx, dy, coalesce=self._fine_run):
            spot = ed.selected_spot()
            corner = ed.selected_corner
            x, y = spot.polygon[corner]
            ed._say(f"corner {corner + 1} of spot {spot.id} {self.ARROW_NAMES[(sx, sy)]} "
                    f"to ({x:.1f}, {y:.1f}){guessed}")
            self._fine_run = True
            return
        if ed.nudge(dx, dy, coalesce=self._fine_run):
            spot = ed.selected_spot()
            cx, cy = spot.polygon.mean(axis=0)
            ed._say(f"spot {spot.id} {self.ARROW_NAMES[(sx, sy)]} to ({cx:.1f}, {cy:.1f}){guessed}")
            self._fine_run = True
            return
        ed._say("select a bay first - click one, or Tab to cycle", "warn")

    def act(self, key: str) -> bool:
        """Run one action. Returns False when the app should quit."""
        ed = self.ed
        # Any ordinary key ends a run of fine nudges, so the next arrow press starts a new undo
        # step rather than joining the previous one. See _fine_nudge.
        self._fine_run = False
        # The layer tabs arrive as @spots / @roads: they are buttons, not keys. Whichever
        # layer is active decides where every other key and click goes.
        if key in ("@spots", "@roads"):
            self.switch_layer("roads" if key == "@roads" else "spots")
            return True
        if key == "t":
            self.switch_layer("roads" if self.layer == "spots" else "spots")
            return True
        if key in ("q", "\x1b"):
            self._save_all()
            return False
        if self.layer == "roads":
            return self._act_roads(key)
        if key == "a":
            frame = getattr(self, "_frame", None)
            if frame is None:
                ed._say("no frame yet", "warn")
            else:
                ed.auto_detect(frame, self.detector_params)
        elif key == "g":
            ed.generate_row(+1, 1)
        elif key == "G":
            ed.generate_row(-1, 1)
        elif key == "n":
            ed.add_at(*self.cursor)
        elif key in ("d", "x"):
            ed.delete_selected()
        elif key in ("r", "R"):
            self._pending_prompt = "rename"
        elif key == "i":
            self._pending_prompt = "inner"
        elif key == "j":
            # Back to working the entrance out, for a bay whose side was set by hand.
            ed.set_access_side(ed.selected, None)
        elif key == "s":
            ed.save(self.out_path)
        elif key == "o":
            ed.load(self.out_path)
        elif key == "A":
            self.align_to_template()
        elif key == "z":
            ed.undo()
        elif key == "c":
            ed.clear()
        elif key == "l":
            self.show_overlay = not self.show_overlay
            ed._say(f"line overlay {'on' if self.show_overlay else 'off'}")
        elif key == "v":
            self.show_inner = not self.show_inner
        elif key == "\t":
            ed.select_next()
        elif key == "m":
            order = ["bay", "row", "all"]
            self.nudge_scope = order[(order.index(self.nudge_scope) + 1) % len(order)]
            ed._say(f"2/4/6/8 now move: {self.nudge_scope}")
        elif key == "4":
            self._nudges(-NUDGE, 0)
        elif key == "6":
            self._nudges(NUDGE, 0)
        elif key == "8":
            self._nudges(0, -NUDGE)
        elif key == "2":
            self._nudges(0, NUDGE)
        return True

    def switch_layer(self, layer: str) -> None:
        self.layer = "roads" if layer == "roads" else "spots"
        if self.layer == "roads":
            if not self.road.net.nodes:
                self.road._say("click on the image to draw where a car can drive")
            else:
                problems = self.road.problems(self.ed.spots)
                self.road._say(problems[0] if problems else "map looks complete")
        else:
            self.ed._say("Parking Spots layer")

    def _save_all(self) -> None:
        """Write the map document once, whichever layers changed.

        Neither write can destroy the other layer: save_spots() re-emits the road sections
        it finds in the file and RoadNetwork.save() re-emits the bay section, so the order
        does not matter and one file stays the single source of truth for both layers.
        """
        wrote = False
        if self.ed.dirty and self.ed.spots:
            self.ed.save(self.out_path)
            wrote = True
        if self.road.dirty:
            self.road.save(self.out_path)
            wrote = True
        if not wrote:
            print("  nothing changed, nothing written")
            return
        # The saved file is the editable map; the page reads the *emitted* one. A rename changes a
        # bay's id here and would leave the page describing a bay that does not exist, so the two
        # documents are kept in step at the one moment they can drift.
        import emit_map
        if emit_map.emit(self.out_path, emit_map.out_for(self.out_path), quiet=True) not in (0, 1):
            print("  warning: could not re-emit out/parking_map.json", file=sys.stderr)
        else:
            print("  also re-emitted out/parking_map.json (the page reads that one)")

    def _act_roads(self, key: str) -> bool:
        """Road layer keys.

        The job is one thing: draw where a car can drive. Where each bay connects is worked
        out from that, so there is no key for it - pressing 'p' only reports on the result.
        """
        road = self.road
        spots = self.ed.spots
        if key == "n":
            road.break_chain()
        elif key == "k":
            road.toggle_entrance()
        elif key in ("d", "x"):
            road.delete_selected()
        elif key == "s":
            self._save_all()
        elif key == "o":
            road.load(self.out_path)
        elif key == "z":
            road.undo()
        elif key == "c":
            road.clear()
        elif key == "\t":
            road.select_next()
        elif key == "4":
            road.nudge(-NUDGE, 0)
        elif key == "6":
            road.nudge(NUDGE, 0)
        elif key == "8":
            road.nudge(0, -NUDGE)
        elif key == "2":
            road.nudge(0, NUDGE)
        elif key == "a":
            road._say("roads are drawn by hand: where a car can drive is a decision, not "
                      "something to find in the picture", "warn")
        elif key == "p":
            problems = road.problems(spots)
            if problems:
                for msg in problems[:5]:
                    print(f"  map: {msg}")
                road._say(problems[0], "warn")
            else:
                bays = len(road.derived(spots).spot_nodes())
                road._say(f"map complete: {bays} bay entrances derived, all reachable "
                          f"from the entrance")
        return True

    def _nudges(self, dx: float, dy: float) -> None:
        """Apply a nudge to whatever the current scope covers.

        A camera that has been nudged usually moves a whole row by the same amount, so
        correcting 48 bays one at a time is 48 corrections where four will do.
        """
        ed = self.ed
        if self.nudge_scope == "all":
            n = ed.nudge_many(ed.spots, dx, dy)
            ed._say(f"moved all {n} bay(s) by {dx:+.0f},{dy:+.0f}")
            return
        if self.nudge_scope == "row":
            rows = ed.rows_by_y()
            sel = ed.selected
            targets = None
            if isinstance(sel, int):
                for row in rows:
                    if sel in row:
                        targets = [ed.spots[i] for i in row]
                        break
            if targets is None and rows:
                targets = [ed.spots[i] for i in rows[0]]
            n = ed.nudge_many(targets or [], dx, dy)
            ed._say(f"moved row of {n} bay(s) by {dx:+.0f},{dy:+.0f}")
            return
        ed.nudge(dx, dy)

    def prompt(self) -> None:
        """Terminal prompts run outside the render loop, with the window closed."""
        kind, self._pending_prompt = self._pending_prompt, None
        spot = self.ed.selected_spot()
        try:
            if kind == "rename" and spot is not None:
                # The terminal is a second window and it is easy to miss a bare prompt buried in
                # the diagnostics, so this one is not bare. Renaming happens here rather than in
                # the picture because OpenCV cannot read a line of text.
                print("\n  == 改名：请在【这个终端窗口】输入新的车位编号，然后按回车 ==")
                new_id = input(f"  新编号（现在叫 {spot.id}）> ").strip()
                if new_id:
                    if self.ed.rename_selected(new_id):
                        print(f"  => 已改名为 {new_id}；按 s 立即保存，按 q 退出时也会自动保存")
                    else:
                        print("  => 没改成（编号不能为空，也不能和别的车位重复）")
                else:
                    print("  => 取消（没有输入）")
            elif kind == "inner" and spot is not None:
                raw = input(f"  inner detection scale for spot {spot.id} "
                            f"[0.2-0.95, now {spot.inner_scale:.2f}]: ").strip()
                if raw:
                    self.ed.set_inner_scale(float(raw))
        except (EOFError, ValueError):
            print("  (cancelled)")

    # -- loop ---------------------------------------------------------------

    def run(self, get_frame: Callable[[], Optional[np.ndarray]],
            detector_params: DetectorParams) -> int:
        self.detector_params = detector_params
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)

        frame = None
        last_good = None
        while True:
            fresh = get_frame()
            if fresh is not None:
                frame = fresh
                last_good = fresh
            elif frame is None:
                frame = last_good
            if frame is None:
                if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                    break
                continue
            self._frame = frame

            while self._pending_prompt is not None:
                cv2.destroyAllWindows()
                self.prompt()
                cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
                cv2.setMouseCallback(WINDOW, self.on_mouse)

            cv2.imshow(WINDOW, self.render(frame))
            # 5 ms, not 20. Measured on this machine: waitKey has a ~19 ms floor here, so a loop
            # asking for 20 ms spent 37 ms per frame while a drag waited to be serviced, and
            # asking for 5 ms spends 26 ms. The drawing is no longer the limit (6 ms) - the event
            # pump is, and a shorter wait is what makes a dragged corner follow the pointer.
            raw = cv2.waitKey(5)
            # PARKING_LOG_KEYS=1 names every code that arrives, which is how the arrows get
            # measured on a machine nobody here can press keys on.
            if raw != -1 and self.log_keys:
                print(f"  key code {raw}", file=sys.stderr)
            # Arrow keys arrive as numbers, not characters, and `& 0xFF` keeps only the low byte
            # of them - 63232 becomes 0 and the direction is gone. So the raw code is offered to
            # the window first, and only what it declines is treated as text.
            if raw != -1 and self.act_code(raw):
                continue
            key = raw & 0xFF
            if key == 255:
                continue
            if not self.act(chr(key)):
                break
        cv2.destroyAllWindows()
        return 0


def write_grid_aid(frame: np.ndarray, out_path: str, step: int = 50) -> None:
    """Save the frame with a labelled grid: a ruler for calibrating by hand."""
    h, w = frame.shape[:2]
    canvas = frame.copy()
    for x in range(0, w, step):
        cv2.line(canvas, (x, 0), (x, h), (0, 200, 255), 1)
        for y in range(0, h, 4 * step):
            cv2.putText(canvas, str(x), (x + 3, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(canvas, str(x), (x + 3, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    for y in range(0, h, step):
        cv2.line(canvas, (0, y), (w, y), (0, 200, 255), 1)
        for x in range(0, w, 4 * step):
            cv2.putText(canvas, str(y), (x + 30, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(canvas, str(y), (x + 30, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, canvas)
    print(f"wrote calibration aid to {out_path} ({w}x{h}, grid every {step}px)")


def grab(get_frame: Callable[[], Optional[np.ndarray]], tries: int = 40,
         delay: float = 0.05) -> Optional[np.ndarray]:
    """Read until a frame arrives. An MJPEG reader needs a few reads per frame."""
    for _ in range(tries):
        frame = get_frame()
        if frame is not None:
            return frame
        time.sleep(delay)
    return None


def detect_only(frame: np.ndarray, params: DetectorParams, out_dir: str,
                start_id: int) -> int:
    """Run detection once, print what it found, save an overlay, and exit.

    This is how the tool gets tuned against the real scene without clicking
    through a window: run it, look at the numbers and the overlay, adjust
    --min-width / --max-width / --band-threshold / --grid-tolerance, repeat.
    """
    res = detect_bays(frame, params)
    d = res.diagnostics
    print(f"frame {frame.shape[1]}x{frame.shape[0]}  method={d.get('method')} "
          f"orientation={d.get('orientation')} bands={d.get('bands')} "
          f"dividers={d.get('dividers')}")
    print(f"candidates: {res.count}   rejected pairs: {d.get('rejected_bays')}")
    for i, row in enumerate(d.get("rows", [])):
        print(f"  row {i}: {row}")
    polys = sort_reading_order(res.polygons)
    pitches = [r.get("grid_spacing") for r in d.get("rows", []) if r.get("grid_spacing")]
    if pitches:
        lo, hi = min(pitches), max(pitches)
        print(f"  measured row pitch: {lo:.0f}-{hi:.0f} px  ->  try "
              f"--min-width {int(lo * 0.8)} --max-width {int(hi * 1.25)}")
    if polys:
        widths = [np.asarray(p)[:, 0].max() - np.asarray(p)[:, 0].min() for p in polys]
        heights = [np.asarray(p)[:, 1].max() - np.asarray(p)[:, 1].min() for p in polys]
        print(f"  bay width  min/median/max = {min(widths):.0f}/{np.median(widths):.0f}/{max(widths):.0f}")
        print(f"  bay height min/median/max = {min(heights):.0f}/{np.median(heights):.0f}/{max(heights):.0f}")
        for n, p in enumerate(polys[:12]):
            a = np.asarray(p)
            print(f"    #{start_id + n}  x {a[:, 0].min():4.0f}..{a[:, 0].max():4.0f}  "
                  f"y {a[:, 1].min():4.0f}..{a[:, 1].max():4.0f}  "
                  f"w {a[:, 0].max() - a[:, 0].min():3.0f} h {a[:, 1].max() - a[:, 1].min():3.0f}")
    for line in render_profile_ascii(res):
        print(line)
    os.makedirs(out_dir, exist_ok=True)
    overlay = draw_debug(frame, res)
    for n, p in enumerate(polys):
        cv2.polylines(overlay, [np.round(np.asarray(p)).astype(np.int32)], True, (255, 0, 255), 2)
        c = np.asarray(p).mean(axis=0).astype(int)
        cv2.putText(overlay, str(start_id + n), (c[0] - 10, c[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    path = os.path.join(out_dir, "detect_only.png")
    cv2.imwrite(path, overlay)
    print(f"overlay -> {path}")
    return 0


def write_detected_spots(frame: np.ndarray, params: DetectorParams, args,
                        out_path: str) -> None:
    """Save the detected candidates as an editable config.

    A convenience for the workflow "let the detector propose the whole lot, then
    delete what is wrong" - the file it writes is the same format the calibrator
    produces, so it can be opened and corrected in the window.
    """
    from detector import detect_bays, sort_reading_order
    from spot_config import Spot, save_spots, validate

    res = detect_bays(frame, params)
    spots = []
    for i, poly in enumerate(sort_reading_order(res.polygons)):
        cand = Spot(id=str(args.start_id + i),
                    polygon=np.asarray(poly, dtype=np.float64),
                    inner_scale=args.inner_scale)
        try:
            cand.prepare(frame.shape[:2])
        except Exception:
            continue
        spots.append(cand)
    if not spots:
        print("nothing detected, not writing a config")
        return
    save_spots(out_path, spots, frame.shape[:2])
    print(f"wrote {len(spots)} detected candidate(s) to {out_path}")
    for problem in validate(spots, frame.shape[:2]):
        print(f"  warning: {problem}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Auto-detect + hand-correct parking bay calibration.")
    ap.add_argument("--source", default="http://192.168.3.118/stream")
    ap.add_argument("--backend", default="auto", choices=("auto", "opencv", "mjpeg"))
    ap.add_argument("--frame", help="use this still image instead of the live stream")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "..", "config", "parking_spots.json"))
    ap.add_argument("--start-id", type=int, default=38)
    ap.add_argument("--inner-scale", type=float, default=DEFAULT_INNER_SCALE)
    ap.add_argument("--auto", action="store_true", help="run Auto Detect once at startup")
    ap.add_argument("--template", metavar="IMAGE",
                    help="frame the bays were drawn on, for --auto-align "
                         "(default: --frame, else data/reference/reference.jpg)")
    ap.add_argument("--auto-align", action="store_true",
                    help="move every bay onto the current frame and save, then exit. "
                         "Cancel camera drift without adding or deleting bays.")
    ap.add_argument("--align-motion", default="translation",
                    choices=("translation", "euclidean", "affine"))
    ap.add_argument("--grid", nargs="?", const="", metavar="IMAGE",
                    help="write a labelled calibration ruler image and exit")
    ap.add_argument("--write-spots", metavar="JSON",
                    help="with --detect-only: also write the detected candidates to this "
                         "config file, as a starting point to correct by hand")
    ap.add_argument("--detect-only", action="store_true",
                    help="run detection once, print the report, save an overlay and exit "
                         "(no window; use this to tune the detector against the real scene)")
    # detector knobs, exposed so the tool can be tuned against the real scene
    ap.add_argument("--detect-method", default=DetectorParams.method,
                    choices=("profile", "segments"))
    ap.add_argument("--detect-orientation", default=DetectorParams.orientation,
                    choices=("both", "vertical", "horizontal", "auto"))
    ap.add_argument("--min-width", type=int, default=DetectorParams.min_width)
    ap.add_argument("--max-width", type=int, default=DetectorParams.max_width)
    ap.add_argument("--band-threshold", type=float, default=DetectorParams.band_threshold)
    ap.add_argument("--grid-tolerance", type=float, default=DetectorParams.grid_tolerance)
    args = ap.parse_args()

    params = DetectorParams(method=args.detect_method, orientation=args.detect_orientation,
                            min_width=args.min_width, max_width=args.max_width,
                            band_threshold=args.band_threshold,
                            grid_tolerance=args.grid_tolerance)

    cam = None
    still = None
    if args.frame:
        still = cv2.imread(args.frame, cv2.IMREAD_COLOR)
        if still is None:
            print(f"error: could not read {args.frame}", file=sys.stderr)
            return 2
        get_frame = lambda: still                               # noqa: E731
        shape = still.shape[:2]
        print(f"offline: {args.frame} ({shape[1]}x{shape[0]})")
    else:
        try:
            cam = open_camera(args.source, backend=args.backend)
        except CameraError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        width = 0
        first = None
        deadline = time.time() + 15
        while first is None and time.time() < deadline:
            first = cam.read()
        if first is None:
            print("error: no frame from the camera", file=sys.stderr)
            cam.close()
            return 2
        get_frame = cam.read
        shape = first.shape[:2]
        print(f"live: {args.source} ({shape[1]}x{shape[0]}) backend {cam.name}")

    editor = SpotEditor(shape, start_id=args.start_id, inner_scale=args.inner_scale)
    out_path = os.path.abspath(args.out)
    debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "debug")

    if args.grid is not None:
        img = cv2.imread(args.grid, cv2.IMREAD_COLOR) if args.grid else (
            still if still is not None else get_frame())
        if img is None:
            print("error: no image to draw the grid on", file=sys.stderr)
            return 2
        write_grid_aid(img, os.path.join(debug_dir, "calibration_aid.png"))
        if cam is not None:
            cam.close()
        return 0

    if args.detect_only:
        img = still if still is not None else grab(get_frame)
        if img is None:
            print("error: no frame", file=sys.stderr)
            return 2
        rc = detect_only(img, params, debug_dir, args.start_id)
        if args.write_spots:
            write_detected_spots(img, params, args, args.write_spots)
        if cam is not None:
            cam.close()
        return rc

    if os.path.exists(out_path):
        editor.load(out_path)
    else:
        editor._say("no calibration yet: press 'a' to auto detect, or add bays by hand (n)")

    template = None
    template_path = args.template or args.frame or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "reference",
        "reference.jpg")
    if os.path.exists(template_path):
        template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        if template is not None:
            print(f"template for alignment: {template_path} "
                  f"({template.shape[1]}x{template.shape[0]})")

    app = CalibrationApp(editor, out_path, template=template,
                         align_motion=args.align_motion)

    if args.auto_align:
        live = still if still is not None else grab(get_frame)
        if live is None:
            print("error: no frame to align", file=sys.stderr)
            return 2
        app._frame = live
        ok = app.align_to_template(save=True)
        if cam is not None:
            cam.close()
        return 0 if ok else 1

    if args.auto:
        frame = still if still is not None else grab(get_frame)
        if frame is not None:
            editor.auto_detect(frame, params)
        else:
            editor._say("auto detect skipped: no frame yet", "warn")

    try:
        rc = app.run(get_frame, params)
    finally:
        if cam is not None:
            cam.close()

    if editor.spots:
        print(f"\n{len(editor.spots)} spot(s): {', '.join(s.id for s in editor.spots)}")
        if editor.dirty:
            editor.save(out_path)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
