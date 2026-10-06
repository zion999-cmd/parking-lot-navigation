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
    A  align all bays to the current frame (camera drift; adds/deletes nothing)
    a  Auto Detect (only ADDS candidates; existing work is never touched)
    g  generate the next bay in the row from the selected one (right)
    G  ... same, to the left
    n  add a new empty bay at the mouse position
    d  delete the selected bay
    r  rename the selected bay (type the id in the terminal)
    i  set the inner detection scale of the selected bay (terminal)
    s  save to config/parking_spots.json      o  load it
    z  undo                                   c  reset (clear all)
    l  toggle the detected lines overlay      v  toggle inner detection polygons
    Tab cycle selection      2/4/6/8 nudge by 5 px
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

BAR_H = 34
NUDGE = 5


class CalibrationApp:
    """Window + input handling on top of SpotEditor (which holds the state)."""

    def __init__(self, editor: SpotEditor, out_path: str,
                 template: Optional[np.ndarray] = None,
                 align_motion: str = "translation"):
        self.ed = editor
        self.out_path = out_path
        self.template = template          # the frame the bays were drawn on
        self.align_motion = align_motion
        self.last_align = None
        self.show_inner = True
        self.show_overlay = False
        self.cursor: Tuple[int, int] = (0, 0)
        self._buttons: List[Tuple[int, int, int, int, str, str]] = []
        self._pending_prompt: Optional[str] = None

    # -- drawing ------------------------------------------------------------

    def render(self, frame: np.ndarray) -> np.ndarray:
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        if self.show_overlay and self.ed.detection is not None:
            canvas = draw_debug(canvas, self.ed.detection)

        for i, spot in enumerate(self.ed.spots):
            sel = (i == self.ed.selected)
            poly = np.round(spot.polygon).astype(np.int32)
            cv2.polylines(canvas, [poly], True, COL_SEL if sel else COL_BAY, 3 if sel else 2)
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

        header = (f"spots {len(self.ed.spots)}   "
                  f"selected {self.ed.spots[self.ed.selected].id if self.ed.selected is not None else '-'}   "
                  f"inner {self.ed.inner_scale:.2f}   "
                  f"{'*unsaved*' if self.ed.dirty else 'saved'}")
        cv2.putText(canvas, header, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(canvas, header, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_TEXT, 1)

        self._buttons = []
        x = 8
        for label, key in (("Auto Detect", "a"), ("Gen Row", "g"), ("Save", "s"),
                           ("Undo", "z"), ("Reset", "c")):
            bw = 12 + len(label) * 9
            self._buttons.append((x, h - BAR_H - 8, bw, BAR_H - 8, label, key))
            x += bw + 8

        msg = self.ed.message
        colour = {"info": COL_OK, "warn": COL_WARN, "error": COL_ERR}.get(msg.level, COL_TEXT)
        if msg.text:
            cv2.putText(canvas, msg.text[:118], (8, h - BAR_H - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 4)
            cv2.putText(canvas, msg.text[:118], (8, h - BAR_H - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 1)

        hint = ("drag handles | A align | a auto | g gen row | n add | d del | "
                "r rename | i inner | s save | o load | z undo | c reset | l lines | "
                "v inner | Tab next | 2468 nudge | q quit")
        cv2.putText(canvas, hint[:135], (8, h - BAR_H - 28),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 3)
        cv2.putText(canvas, hint[:135], (8, h - BAR_H - 28),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 200, 200), 1)

        for (bx, by, bw, bh, label, _key) in self._buttons:
            hover = bx <= self.cursor[0] <= bx + bw and by <= self.cursor[1] <= by + bh
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh),
                          COL_BTN_HOT if hover else COL_BTN, -1)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (200, 200, 200), 1)
            cv2.putText(canvas, label, (bx + 6, by + bh - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_TEXT, 1)
        return canvas

    # -- input --------------------------------------------------------------

    def _button_at(self, x: int, y: int) -> Optional[str]:
        for (bx, by, bw, bh, _label, key) in self._buttons:
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return key
        return None

    def on_mouse(self, event, x, y, flags, param) -> None:
        self.cursor = (x, y)

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

    def act(self, key: str) -> bool:
        """Run one action. Returns False when the app should quit."""
        ed = self.ed
        if key in ("q", "\x1b"):
            if ed.dirty and ed.spots:
                ed.save(self.out_path)
            return False
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
        elif key == "r":
            self._pending_prompt = "rename"
        elif key == "i":
            self._pending_prompt = "inner"
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
        elif key == "4":
            ed.nudge(-NUDGE, 0)
        elif key == "6":
            ed.nudge(NUDGE, 0)
        elif key == "8":
            ed.nudge(0, -NUDGE)
        elif key == "2":
            ed.nudge(0, NUDGE)
        return True

    def prompt(self) -> None:
        """Terminal prompts run outside the render loop, with the window closed."""
        kind, self._pending_prompt = self._pending_prompt, None
        spot = self.ed.selected_spot()
        try:
            if kind == "rename" and spot is not None:
                new_id = input(f"  new id for spot {spot.id}: ").strip()
                if new_id:
                    self.ed.rename_selected(new_id)
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
            key = cv2.waitKey(20) & 0xFF
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
