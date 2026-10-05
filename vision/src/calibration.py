"""Interactive parking bay calibration.

Draws the live camera image and lets you click the four corners of a bay. The
result goes to ``config/parking_spots.json``; nothing else in the project hard
codes coordinates.

Click order does not matter (any convex quadrilateral works), but go around the
bay rather than crossing over, or the polygon will be a bow tie.

Keys
----
    left click   add one corner (the 4th click commits the spot)
    u            remove the last corner of the polygon being drawn
    d            delete the last committed spot
    r            rename the last committed spot (type the id in the terminal)
    i            set the inner (detection) scale for the last spot
    s            save to the JSON file
    l            reload the JSON file
    g            toggle the coordinate grid
    v            toggle showing the inner detection polygon
    q / ESC      save and quit

The outline is the bay; the dashed-looking second outline inside it is what
occupancy actually looks at. Keeping the detection area inside the bay is what
keeps white bay lines, bay numbers and kerb edges out of the difference.

Two ways to work:

* live   - ``python src/calibration.py --source http://192.168.3.118/stream``
* offline- ``python src/calibration.py --frame data/debug/frame.jpg``
           ``python src/calibration.py --grid data/debug/frame.jpg``
           (``--grid`` just writes an annotated ruler image and exits, for when a
           GUI is not convenient; you can then edit the JSON by hand)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera import CameraError, open_camera            # noqa: E402
from spot_config import (DEFAULT_INNER_SCALE, Spot, SpotConfigError,  # noqa: E402
                         load_spots, save_spots, validate)

WINDOW = "Parking bay calibration"

COL_BAY = (0, 255, 255)        # yellow outline: the whole bay
COL_INNER = (0, 200, 0)        # green: what the detector looks at
COL_DRAW = (255, 120, 0)       # blue-ish: polygon being drawn
COL_TEXT = (255, 255, 255)


class Calibrator:
    def __init__(self, out_path: str, start_id: int, inner_scale: float,
                 width: int, height: int):
        self.out_path = out_path
        self.next_id = start_id
        self.inner_scale = inner_scale
        self.spots: List[Spot] = []
        self.pending: List[Tuple[int, int]] = []
        self.cursor: Tuple[int, int] = (0, 0)
        self.show_grid = False
        self.show_inner = True
        self.dirty = False
        self.width = width
        self.height = height

    # -- input --------------------------------------------------------------

    def on_mouse(self, event, x, y, flags, param) -> None:
        self.cursor = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.pending) >= 4:
                print("  already have 4 corners; press 'u' to undo or 'd' to drop the last spot")
                return
            self.pending.append((x, y))
            print(f"  corner {len(self.pending)}/4 at ({x}, {y})")
            if len(self.pending) == 4:
                self.commit()

    def commit(self) -> None:
        poly = np.asarray(self.pending, dtype=np.float64)
        spot = Spot(id=str(self.next_id), polygon=poly, inner_scale=self.inner_scale)
        try:
            spot.prepare((self.height, self.width))
        except SpotConfigError as exc:
            print(f"  rejected: {exc}")
            return
        self.spots.append(spot)
        print(f"  committed spot {spot.id}  bay area={spot.area():.0f}px^2  "
              f"detect area={spot.inner_area()}px^2 (scale {self.inner_scale:.2f})")
        self.pending = []
        self.next_id += 1
        self.dirty = True
        if len(self.spots) == 3:
            print("  (3 spots is the target for this stage - press 's' to save)")

    # -- terminal actions ---------------------------------------------------

    def rename_last(self) -> None:
        if not self.spots:
            print("  nothing to rename")
            return
        try:
            new_id = input(f"  new id for spot {self.spots[-1].id}: ").strip()
        except EOFError:
            return
        if new_id:
            self.spots[-1].id = new_id
            self.dirty = True
            print(f"  renamed to {new_id}")

    def set_inner_scale(self) -> None:
        if not self.spots:
            print("  nothing to adjust")
            return
        try:
            raw = input(f"  inner detection scale for spot {self.spots[-1].id} "
                        f"[0.2-0.95, current {self.spots[-1].inner_scale:.2f}]: ").strip()
        except EOFError:
            return
        if not raw:
            return
        try:
            val = float(raw)
        except ValueError:
            print("  not a number")
            return
        if not 0.2 <= val <= 0.95:
            print("  refused: keep it in 0.2-0.95")
            return
        self.spots[-1].inner_scale = val
        self.spots[-1].prepare((self.height, self.width))
        self.dirty = True
        print(f"  spot {self.spots[-1].id}: inner scale {val:.2f}, "
              f"detect area {self.spots[-1].inner_area()}px^2")

    # -- persistence --------------------------------------------------------

    def save(self) -> None:
        save_spots(self.out_path, self.spots, (self.height, self.width))
        self.dirty = False
        print(f"  saved {len(self.spots)} spot(s) to {self.out_path}")

    def reload(self) -> None:
        try:
            self.spots = load_spots(self.out_path, (self.height, self.width))
        except SpotConfigError as exc:
            print(f"  reload failed: {exc}")
            return
        self.dirty = False
        print(f"  reloaded {len(self.spots)} spot(s) from {self.out_path}")

    # -- drawing ------------------------------------------------------------

    def draw(self, frame: np.ndarray) -> np.ndarray:
        canvas = frame.copy()

        if self.show_grid:
            for x in range(0, self.width, 50):
                cv2.line(canvas, (x, 0), (x, self.height), (90, 90, 90), 1)
                cv2.putText(canvas, str(x), (x + 2, 14), cv2.FONT_HERSHEY_PLAIN, 1.0,
                            (200, 200, 200), 1)
            for y in range(0, self.height, 50):
                cv2.line(canvas, (0, y), (self.width, y), (90, 90, 90), 1)
                cv2.putText(canvas, str(y), (2, y + 14), cv2.FONT_HERSHEY_PLAIN, 1.0,
                            (200, 200, 200), 1)

        for spot in self.spots:
            poly = np.round(spot.polygon).astype(np.int32)
            cv2.polylines(canvas, [poly], True, COL_BAY, 2)
            if self.show_inner and spot.inner_points is not None:
                inner = np.round(spot.inner_points).astype(np.int32)
                cv2.polylines(canvas, [inner], True, COL_INNER, 2)
            cx, cy = spot.polygon.mean(axis=0).astype(int)
            cv2.putText(canvas, f"Spot {spot.id}", (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_BAY, 2)

        if self.pending:
            pts = np.asarray(self.pending, dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(canvas, [pts], False, COL_DRAW, 2)
            for i, (x, y) in enumerate(self.pending):
                cv2.circle(canvas, (x, y), 4, COL_DRAW, -1)
                cv2.putText(canvas, str(i + 1), (x + 6, y - 6),
                            cv2.FONT_HERSHEY_PLAIN, 1.2, COL_DRAW, 1)

        help_lines = [
            f"spots: {len(self.spots)}   pending corners: {len(self.pending)}/4   "
            f"inner scale: {self.inner_scale:.2f}{'  *unsaved*' if self.dirty else ''}",
            f"cursor: {self.cursor}    click 4 corners | u undo | d delete | r rename | "
            f"i inner scale | s save | l reload | g grid | v inner | q quit",
        ]
        for i, text in enumerate(help_lines):
            cv2.putText(canvas, text, (10, self.height - 30 + i * 16),
                        cv2.FONT_HERSHEY_PLAIN, 1.05, (0, 0, 0), 3)
            cv2.putText(canvas, text, (10, self.height - 30 + i * 16),
                        cv2.FONT_HERSHEY_PLAIN, 1.05, COL_TEXT, 1)
        return canvas


def write_grid_aid(frame: np.ndarray, out_path: str, step: int = 50) -> None:
    """Save the frame with a labelled grid, as a ruler for manual calibration."""
    h, w = frame.shape[:2]
    canvas = frame.copy()
    for x in range(0, w, step):
        cv2.line(canvas, (x, 0), (x, h), (0, 200, 255), 1)
        for y in range(0, h, 4 * step):
            cv2.putText(canvas, str(x), (x + 3, y + 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3)
            cv2.putText(canvas, str(x), (x + 3, y + 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1)
    for y in range(0, h, step):
        cv2.line(canvas, (0, y), (w, y), (0, 200, 255), 1)
        for x in range(0, w, 4 * step):
            cv2.putText(canvas, str(y), (x + 30, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3)
            cv2.putText(canvas, str(y), (x + 30, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, canvas)
    print(f"wrote calibration aid to {out_path} ({w}x{h}, grid every {step}px)")


def grab_one_frame(source: str, timeout: float = 15.0) -> np.ndarray:
    cam = open_camera(source)
    deadline = time.time() + timeout
    while time.time() < deadline:
        frame = cam.read()
        if frame is not None:
            cam.close()
            return frame
    cam.close()
    raise CameraError(f"no frame from {source!r} within {timeout:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate parking bay polygons.")
    ap.add_argument("--source", default="http://192.168.3.118/stream",
                    help="camera stream URL or video file")
    ap.add_argument("--frame", help="use this still image instead of the live stream")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "..", "config", "parking_spots.json"))
    ap.add_argument("--start-id", type=int, default=38,
                    help="id given to the first spot (default 38)")
    ap.add_argument("--inner-scale", type=float, default=DEFAULT_INNER_SCALE,
                    help="detection polygon shrink factor (default 0.62)")
    ap.add_argument("--grid", nargs="?", const="", metavar="IMAGE",
                    help="write a labelled calibration ruler image and exit "
                         "(--grid alone uses --frame or grabs one frame)")
    args = ap.parse_args()

    if args.grid is not None:
        if args.grid:
            img = cv2.imread(args.grid, cv2.IMREAD_COLOR)
            if img is None:
                print(f"error: could not read {args.grid}", file=sys.stderr)
                return 2
        elif args.frame:
            img = cv2.imread(args.frame, cv2.IMREAD_COLOR)
            if img is None:
                print(f"error: could not read {args.frame}", file=sys.stderr)
                return 2
        else:
            img = grab_one_frame(args.source)
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "data", "debug", "calibration_aid.png")
        write_grid_aid(img, out)
        return 0

    if args.frame:
        frame = cv2.imread(args.frame, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"error: could not read {args.frame}", file=sys.stderr)
            return 2
        reuse_still = True
    else:
        try:
            cam = open_camera(args.source)
        except CameraError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        frame = None
        deadline = time.time() + 15
        while frame is None and time.time() < deadline:
            frame = cam.read()
        if frame is None:
            print("error: no frame received", file=sys.stderr)
            return 2
        reuse_still = False

    h, w = frame.shape[:2]
    cal = Calibrator(os.path.abspath(args.out), args.start_id, args.inner_scale, w, h)
    try:
        if os.path.exists(cal.out_path):
            cal.reload()
    except Exception:
        pass

    print(f"calibrating {w}x{h}. Existing spots: {len(cal.spots)}")
    print("click the four corners of a bay; the 4th click commits it.")
    print("press 's' when done - nothing is written until you save.")

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, cal.on_mouse)

    while True:
        if not reuse_still:
            latest = cam.read()
            if latest is not None:
                frame = latest
            elif frame is None:
                continue

        cv2.imshow(WINDOW, cal.draw(frame))
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('q'), 27):
            if cal.dirty:
                cal.save()
            break
        elif key == ord('u'):
            if cal.pending:
                print(f"  removed corner {cal.pending.pop()}")
            elif cal.spots:
                print("  (nothing being drawn - press 'd' to drop the last spot)")
        elif key == ord('d'):
            if cal.spots:
                print(f"  deleted spot {cal.spots.pop().id}")
                cal.dirty = True
        elif key == ord('r'):
            cv2.destroyAllWindows()          # otherwise input() fights the window
            cal.rename_last()
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WINDOW, cal.on_mouse)
        elif key == ord('i'):
            cv2.destroyAllWindows()
            cal.set_inner_scale()
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WINDOW, cal.on_mouse)
        elif key == ord('s'):
            if not cal.spots:
                print("  nothing to save - click 4 corners first")
            else:
                cal.save()
                problems = validate(cal.spots, (h, w))
                for p in problems:
                    print(f"  warning: {p}")
        elif key == ord('l'):
            cal.reload()
        elif key == ord('g'):
            cal.show_grid = not cal.show_grid
        elif key == ord('v'):
            cal.show_inner = not cal.show_inner

    cv2.destroyAllWindows()
    if not reuse_still:
        cam.close()
    if cal.spots:
        problems = validate(cal.spots, (h, w))
        if problems:
            print("warnings:")
            for p in problems:
                print(f"  - {p}")
        print(f"final: {len(cal.spots)} spot(s) in {cal.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
