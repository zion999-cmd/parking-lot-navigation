"""Parking vision PoC - live occupancy of fixed parking bays.

    python src/main.py                                   # window, live stream
    python src/main.py --headless --duration 20 --csv data/cases/case1_all_empty.csv

Window keys
-----------
    r      capture the empty reference from the current frame (writes
           data/reference/reference.jpg and prints frame diagnostics)
    g      toggle the inner detection polygons
    d      toggle the red "changed pixels" overlay
    + / -  raise / lower the changed_ratio threshold by 0.01
    [ / ]  raise / lower the per-pixel threshold by 5
    p      print the current numbers (* marks bays over the threshold)
    t      reference audit: why a bay is stuck OCCUPIED
    s      save an annotated snapshot to data/debug/
    e      edit mode: correct the calibration on the live view
           (same keys as src/calibration.py; s saves, q/ESC returns here)
    q/ESC  quit

Headless mode prints one line per second and, with --csv, records every frame as
    t_seconds, spot_id, changed_ratio, state
so the acceptance cases can be turned into a threshold instead of a guess
(see tests/fit_threshold.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import contract                                                # noqa: E402
import emit_map                                                # noqa: E402
import live_view                                               # noqa: E402
import vehicle                                                 # noqa: E402
from road_network import RoadNetwork                            # noqa: E402
from alignment import Aligner                                    # noqa: E402
from calibration import CalibrationApp                           # noqa: E402
from camera import CameraError, open_camera                      # noqa: E402
from detector import DetectorParams                              # noqa: E402
from occupancy import (EMPTY, OCCUPIED, OccupancyDetector,        # noqa: E402
                       OccupancyParams, SpotResult)
from spot_config import Spot, SpotConfigError, load_spots         # noqa: E402
import zoom
from spot_editor import SpotEditor                                # noqa: E402

WINDOW = "ESP32 Camera - parking occupancy"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

COL_BAY = (0, 255, 255)
COL_INNER = (0, 180, 0)
COL_EMPTY = (0, 220, 0)
COL_OCCUPIED = (0, 0, 255)


COL_CAR = (255, 120, 0)


def annotate(frame: np.ndarray, spots: List[Spot], results: Dict[str, SpotResult],
             show_inner: bool, show_diff: bool,
             pose: Optional[vehicle.VehiclePose] = None,
             location: Optional[dict] = None) -> np.ndarray:
    canvas = frame.copy()

    if pose is not None:
        # The hero car. The axis is drawn as a line with a small ring at each end, not as
        # arrows: colour and shape know the car's axis and not which end is the front, and two
        # arrowheads read as two competing directions rather than as "no direction".
        x, y, w, h = pose.bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COL_CAR, 2)
        length = max(w, h) * 0.7
        if (pose.heading_from_arrow or pose.elongation >= vehicle.AXIS_MIN_ELONGATION):
            # Only draw an axis when there is one. A line through a square is a claim the mask
            # cannot support, and on screen it reads as confidently as a real one.
            rad = math.radians(pose.heading)
            dx, dy = length * math.cos(rad) / 2, length * math.sin(rad) / 2
            cv2.line(canvas, (int(pose.x - dx), int(pose.y - dy)),
                     (int(pose.x + dx), int(pose.y + dy)), COL_CAR, 2)
        cv2.circle(canvas, (int(pose.x), int(pose.y)), 4, COL_CAR, -1)
        if pose.front_is_known and pose.front_heading is not None:
            # A filled dot at the end the car faces and a hollow one behind it. The picture
            # can say which way is forward now, but only because a person told us which end
            # of this car is the front - nothing in the pixels knows that.
            facing = math.radians(pose.front_heading)
            half = length / 2
            fx = (int(pose.x + half * math.cos(facing)),
                  int(pose.y + half * math.sin(facing)))
            rear = (int(pose.x - half * math.cos(facing)),
                    int(pose.y - half * math.sin(facing)))
            cv2.circle(canvas, fx, 7, COL_CAR, -1)      # filled = the front
            cv2.circle(canvas, rear, 7, COL_CAR, 2)     # hollow = the back
        elif (pose.heading_from_arrow or pose.elongation >= vehicle.AXIS_MIN_ELONGATION):
            # No front known: both ends get a ring, so the drawing does not claim a direction it
            # does not have. Which needs an axis to put them on: with no axis there are no ends
            # to mark, and dx/dy above were never computed - wrapping the axis line in the same
            # condition crashed here with UnboundLocalError the moment the car had no axis,
            # which is exactly the situation this code was added for.
            for p in ((int(pose.x - dx), int(pose.y - dy)),
                      (int(pose.x + dx), int(pose.y + dy))):
                cv2.circle(canvas, p, 5, COL_CAR, 2)

        where = ""
        if location:
            if location.get("kind") == "spot":
                where = f"  in {location['id']}"
            elif location.get("kind") == "road":
                where = f"  on road ({location['distance_to_road']:.0f}px)"
            elif location.get("kind") == "floor":
                where = f"  floor ({location['distance_to_road']:.0f}px from road)"
        # The numbers that say whether the reading can be trusted at all: an axis only means
        # something if the shape is elongated, and if the two independent estimates disagree
        # the blob is not car shaped, whatever the angle says.
        agree = min(abs(pose.heading - pose.heading_pca),
                    180 - abs(pose.heading - pose.heading_pca))
        if pose.front_is_known and pose.front_heading is not None:
            text = (f"car ({pose.x:.0f},{pose.y:.0f}) FRONT {pose.front_heading:.0f}deg"
                    f"  (axis {pose.heading:.0f}){where}")
        else:
            axis_txt = (f"{pose.heading:.0f}deg" if (pose.heading_from_arrow or pose.elongation >= vehicle.AXIS_MIN_ELONGATION)
                        else "unknown")
            text = (f"car ({pose.x:.0f},{pose.y:.0f}) axis {axis_txt}"
                    f"  front unknown{where}")
        ev = f"{pose.front_evidence:+.2f}" if pose.front_evidence is not None else "  -  "
        detail = (f"elon {pose.elongation:.2f}  pca {pose.heading_pca:.0f}deg  d{agree:.0f}deg"
                  f"  ends {pose.ends_grey:+.1f}  taper {pose.taper:+.2f}  ev {ev}"
                  f"  turn {pose.observed_turn:+.0f}°")
        if pose.elongation < 1.6:
            detail += "  NO CLEAR AXIS"
        if agree > 10:
            detail += "  SHAPE SUSPECT"
        if not pose.front_is_known:
            detail += "  front UNSET (taper too small)"
        cv2.putText(canvas, text, (max(x - 20, 6), max(y - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
        cv2.putText(canvas, text, (max(x - 20, 6), max(y - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_CAR, 1)
        cv2.putText(canvas, detail, (max(x - 20, 6), max(y + h + 16, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
        cv2.putText(canvas, detail, (max(x - 20, 6), max(y + h + 16, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_CAR, 1)

    if show_diff:
        for s in spots:
            res = results.get(s.id)
            if res is None or res.diff_mask is None:
                continue
            x0, y0, x1, y1 = s.roi
            roi = canvas[y0:y1, x0:x1]
            hot = res.diff_mask > 0
            if hot.any():
                tint = np.zeros_like(roi)
                tint[:] = COL_OCCUPIED
                roi[hot] = (roi[hot].astype(np.uint16) * 1 +
                            tint[hot].astype(np.uint16) * 1).astype(np.uint8)

    for s in spots:
        res = results.get(s.id)
        cv2.polylines(canvas, [np.round(s.polygon).astype(np.int32)], True, COL_BAY, 2)
        if show_inner and s.inner_points is not None:
            cv2.polylines(canvas, [np.round(s.inner_points).astype(np.int32)],
                          True, COL_INNER, 1)
        if res is None:
            continue
        colour = COL_OCCUPIED if res.state == OCCUPIED else COL_EMPTY
        x0, y0, x1, y1 = s.roi
        label = f"Spot {s.id}  {res.state}"
        detail = f"{res.changed_ratio:.3f}"
        cv2.putText(canvas, label, (x0, max(y0 - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3)
        cv2.putText(canvas, label, (x0, max(y0 - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, colour, 1)
        cv2.putText(canvas, detail, (x0, min(y1 + 16, canvas.shape[0] - 4)),
                    cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 0, 0), 3)
        cv2.putText(canvas, detail, (x0, min(y1 + 16, canvas.shape[0] - 4)),
                    cv2.FONT_HERSHEY_PLAIN, 1.1, colour, 1)
    return canvas


def _declared_image_size(path: str):
    """The ``image_size`` recorded in the config, as (height, width), or None."""
    try:
        with open(path, encoding="utf-8") as fh:
            size = json.load(fh).get("image_size")
    except Exception:
        return None
    if not size or len(size) != 2:
        return None
    return (int(size[0]), int(size[1]))


def status_line(spots: List[Spot], results: Dict[str, SpotResult],
                threshold: Optional[float] = None) -> str:
    parts = []
    for s in spots:
        r = results.get(s.id)
        if r is None:
            parts.append(f"{s.id}:?")
        else:
            # '*' marks the bays over the threshold, so the odd one out in a wall of
            # 0.000s is impossible to miss.
            parts.append(f"{s.id}:{r.changed_ratio:.3f}{'*' if r.state == OCCUPIED else ''}")
    head = f"threshold {threshold:.2f}  (* = occupied)  " if threshold is not None else ""
    return head + " ".join(parts)


def reference_audit(ref: np.ndarray, live: np.ndarray, spots: List[Spot],
                    detector: OccupancyDetector) -> str:
    """Say whether the reference frame is still usable, and if not, why.

    The first version of this got it badly wrong in a way worth keeping a note of. It compared
    the reference with the live frame outside the bays, saw 7% of pixels differ, and concluded
    "the reference is stale - re-capture it". But 41 of the 48 bays read exactly 0.000, which
    says the camera had not moved and the light had not changed at all: the 7% was cars parked
    on the roads, and roads are outside the bays. Following that advice would have baked the
    cars into the reference - the exact mistake this project had already spent an evening
    undoing.

    So the bays decide, not the outside pixels. If most bays are untouched the view and the
    light are fine by construction, whatever is happening elsewhere in the frame. The outside
    count is still reported, because it is how you notice a car on a road or something left in
    the scene, but it no longer decides anything on its own. It also ignores a margin around
    every bay, since the painted bay lines sit just outside their own detection areas and
    dominated the count with thin differences.
    """
    p = detector.params.normalized()
    a = detector._prep(ref)
    b = detector._prep(live)
    if a.shape != b.shape:
        print("  audit: the reference and the current frame differ in size - press r")
        return "size-mismatch"

    mask = cv2.threshold(cv2.absdiff(a, b), p.pixel_threshold, 255, cv2.THRESH_BINARY)[1]
    if p.open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.open_ksize,) * 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if p.close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.close_ksize,) * 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # Blank every bay *and a margin around it*: the bay's own painted line lies just outside
    # its detection area, and those thin differences used to make up most of the count.
    outside = mask.copy()
    for s in spots:
        cv2.fillPoly(outside, [np.round(s.polygon).astype(np.int32)], 0)
        cv2.polylines(outside, [np.round(s.polygon).astype(np.int32)], True, 0, 12)
    outside_pct = 100.0 * cv2.countNonZero(outside) / max(outside.size, 1)

    results = detector.evaluate(live, spots)
    ranked = sorted(results.values(), key=lambda r: -r.changed_ratio)
    over = [r for r in ranked if r.changed_ratio >= p.changed_ratio_threshold]
    quiet = [r for r in ranked if r.changed_ratio < 0.01]
    quiet_pct = 100.0 * len(quiet) / max(len(ranked), 1)

    verdict = "ok"
    print("  ---------- reference audit -----------------------------------------")
    print(f"  bays reading exactly 0.000 : {len(quiet)}/{len(ranked)} "
          f"({quiet_pct:.0f}%)   <- this is what decides")
    print(f"  bays over the threshold    : {len(over)}")
    print(f"  pixels changed outside the bays: {outside_pct:.2f}% (a car on a road shows up here)")
    for r in ranked[:6]:
        print(f"    {r.spot_id:>5}  {r.changed_ratio:.3f}  {r.state}")

    if quiet_pct >= 70.0:
        print("  verdict: the camera and the light are unchanged - the reference is still good.")
        if outside_pct >= 0.5:
            print(f"  {outside_pct:.1f}% of pixels outside the bays differ, which is what a car on a")
            print("  road looks like. Do NOT re-capture the reference while it is there: whatever")
            print("  is standing in the scene would become part of the reference.")
            print("  Fix: nothing, if that car is meant to be there. If it is not, remove it and")
            print("  run the audit again.")
            verdict = "ok-with-extra"
        else:
            print("  Nothing outside the bays changed either, so there is nothing to do.")
            verdict = "ok"
        if over:
            who = ", ".join(r.spot_id for r in over[:8])
            print(f"  The {len(over)} bay(s) over the threshold ({who}) differ from the reference")
            print("  while everything else matches, so those bays' contents really did change.")
    else:
        verdict = "stale"
        print(f"  verdict: only {quiet_pct:.0f}% of bays match the reference, so the view itself has")
        print("  moved or the light has changed: the reference is stale.")
        print("  Fix: realign the bays with 'e' if needed, then press r with every bay empty.")
    print("  --------------------------------------------------------------------")
    return verdict


def capture_reference(frame: np.ndarray, path: str) -> bool:
    """Write the empty-bay reference, atomically.

    cv2.imwrite truncates the file it is given and then writes into it, so any process
    reading that path at the wrong moment gets half a JPEG. That is not hypothetical: it
    happened while diagnosing this, and a torn reference frame makes every bay report a
    difference against garbage. Writing beside the target and moving it into place costs
    nothing and removes the window entirely - which matters twice over here, because
    tools that inspect the reference (this viewer in another terminal, tests, an
    experiment script) routinely run while a capture is in progress.

    imwrite also returns False on failure instead of raising, which used to be ignored
    and printed "captured" regardless.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = os.path.splitext(path)[0] + ".writing.jpg"   # the extension picks the encoder
    if not cv2.imwrite(tmp, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
        print(f"  ERROR: could not write {tmp}", file=sys.stderr)
        print("  the previous reference is untouched", file=sys.stderr)
        return False
    os.replace(tmp, path)

    print(f"  reference captured -> {path}")
    print("  every bay must be EMPTY in this frame: whatever is parked here now becomes")
    print("  a permanent 'occupied', because its absence is what the detector would see.")
    print(f"  frame diagnostics: {OccupancyDetector.describe(frame)}")
    print("  press t for a reference audit if any bay still reads occupied")
    return True


def vehicle_state(pose: Optional[vehicle.VehiclePose],
                  location: Optional[dict] = None) -> Optional[dict]:
    """The car's entry for the contract, or None when it cannot be seen.

    None rather than a stale position: the same rule that makes an unseen bay UNKNOWN. A
    navigator can be told "the car is not visible, look again"; it cannot recover from being
    told where the car was ten seconds ago.
    """
    if pose is None:
        return None
    out = pose.as_dict()
    if location:
        out["location"] = location
    return out


# How long without a frame before the contract admits it does not know anything.
STALE_AFTER_S = 2.0


def state_key(document: dict) -> str:
    """What makes the state worth rewriting: everything except when it was written.

    Compared so that a change can reach the page as soon as it happens instead of waiting for the
    next heartbeat - which is half the end-to-end lag the page used to have. The grain matters
    more than the comparison: a parked car's position wanders by tenths of a pixel and its axis by
    a fraction of a degree, and a bay's ratio jitters by more than it moves, so measuring to the
    pixel, the degree and the hundredth keeps a still scene still. All three are finer than
    anything a route can act on, and the heading is rounded the way the overlay prints it.
    """
    def canon(value, path: str = ""):
        if isinstance(value, dict):
            return {k: canon(v, f"{path}.{k}") for k, v in sorted(value.items())
                    if k != "timestamp"}
        if isinstance(value, list):
            return [canon(v, path) for v in value]
        if isinstance(value, float):
            if path.endswith((".x", ".y")):
                return round(value)                     # to the pixel
            if path.endswith(("heading", "front_heading")):
                return round(value)                     # to the degree
            return round(value, 2)
        return value
    return json.dumps(canon(document), sort_keys=True)


def write_state(args, results, spots, have_reference: bool, stall: float,
                last_written: float, vehicle_doc: Optional[dict] = None,
                last_key: Optional[str] = None
                ) -> Optional[Tuple[float, str]]:
    """Write the live half of the P0002 contract. Returns (when, key) if it wrote, else None.

    Written when the state has changed, and otherwise on the heartbeat. A change may go out after
    half the heartbeat interval rather than a whole one, so a car turning into a bay shows up in
    about a quarter of a second rather than up to a second later.

    Two situations produce UNKNOWN for every bay rather than EMPTY, and both matter: no
    reference frame means nothing can be compared at all, and no recent frames means the
    camera or the network has stopped. A navigator can be told to come back and look again.
    It cannot recover from being told a bay is free when nobody knows.
    """
    path = args.state_out
    if not path:
        return None
    now = time.time()

    reason = None
    if stall > STALE_AFTER_S:
        reason = f"no frames for {stall:.1f}s"
    elif not have_reference:
        reason = "no empty-bay reference captured"
    elif not results:
        reason = "nothing evaluated yet"

    if reason is not None:
        document = contract.unknown_state([s.id for s in spots], reason,
                                          vehicle=vehicle_doc)
    else:
        document = contract.build_state(results, vehicle=vehicle_doc)

    every = max(args.state_every, 0.0)
    key = state_key(document)
    # A change is worth a write after half the interval; an unchanged state waits for the whole
    # one, which is what keeps a file being written for a scene that is not moving at all.
    if now - last_written < (every / 2.0 if key != last_key else every):
        return None

    try:
        contract.write_json(path, document)
    except OSError as exc:
        print(f"  could not write {path}: {exc}", file=sys.stderr)
    return now, key


def bay_pitch_of(spots: List[Spot]) -> Optional[float]:
    """A bay's shorter side: the natural scale for "is this shift close to one bay?"""
    widths = []
    for sp in spots:
        pts = np.asarray(sp.polygon, dtype=np.float64)
        widths.append(min(np.linalg.norm(pts[1] - pts[0]),
                          np.linalg.norm(pts[3] - pts[0])))
    return float(np.median(widths)) if widths else None


def run_edit_mode(get_frame, config_path: str, template_path: Optional[str],
                  reference_path: Optional[str],
                  frame_shape, align_motion: str) -> Optional[int]:
    """Hand the window to the calibration editor until the user leaves it.

    spot_editor.SpotEditor holds the state and calibration.CalibrationApp adds the
    window, mouse and key handling, so every key behaves exactly as in
    src/calibration.py: drag a corner or a bay body, 2/4/6/8 nudge, Tab cycle, n add,
    d delete, g/G generate a row, a auto detect, z undo, A align, s save.

    Running it from here is the point: the misalignment is visible on screen, and the
    standalone tool has to be told which frame and which config you mean.

    Returns the number of spots after editing, or None if the editor never started.
    """
    # The editor writes the config with 's', so keep the version before this session.
    if os.path.exists(config_path):
        backup = config_path + ".bak"
        shutil.copyfile(config_path, backup)
        print(f"  backup: {backup}")

    if template_path is None:
        guess = os.path.join(ROOT, "data", "debug", "frame.jpg")
        template_path = guess if os.path.exists(guess) else None
    template = cv2.imread(template_path, cv2.IMREAD_COLOR) if template_path else None

    editor = SpotEditor(frame_shape)
    if not editor.load(config_path):
        print(f"  error: could not load {config_path}", file=sys.stderr)
        return None

    app = CalibrationApp(editor, config_path, template=template,
                         align_motion=align_motion)
    # act() reaches for this the moment 'a' (auto detect) is pressed; calibration.py
    # sets it in its own run().
    app.detector_params = DetectorParams()

    print("  ---------- edit mode ----------------------------------------------")
    print("  left drag a corner or a body to move a bay, right click snaps to an edge")
    print("  2/4/6/8 nudge 5 px    Tab cycle    n add    d delete    z undo    c clear")
    print("  g / G generate a row right / left from the selected bay")
    print("  a auto detect (an assistant - the human's confirmation is the truth)")
    print("  i inner scale    v inner    l line overlay    A align to the template")
    print("  m cycles what 2/4/6/8 move: this bay -> its row -> all bays")
    print("  s save to the config file        q / ESC back to detection")
    print("  r captures the empty reference, exactly as outside edit mode")
    print("  NOTE: renaming a bay is 'R' here, because 'r' keeps its main.py meaning")
    print("  ------------------------------------------------------------------")

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, app.on_mouse)

    frame = None
    # The window's own key hint comes from the editor, which standalone binds 'r' to rename.
    # Here 'r' captures the reference frame, so the window must say 'R' instead - a window that
    # tells you to press a key that does something else is how a rename "did not work".
    # No rename_hint override: 'r' renames here, which is what the editor's hint already says.
    # (The reference capture moved to 'R' so that both agree on 'r'.)
    print("  edit mode:  r = rename a bay (type the new id in this terminal)")
    print("              R = capture the empty reference   s = save now   q = save and leave")

    while True:
        fresh = get_frame()
        if fresh is not None:
            frame = fresh

        if frame is None:
            if (cv2.waitKey(30) & 0xFF) == 27:
                break
            continue
        app._frame = frame

        # Terminal prompts must run with the window gone; same pattern as calibration.py.
        while app._pending_prompt is not None:
            cv2.destroyAllWindows()
            app.prompt()
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WINDOW, app.on_mouse)

        cv2.imshow(WINDOW, app.render(frame))
        # Same 5 ms as calibration.py's loop, and for the same measured reason: it is the event
        # pump, not the drawing, that decides how closely a dragged bay follows the pointer.
        raw = cv2.waitKey(5)
        # PARKING_LOG_KEYS=1 names every code that arrives. Same as calibration.py's loop.
        if raw != -1 and app.log_keys:
            print(f"  key code {raw}", file=sys.stderr)
        # Arrow keys are numbers, not characters: the raw code is offered to the window before
        # `& 0xFF` keeps only its low byte and throws the direction away. Same order as
        # calibration.py's loop, so the two windows answer the arrow keys the same way.
        if raw != -1 and app.act_code(raw):
            continue
        key = raw & 0xFF
        if key == 255:
            continue
        if key == ord('R'):
            # Uppercase R captures the empty reference *here*, so that lowercase r can mean in
            # this window exactly what the editor's own hint says it means: rename. The two used
            # to disagree, and whichever way round they were, the window told one of the two
            # people to press a key that did the other thing.
            if reference_path:
                capture_reference(frame, reference_path)
                editor._say("empty reference captured from this frame")
            continue
        if not app.act(chr(key)):
            break

    cv2.destroyAllWindows()
    # Same behaviour as the standalone tool: leaving with unsaved edits writes them.
    # The previous version is already sitting next to it as <config>.bak.
    if editor.dirty:
        if editor.save(config_path):
            print(f"  saved {config_path}")
            print(f"  previous version kept at {config_path}.bak")
        else:
            print(f"  error: could not write {config_path}", file=sys.stderr)
    else:
        print("  config unchanged")
    # q and ESC both save on the way out, but this is a second net: the caller reloads the config
    # from disk the moment we return, so an edit that never reached the file would vanish.
    if app.ed.dirty or app.road.dirty:
        app._save_all()

    return len(editor.spots)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fixed-ROI parking occupancy (PoC).")
    ap.add_argument("--source", default="http://192.168.3.118/stream")
    ap.add_argument("--backend", default="auto", choices=("auto", "opencv", "mjpeg"))
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "parking_spots.json"))
    ap.add_argument("--reference", default=os.path.join(ROOT, "data", "reference", "reference.jpg"))
    ap.add_argument("--template", metavar="IMAGE",
                    help="the frame the bays were drawn on, used by 'A' in edit mode; "
                         "defaults to data/debug/frame.jpg when that file exists")
    ap.add_argument("--no-reference", action="store_true",
                    help="start without a reference (capture one with 'r')")
    # Defaults come from OccupancyParams, not from copies of its numbers. They used to be copies -
    # 0.15 here against 0.15 there - and raising the one in occupancy.py had no effect at all
    # because this default overrode it: measured, the service still called bays at changed_ratio
    # 0.284 occupied with a threshold of 0.40 on the other file. One source of truth, or the change
    # is a rumour.
    _occ = OccupancyParams()
    ap.add_argument("--changed-ratio", type=float, default=_occ.changed_ratio_threshold,
                    help=f"occupancy threshold on changed_ratio "
                         f"(default {_occ.changed_ratio_threshold:g}, from OccupancyParams)")
    ap.add_argument("--pixel-threshold", type=int, default=_occ.pixel_threshold)
    ap.add_argument("--blur", type=int, default=_occ.blur_ksize)
    ap.add_argument("--headless", action="store_true", help="no window; print/CVS only")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="stop after N seconds (0 = run until Ctrl-C)")
    ap.add_argument("--csv", help="write per-frame records to this CSV")
    ap.add_argument("--print-every", type=float, default=0.0,
                    help="seconds between status lines (default 1 headless / 5 window)")
    ap.add_argument("--align", default="off",
                    choices=("off", "translation", "euclidean", "affine"),
                    help="compensate camera drift by warping each frame onto the reference "
                         "frame. Off by default and that is deliberate: measured on this "
                         "board, the reference frame and a live frame correlate at 0.61 "
                         "once cars are parked, below the 0.8 gate, because cars change "
                         "the scene's structure. The camera is bolted to the sandbox "
                         "(two live frames correlate at 0.996), so there is nothing to "
                         "compensate; if it is ever knocked, re-align the bays with 'e' "
                         "in the viewer instead of trusting this")
    ap.add_argument("--align-every", type=int, default=3,
                    help="re-estimate the alignment every N frames (default 3)")
    ap.add_argument("--align-min-corr", type=float, default=0.8,
                    help="accept an alignment only at or above this structure "
                         "correlation (default 0.8; measured 0.95-0.97 for a real "
                         "nudge, 0.54-0.64 for a failed one)")
    ap.add_argument("--no-vehicle", action="store_true",
                    help="do not look for the hero car (the red one); it is located by "
                         "colour and shape, see src/vehicle.py")
    ap.add_argument("--zoom", type=float, default=1.0, metavar="X",
                    help="digital zoom: keep the central 1/X of the frame and scale it "
                         "back to the same size, so the car is X times bigger (default 1.0 "
                         "= off). Bays and roads are moved into the zoomed frame "
                         "automatically, so the calibration stays valid. Costs sharpness, "
                         "not correctness: the camera has already scaled to 800x600.")
    ap.add_argument("--vehicle-min-area", type=int, default=vehicle.DEFAULT_MIN_AREA,
                    help="ignore red blobs smaller than this (default %d)"
                         % vehicle.DEFAULT_MIN_AREA)
    ap.add_argument("--state-out", metavar="JSON",
                    default=os.path.join(ROOT, "out", "parking_state.json"),
                    help="write the live half of the P0002 contract here: every bay's "
                         "state, plus the vehicle when it can be localised (default "
                         "out/parking_state.json)")
    ap.add_argument("--window", action="store_true",
                    help="show the cv2 window even when serving the board. Serving implies no "
                         "window: the board is the window, and the one here is only wanted to "
                         "edit bays and roads, which is a job for src/calibration.py")
    ap.add_argument("--capture-reference", action="store_true",
                    help="capture the empty-bay reference from the next frame, then exit. Takes "
                         "the place of pressing r in the window, for runs that have none")
    ap.add_argument("--map-out", default=None,
                    help="where to emit the map. Default: out/parking_map.json for the real "
                         "config, and beside any other config, so a copy cannot overwrite it")
    ap.add_argument("--no-emit-map", dest="emit_map", action="store_false",
                    help="do not re-emit out/parking_map.json at startup. It is emitted by default "
                         "so that the pages, which click the emitted map, cannot end up describing "
                         "a different car park from the picture drawn here")
    ap.add_argument("--raw-port", type=int, default=0,
                    help="also serve the UNANNOTATED picture on this port, so browsers and the "
                         "labeller read from this PC instead of the board. The board has one frame "
                         "queue and therefore allows one viewer; this is how N people look without "
                         "splitting its frames.")
    ap.add_argument("--annotated-port", type=int, default=0,
                    help="serve the analysed picture as an MJPEG stream on this port (0 = off). "
                         "The board page shows this instead of the camera's own /stream, so the "
                         "camera does not split its frames between the analysis and a tab.")
    ap.add_argument("--annotated-quality", type=int, default=80,
                    help="JPEG quality for that stream")
    ap.add_argument("--state-every", type=float, default=0.5,
                    help="seconds between contract writes (default 0.5)")
    ap.add_argument("--save-debug", metavar="PNG",
                    help="save one annotated frame with this path and exit")
    args = ap.parse_args()

    params = OccupancyParams(blur_ksize=args.blur,
                            pixel_threshold=args.pixel_threshold,
                            changed_ratio_threshold=args.changed_ratio)
    detector = OccupancyDetector(params)

    # -- spots ---------------------------------------------------------------
    declared_size = None
    # Every bay and road load goes through these two, so no code path can forget the zoom
    # and end up with polygons in the unzoomed frame while the picture is zoomed - which
    # would read as "every bay is occupied" rather than as a mistake.
    zoom_shape = _declared_image_size(args.config) or (600, 800)
    
    def _spots(path, shape=None, skip_bad=False):
        # No coordinate transform, on purpose.
        #
        # The zoom undoes a lens change: a 5640 sees wider than the 3660 it replaced, so the scene
        # came out smaller, and zooming 1.2x puts the framing back. The zoomed frame is therefore
        # the frame the bays were calibrated on, and they are already in the right place in it.
        # Transforming them - what the first version did - moved every polygon off the object it
        # was drawn around, which looked like "the detection boxes got bigger".
        #
        # If a future change really alters the framing instead of restoring it, this is the single
        # place to add the transform - and then the bays have to be re-calibrated, not rescaled.
        return load_spots(path, shape, skip_bad=skip_bad)
    
    def _roads(path):
        return RoadNetwork.load(path)
    
    try:
        still = (cv2.imread(args.reference, cv2.IMREAD_COLOR)
                 if os.path.exists(args.reference) else None)
        declared_size = _declared_image_size(args.config)
        if still is not None:
            spots = _spots(args.config, still.shape[:2], skip_bad=True)
        else:
            # No reference yet, so the frame geometry is unknown. Do NOT invent one:
            # preparing a polygon against a guessed shape is what produced
            # "spot 16: detection area is empty after clipping" (the guess was also
            # passed height-first). set_reference() prepares the spots with the real
            # frame once the reference has been captured.
            spots = _spots(args.config)
    except SpotConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("  hint: the calibration and the frame it was drawn on must be the same "
              "size. See config/parking_spots.json -> image_size.", file=sys.stderr)
        return 2

    print(f"loaded {len(spots)} spot(s) from {args.config}: "
          f"{', '.join(s.id for s in spots)}")

    # The road network, for answering "which bay, or on which road" about the car. Only the
    # geometry is needed here, so it does not matter that the frame size is not known yet.
    road_net = _roads(args.config) if os.path.exists(args.config) else None

    # The picture on screen is drawn from the config; the board page clicks the *emitted* map.
    # Those are two files, and when they disagree the page highlights a bay that is not where the
    # picture shows one - which is exactly what a stale map with two bays in it did. Emitting here
    # means whatever this process draws, the page can click.
    if args.emit_map:
        em = emit_map.emit(args.config, args.map_out, quiet=True)
        if em in (0, 1):
            # Where it actually went: --map-out defaults to None and the path is decided by
            # emit_map.out_for (the real config emits to out/, a copy emits beside itself).
            print(f"emitted   {emit_map.out_for(args.config)}")
        else:
            print(f"warning: could not emit the map from {args.config}; the page may be out of "
                  f"step with this picture", file=sys.stderr)

    # -- reference -----------------------------------------------------------
    have_reference = False
    detector_ref_frame = None
    if not args.no_reference and os.path.exists(args.reference):
        ref = cv2.imread(args.reference, cv2.IMREAD_COLOR)
        if ref is not None:
            detector_ref_frame = ref
            detector.set_reference(ref, spots)
            have_reference = True
            print(f"reference: {args.reference} ({ref.shape[1]}x{ref.shape[0]})")
            print(f"  frame diagnostics: {OccupancyDetector.describe(ref)}")
    aligner = None
    align_result = None
    align_stats = {"ok": 0, "failed": 0}
    if have_reference and args.align != "off":
        # A bay's width is the natural scale for the "is this shift suspiciously
        # close to one bay" check, so it comes from the calibration itself.
        pitch = bay_pitch_of(spots)
        aligner = Aligner(detector_ref_frame, motion=args.align,
                          min_correlation=args.align_min_corr, bay_pitch=pitch)
        print(f"alignment: on, model {args.align}, working at "
              f"{aligner._work_size[0]}px, bay pitch ~{pitch:.0f}px, "
              f"min correlation {args.align_min_corr}")

    if not have_reference and not args.capture_reference:
        msg = (f"no empty-bay reference yet ({args.reference}).\n"
               f"  With all bays empty, run the windowed mode and press 'r'.")
        if args.headless:
            print(f"error: {msg}", file=sys.stderr)
            # Still emit a well formed state before giving up. The contract says the state
            # document always exists and always says one of EMPTY/OCCUPIED/UNKNOWN, so a
            # consumer never has to cope with a missing file - and "no reference" is
            # exactly the case that must not be reported as EMPTY.
            write_state(args, {}, spots, False, 0.0, 0.0, None, None)
            return 2
        print(f"WARNING: {msg}")

    # -- camera --------------------------------------------------------------
    # A still image is a legitimate source (useful for offline checks and for
    # calibrating from a saved frame), and it must not be opened as a video.
    STILL_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".pgm", ".tif", ".tiff")
    use_still = args.source.lower().endswith(STILL_EXT)
    cam = None
    still_frame: Optional[np.ndarray] = None
    if use_still:
        still_frame = cv2.imread(args.source, cv2.IMREAD_COLOR)
        if still_frame is None:
            print(f"error: could not read {args.source}", file=sys.stderr)
            return 2
        print(f"source: still image {args.source} ({still_frame.shape[1]}x{still_frame.shape[0]})")
    else:
        try:
            cam = open_camera(args.source, backend=args.backend)
        except CameraError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"source: {args.source} (backend {cam.name})")

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    csv_fh = open(args.csv, "w", newline="", encoding="utf-8") if args.csv else None
    csv_w = csv.writer(csv_fh) if csv_fh else None
    if csv_w:
        csv_w.writerow(["t_seconds", "spot_id", "changed_ratio", "state",
                        "align_ok", "align_dx", "align_dy", "align_corr"])

    threshold = params.changed_ratio_threshold
    show_inner = True
    show_diff = True
    started = time.time()
    last_print = 0.0
    print_every = args.print_every or (1.0 if args.headless else 5.0)
    frames = 0
    sums = {s.id: [0.0, 0, 0] for s in spots}      # sum, occupied frames, total
    snapshot_done = False
    audit_done = False
    size_warned = False
    frame = zoom.zoom_frame(still_frame, args.zoom)
    last_frame_at = started
    stall_warned = False
    state_written = [0.0, None]      # when it was written, and what was written
    config_mtime = None              # so an edit made in another window is noticed
    look_for_car = not args.no_vehicle
    show_car = True
    live = None
    raw_view = None
    if args.raw_port:
        raw_view = live_view.LiveView(args.raw_port, quality=args.annotated_quality)
        print(f"raw stream   : {raw_view.url}   <- 原始画面（无标注），给标注工具/浏览器用")
    if args.annotated_port:
        live = live_view.LiveView(args.annotated_port, quality=args.annotated_quality)
        print(f"analysed stream: {live.url}")
        print("  board page: http://localhost:8010/web/board.html")
        # Serving the board means the picture is already on screen somewhere else.
        if not args.window:
            args.headless = True
            print("  window: off.  编辑车位/道路请另开一个终端跑：")
            print(f"      {sys.executable} src/calibration.py --frame data/reference/reference.jpg")
            print("  （要这个进程自己开窗，加 --window）")
        live.start()
    if raw_view is not None:
        raw_view.start()
    # Smooths the readings: the car does not turn instantly, and a single frame's shape is
    # noisy enough to make the front appear and disappear. See vehicle.VehicleTracker.
    car_tracker = vehicle.VehicleTracker()

    try:
        while True:
            if frame is not None and args.capture_reference:
                ok = capture_reference(frame, args.reference)
                if ok:
                    print(f"captured the empty reference -> {args.reference}")
                    print("  (all bays must have been empty, or the reference contains a car)")
                else:
                    print("could not capture the reference", file=sys.stderr)
                return 0 if ok else 2

            if frame is not None:
                # The map may be edited in another window: the editor runs off a still frame and does not
                # need this process to stop. When the config changes, pick it up - the picture, the ROIs
                # and the page's map must all describe the same car park.
                try:
                    mtime = os.path.getmtime(args.config)
                except OSError:
                    mtime = None
                if config_mtime is None:
                    config_mtime = mtime
                elif mtime is not None and mtime != config_mtime:
                    config_mtime = mtime
                    try:
                        spots = _spots(args.config, frame.shape[:2], skip_bad=True)
                        # Anything keyed by bay id has to be rebuilt with the new ones, or the
                        # next frame's totals raise KeyError on a bay that was just renamed.
                        sums = {s.id: [0.0, 0, 0] for s in spots}
                        road_net = (_roads(args.config)
                                    if os.path.exists(args.config) else None)
                        if detector_ref_frame is not None:
                            detector.set_reference(detector_ref_frame, spots)
                        if args.emit_map:
                            emit_map.emit(args.config, args.map_out, quiet=True)
                        print(f"  reloaded {args.config}: {len(spots)} bay(s), and re-emitted "
                              f"{emit_map.out_for(args.config)}")
                    except Exception as exc:                             # noqa: BLE001
                        print(f"  could not reload {args.config}: {exc}", file=sys.stderr)

            fresh = still_frame if use_still else cam.read()
            if fresh is not None:
                frame = zoom.zoom_frame(fresh, args.zoom)
                last_frame_at = time.time()
                stall_warned = False

            if frame is None:
                # Nothing to show yet. Keep the window alive and keep polling the
                # keyboard: this used to `continue` straight past cv2.waitKey, so a
                # slow stream froze the window and swallowed every key press.
                if not args.headless and (cv2.waitKey(30) & 0xFF) in (ord('q'), 27):
                    break
                if time.time() - started > 20:
                    print("error: no frames at all", file=sys.stderr)
                    break
                if args.headless:
                    time.sleep(0.01)
                continue

            frames += 1
            elapsed = time.time() - started

            # A drought is measured from the last frame that actually arrived, not
            # from process start. Measuring from process start made the viewer exit by
            # itself: past the 20 s mark the very next read() that returned no complete
            # frame yet - which happens several times per frame - ended the run. In
            # windowed mode it now says so and carries on.
            stall = time.time() - last_frame_at
            if stall > 20.0:
                if args.headless:
                    print(f"error: no frames for {stall:.0f} s", file=sys.stderr)
                    break
                if not stall_warned:
                    print(f"WARNING: no new frame for {stall:.0f} s - the camera or Wi-Fi "
                          f"is not delivering. Still listening; press q to quit.")
                    stall_warned = True

            if declared_size is not None and frame.shape[:2] != declared_size \
                    and not size_warned:
                print(f"WARNING: the frame is {frame.shape[1]}x{frame.shape[0]} but the "
                      f"calibration was drawn on {declared_size[1]}x{declared_size[0]}. "
                      f"The bays will not line up - re-calibrate, or point --reference at "
                      f"the frame they were drawn on.")
                size_warned = True

            # Align the frame to the reference frame the bays were drawn on, so a
            # camera nudge does not move every ROI off its bay. The ROIs and the
            # reference ROIs never change; only the incoming frame is moved.
            eval_frame = frame
            if aligner is not None and (align_result is None or
                                        frames % max(args.align_every, 1) == 0):
                align_result = aligner.estimate(frame)
                if align_result.ok:
                    align_stats["ok"] += 1
                else:
                    align_stats["failed"] += 1
            if aligner is not None and align_result is not None and align_result.ok:
                eval_frame = aligner.warp_frame(frame, align_result.matrix)

            if have_reference:
                results = detector.evaluate(eval_frame, spots, want_debug=True)
            else:
                results = {}

            # The hero car: found by colour and shape, with no marker and no reference frame.
            pose = None
            location = None
            if look_for_car:
                pose = car_tracker.update(
                    vehicle.locate(frame, min_area=args.vehicle_min_area))
                if pose is not None and any((spots, road_net)):
                    location = vehicle.location_of(pose, spots, road_net)

            wrote = write_state(args, results, spots, have_reference, stall,
                                state_written[0],
                                vehicle_state(pose, location) if pose else None,
                                state_written[1])
            if wrote is not None:
                state_written[0], state_written[1] = wrote

            for sid, r in results.items():
                sums[sid][0] += r.changed_ratio
                sums[sid][1] += 1 if r.state == OCCUPIED else 0
                sums[sid][2] += 1
            if csv_w:
                a_ok = 1 if (align_result is not None and align_result.ok) else 0
                a_dx = align_result.dx if align_result else 0.0
                a_dy = align_result.dy if align_result else 0.0
                a_cc = align_result.correlation if align_result else 0.0
                for sid, r in results.items():
                    csv_w.writerow([f"{elapsed:.3f}", sid, f"{r.changed_ratio:.5f}",
                                    r.state, a_ok, f"{a_dx:.2f}", f"{a_dy:.2f}",
                                    f"{a_cc:.4f}"])

            if args.save_debug and not snapshot_done:
                os.makedirs(os.path.dirname(os.path.abspath(args.save_debug)), exist_ok=True)
                out = annotate(frame, spots, results, show_inner, show_diff,
                               pose if show_car else None, location)
                cv2.imwrite(args.save_debug, out)
                print(f"saved annotated snapshot -> {args.save_debug}")
                if have_reference:
                    print(f"  {status_line(spots, results)}")
                snapshot_done = True
                break            # --save-debug is a one-shot diagnostic

            if elapsed - last_print >= print_every:
                last_print = elapsed
                if have_reference:
                    fps = frames / max(elapsed, 1e-6)
                    if align_result is None:
                        a = "align=off"
                    elif align_result.ok:
                        a = (f"align={align_result.dx:+.0f},{align_result.dy:+.0f}"
                             f"@{align_result.correlation:.2f}")
                    else:
                        a = f"align=FAILED"
                    src = f" src={cam.stats.fps():4.1f}" if cam is not None else ""
                    print(f"t={elapsed:5.1f}s fps={fps:4.1f}{src} thr={threshold:.2f} {a} "
                          f"{status_line(spots, results)}")
                else:
                    print(f"t={elapsed:5.1f}s  waiting for a reference (press 'r')")

            if not args.headless or live is not None:
                canvas = annotate(frame, spots, results, show_inner, show_diff,
                                  pose if show_car else None, location)
                if live is not None:
                    live.publish(canvas)      # before the HUD: the board writes its own labels
                    if raw_view is not None:
                        raw_view.publish(frame)   # unannotated: what the labeller traces
            if not args.headless:
                if aligner is None:
                    align_text = "align off"
                elif align_result is None:
                    align_text = "align ..."
                elif align_result.ok:
                    align_text = (f"align dx={align_result.dx:+.0f} "
                                  f"dy={align_result.dy:+.0f} corr={align_result.correlation:.2f}")
                else:
                    align_text = ("align ignored: the scene itself differs from the "
                                  "reference (cars in the bays change the structure)")
                src_fps = cam.stats.fps() if cam is not None else 0.0
                stall_note = f"  NO FRAMES {time.time() - last_frame_at:.0f}s" \
                    if time.time() - last_frame_at > 3 else ""
                header = (f"threshold {threshold:.2f}  pixel {detector.params.pixel_threshold}  "
                          f"ref {'OK' if have_reference else 'MISSING - press r'}  "
                          f"{align_text}  loop {frames / max(elapsed, 1e-6):.1f} fps"
                          f" (source {src_fps:.1f}){stall_note}")
                if not look_for_car:
                    car_text = "off"
                elif pose is None:
                    car_text = "not visible"
                else:
                    car_text = f"({pose.x:.0f},{pose.y:.0f}) axis {pose.heading:.0f}deg"
                    if location:
                        car_text += f" {location['kind']}" + (f" {location['id']}"
                                                              if location.get("id") else "")
                header += f"   car {car_text}"
                cv2.putText(canvas, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 0), 3)
                cv2.putText(canvas, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1)
                if not audit_done and have_reference and detector_ref_frame is not None:
                    # Once, on the first real frame. Turns "why is that bay stuck?" from a
                    # mystery into an answer, for the cost of one diff.
                    audit_done = True
                    reference_audit(detector_ref_frame, frame, spots, detector)

                cv2.imshow(WINDOW, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                elif key == ord('r'):
                    capture_reference(frame, args.reference)
                    detector.set_reference(frame, spots)
                    have_reference = True
                elif key == ord('e'):
                    # Correct the calibration without leaving the viewer: the offset is
                    # plainly visible here, which is the whole point.
                    # The editor and the calibration app take their frames from here, so the
                    # zoom has to be applied at this one point - otherwise edit mode shows the
                    # unzoomed picture and the bays, which are drawn in the zoomed frame, no
                    # longer sit on the objects they were calibrated on.
                    get_frame = (lambda: zoom.zoom_frame(still_frame, args.zoom)) if use_still \
                        else (lambda: zoom.zoom_frame(cam.read(), args.zoom))
                    shape = run_edit_mode(get_frame, args.config, args.template,
                                          args.reference, frame.shape[:2], args.align)
                    if shape is None:
                        continue
                    try:
                        spots = _spots(args.config, frame.shape[:2], skip_bad=True)
                    except SpotConfigError as exc:
                        print(f"  error reloading {args.config}: {exc}", file=sys.stderr)
                        continue
                    # Not gated on the earlier flag: a reference can be captured from
                    # inside edit mode with 'r'.
                    ref_img = (cv2.imread(args.reference, cv2.IMREAD_COLOR)
                               if os.path.exists(args.reference) else None)
                    if ref_img is not None:
                        detector_ref_frame = ref_img
                        detector.set_reference(ref_img, spots)
                        have_reference = True
                        if args.align != "off":
                            aligner = Aligner(detector_ref_frame, motion=args.align,
                                              min_correlation=args.align_min_corr,
                                              bay_pitch=bay_pitch_of(spots))
                    print(f"  edit mode finished: {len(spots)} spot(s) now in use")
                    if have_reference:
                        print("  if the camera itself moved, press r to capture a new "
                              "empty reference - the current one is from the old view")
                elif key == ord('g'):
                    show_inner = not show_inner
                elif key == ord('c'):
                    show_car = not show_car
                elif key == ord('d'):
                    show_diff = not show_diff
                elif key in (ord('+'), ord('=')):
                    threshold = min(threshold + 0.01, 1.0)
                    detector.params.changed_ratio_threshold = threshold
                elif key == ord('-'):
                    threshold = max(threshold - 0.01, 0.0)
                    detector.params.changed_ratio_threshold = threshold
                elif key == ord(']'):
                    detector.params.pixel_threshold = min(detector.params.pixel_threshold + 5, 255)
                    if have_reference:
                        detector.set_reference(cv2.imread(args.reference), spots)
                elif key == ord('['):
                    detector.params.pixel_threshold = max(detector.params.pixel_threshold - 5, 1)
                    if have_reference:
                        detector.set_reference(cv2.imread(args.reference), spots)
                elif key == ord('p'):
                    if have_reference:
                        print(f"  {status_line(spots, results, detector.params.changed_ratio_threshold)}")
                elif key == ord('t'):
                    if have_reference and detector_ref_frame is not None:
                        reference_audit(detector_ref_frame, frame, spots, detector)
                    else:
                        print("  no reference to audit yet - press r with every bay empty")
                elif key == ord('s'):
                    name = os.path.join(ROOT, "data", "debug",
                                        f"snapshot_{int(time.time())}.png")
                    os.makedirs(os.path.dirname(name), exist_ok=True)
                    cv2.imwrite(name, canvas)
                    print(f"  saved {name}")
            else:
                if args.duration and elapsed >= args.duration:
                    break
                if use_still:
                    break            # nothing new will arrive from a still image
                time.sleep(0.001)

            if args.duration and elapsed >= args.duration and (args.headless or snapshot_done):
                break
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if live is not None:
            if raw_view is not None:
                raw_view.stop()
            live.stop()
        if csv_fh:
            csv_fh.close()
        if cam is not None:
            cam.close()
        if not args.headless:
            cv2.destroyAllWindows()

    # -- summary -------------------------------------------------------------
    elapsed = time.time() - started
    print(f"\nran {elapsed:.1f}s, {frames} frames ({frames / max(elapsed, 1e-6):.1f} fps)")
    if cam is not None:
        print(f"stream stats: {cam.stats.summary()}")
        if cam.stats.fps() < 4.0:
            print("  NOTE: the source itself is slow. The frame rate above is what the "
                  "camera/Wi-Fi delivered, not a limit of this program.")
    if aligner is not None:
        print(f"alignment: {align_stats['ok']} ok, {align_stats['failed']} failed")
        if align_stats["failed"]:
            print("  WARNING: alignment failed on some frames - the ROIs may not sit on "
                  "the bays.")
            print("    The camera has moved further than a pan/tilt/zoom can explain "
                  "(translation/rotation/affine all scored below the gate), or the view "
                  "itself changed.")
            print("    Check config/parking_spots.json -> image_size against the live "
                  "frame, then re-capture the base frame with 'r' from the current view "
                  "and confirm the bays still land on the bays.")
        if align_result is not None:
            print(f"  last: {align_result.describe()}")

    if have_reference:
        print(f"per-spot changed_ratio over this run:")
        for s in spots:
            total, occ, n = sums[s.id]
            if n:
                print(f"  spot {s.id}: mean={total / n:.4f}  "
                      f"occupied {occ}/{n} frames ({100.0 * occ / n:.1f}%)  "
                      f"final={results[s.id].state}")
        print(f"threshold used: {threshold:.3f} "
              f"(pixel_threshold={detector.params.pixel_threshold})")
    if args.csv:
        print(f"records written to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
