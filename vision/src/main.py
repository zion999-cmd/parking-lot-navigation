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
    p      print the current numbers
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
import os
import shutil
import sys
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alignment import Aligner                                    # noqa: E402
from calibration import CalibrationApp                           # noqa: E402
from camera import CameraError, open_camera                      # noqa: E402
from detector import DetectorParams                              # noqa: E402
from occupancy import (EMPTY, OCCUPIED, OccupancyDetector,        # noqa: E402
                       OccupancyParams, SpotResult)
from spot_config import Spot, SpotConfigError, load_spots         # noqa: E402
from spot_editor import SpotEditor                                # noqa: E402

WINDOW = "ESP32 Camera - parking occupancy"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

COL_BAY = (0, 255, 255)
COL_INNER = (0, 180, 0)
COL_EMPTY = (0, 220, 0)
COL_OCCUPIED = (0, 0, 255)


def annotate(frame: np.ndarray, spots: List[Spot], results: Dict[str, SpotResult],
             show_inner: bool, show_diff: bool) -> np.ndarray:
    canvas = frame.copy()

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


def status_line(spots: List[Spot], results: Dict[str, SpotResult]) -> str:
    parts = []
    for s in spots:
        r = results.get(s.id)
        parts.append(f"{s.id}:{r.changed_ratio:.3f}({r.state[0]})" if r else f"{s.id}:?")
    return " ".join(parts)


def capture_reference(frame: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"  reference captured -> {path}")
    print(f"  frame diagnostics: {OccupancyDetector.describe(frame)}")


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
        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            continue
        if key == ord('r'):
            # 'r' means "capture the empty reference" in main.py, but the editor binds it
            # to rename, so pressing r here renamed a bay instead of taking a reference -
            # which is how a bay ended up renamed from 162 to 196. The host app's meaning
            # wins inside its own window; renaming is 'R'.
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
    ap.add_argument("--changed-ratio", type=float, default=0.15,
                    help="occupancy threshold on changed_ratio (default 0.15)")
    ap.add_argument("--pixel-threshold", type=int, default=25)
    ap.add_argument("--blur", type=int, default=5)
    ap.add_argument("--headless", action="store_true", help="no window; print/CVS only")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="stop after N seconds (0 = run until Ctrl-C)")
    ap.add_argument("--csv", help="write per-frame records to this CSV")
    ap.add_argument("--print-every", type=float, default=0.0,
                    help="seconds between status lines (default 1 headless / 5 window)")
    ap.add_argument("--align", default="translation",
                    choices=("off", "translation", "euclidean", "affine"),
                    help="compensate camera drift by aligning each frame to the "
                         "reference frame (default translation)")
    ap.add_argument("--align-every", type=int, default=3,
                    help="re-estimate the alignment every N frames (default 3)")
    ap.add_argument("--align-min-corr", type=float, default=0.8,
                    help="accept an alignment only at or above this structure "
                         "correlation (default 0.8; measured 0.95-0.97 for a real "
                         "nudge, 0.54-0.64 for a failed one)")
    ap.add_argument("--save-debug", metavar="PNG",
                    help="save one annotated frame with this path and exit")
    args = ap.parse_args()

    params = OccupancyParams(blur_ksize=args.blur,
                            pixel_threshold=args.pixel_threshold,
                            changed_ratio_threshold=args.changed_ratio)
    detector = OccupancyDetector(params)

    # -- spots ---------------------------------------------------------------
    declared_size = None
    try:
        still = (cv2.imread(args.reference, cv2.IMREAD_COLOR)
                 if os.path.exists(args.reference) else None)
        declared_size = _declared_image_size(args.config)
        if still is not None:
            spots = load_spots(args.config, still.shape[:2], skip_bad=True)
        else:
            # No reference yet, so the frame geometry is unknown. Do NOT invent one:
            # preparing a polygon against a guessed shape is what produced
            # "spot 16: detection area is empty after clipping" (the guess was also
            # passed height-first). set_reference() prepares the spots with the real
            # frame once the reference has been captured.
            spots = load_spots(args.config)
    except SpotConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("  hint: the calibration and the frame it was drawn on must be the same "
              "size. See config/parking_spots.json -> image_size.", file=sys.stderr)
        return 2

    print(f"loaded {len(spots)} spot(s) from {args.config}: "
          f"{', '.join(s.id for s in spots)}")

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

    if not have_reference:
        msg = (f"no empty-bay reference yet ({args.reference}).\n"
               f"  With all bays empty, run the windowed mode and press 'r'.")
        if args.headless:
            print(f"error: {msg}", file=sys.stderr)
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
    size_warned = False
    frame = still_frame
    last_frame_at = started
    stall_warned = False

    try:
        while True:
            fresh = still_frame if use_still else cam.read()
            if fresh is not None:
                frame = fresh
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
                out = annotate(frame, spots, results, show_inner, show_diff)
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

            if not args.headless:
                canvas = annotate(frame, spots, results, show_inner, show_diff)
                if aligner is None:
                    align_text = "align off"
                elif align_result is None:
                    align_text = "align ..."
                elif align_result.ok:
                    align_text = (f"align dx={align_result.dx:+.0f} "
                                  f"dy={align_result.dy:+.0f} corr={align_result.correlation:.2f}")
                else:
                    align_text = f"align FAILED: view no longer matches the reference"
                src_fps = cam.stats.fps() if cam is not None else 0.0
                stall_note = f"  NO FRAMES {time.time() - last_frame_at:.0f}s" \
                    if time.time() - last_frame_at > 3 else ""
                header = (f"threshold {threshold:.2f}  pixel {detector.params.pixel_threshold}  "
                          f"ref {'OK' if have_reference else 'MISSING - press r'}  "
                          f"{align_text}  loop {frames / max(elapsed, 1e-6):.1f} fps"
                          f" (source {src_fps:.1f}){stall_note}")
                cv2.putText(canvas, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 0), 3)
                cv2.putText(canvas, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1)
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
                    get_frame = (lambda: still_frame) if use_still else cam.read
                    shape = run_edit_mode(get_frame, args.config, args.template,
                                          args.reference, frame.shape[:2], args.align)
                    if shape is None:
                        continue
                    try:
                        spots = load_spots(args.config, frame.shape[:2], skip_bad=True)
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
                        print(f"  {status_line(spots, results)}")
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
