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
    q/ESC  quit

Headless mode prints one line per second and, with --csv, records every frame as
    t_seconds, spot_id, changed_ratio, state
so the acceptance cases can be turned into a threshold instead of a guess
(see tests/fit_threshold.py).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera import CameraError, open_camera                      # noqa: E402
from occupancy import (EMPTY, OCCUPIED, OccupancyDetector,        # noqa: E402
                       OccupancyParams, SpotResult)
from spot_config import Spot, SpotConfigError, load_spots         # noqa: E402

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


def main() -> int:
    ap = argparse.ArgumentParser(description="Fixed-ROI parking occupancy (PoC).")
    ap.add_argument("--source", default="http://192.168.3.118/stream")
    ap.add_argument("--backend", default="auto", choices=("auto", "opencv", "mjpeg"))
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "parking_spots.json"))
    ap.add_argument("--reference", default=os.path.join(ROOT, "data", "reference", "reference.jpg"))
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
    ap.add_argument("--save-debug", metavar="PNG",
                    help="save one annotated frame with this path and exit")
    args = ap.parse_args()

    params = OccupancyParams(blur_ksize=args.blur,
                            pixel_threshold=args.pixel_threshold,
                            changed_ratio_threshold=args.changed_ratio)
    detector = OccupancyDetector(params)

    # -- spots ---------------------------------------------------------------
    try:
        still = cv2.imread(args.reference, cv2.IMREAD_COLOR) if os.path.exists(args.reference) else None
        if still is not None:
            spots = load_spots(args.config, still.shape[:2])
        else:
            spots = load_spots(args.config)
            for s in spots:                       # prepare from the polygon alone
                s.prepare(tuple(int(v) for v in np.ceil(s.polygon.max(axis=0)).astype(int) * 2 + 8))
    except SpotConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"loaded {len(spots)} spot(s) from {args.config}: "
          f"{', '.join(s.id for s in spots)}")

    # -- reference -----------------------------------------------------------
    have_reference = False
    if not args.no_reference and os.path.exists(args.reference):
        ref = cv2.imread(args.reference, cv2.IMREAD_COLOR)
        if ref is not None:
            detector.set_reference(ref, spots)
            have_reference = True
            print(f"reference: {args.reference} ({ref.shape[1]}x{ref.shape[0]})")
            print(f"  frame diagnostics: {OccupancyDetector.describe(ref)}")
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
        csv_w.writerow(["t_seconds", "spot_id", "changed_ratio", "state"])

    threshold = params.changed_ratio_threshold
    show_inner = True
    show_diff = True
    started = time.time()
    last_print = 0.0
    print_every = args.print_every or (1.0 if args.headless else 5.0)
    frames = 0
    sums = {s.id: [0.0, 0, 0] for s in spots}      # sum, occupied frames, total
    snapshot_done = False

    try:
        while True:
            if use_still:
                frame = still_frame
            else:
                frame = cam.read()
                if frame is None:
                    if time.time() - started > 20:
                        print("error: no frames for 20 s", file=sys.stderr)
                        break
                    continue
            frames += 1
            elapsed = time.time() - started

            if have_reference:
                results = detector.evaluate(frame, spots, want_debug=True)
            else:
                results = {}

            for sid, r in results.items():
                sums[sid][0] += r.changed_ratio
                sums[sid][1] += 1 if r.state == OCCUPIED else 0
                sums[sid][2] += 1
            if csv_w:
                for sid, r in results.items():
                    csv_w.writerow([f"{elapsed:.3f}", sid, f"{r.changed_ratio:.5f}", r.state])

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
                    print(f"t={elapsed:5.1f}s fps={fps:4.1f} thr={threshold:.2f} "
                          f"{status_line(spots, results)}")
                else:
                    print(f"t={elapsed:5.1f}s  waiting for a reference (press 'r')")

            if not args.headless:
                canvas = annotate(frame, spots, results, show_inner, show_diff)
                header = (f"threshold {threshold:.2f}  pixel {detector.params.pixel_threshold}  "
                          f"ref {'OK' if have_reference else 'MISSING - press r'}  "
                          f"{frames / max(elapsed, 1e-6):.1f} fps")
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
