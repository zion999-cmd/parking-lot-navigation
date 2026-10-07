"""Measure one real drag in the editor, and say where the time went.

Reported symptom: dragging a bay's corner is explosively slow. Measurement from the outside has
not reproduced it - the drawing is 6.3 ms per frame while a corner is dragged, a mouse callback
costs 0.09 ms, and a whole loop iteration is 27 ms - so the next step is to measure the drag that
is actually slow, in the window that is actually slow, with a breakdown rather than a total.

    python tests/measure_editor_drag.py                     # live camera, like calibration.py
    python tests/measure_editor_drag.py --frame data/reference/reference.jpg

Then: drag a bay corner (or a road point) for a few seconds, and press q. It prints, for the
frames while the drag was held:

    get_frame   how long each frame waited for a picture (a camera read dominates here)
    render      the drawing, split into the bays layer, the roads layer and everything else
    imshow      handing the picture to the window
    waitKey     the event pump - this is when the mouse is serviced at all
    events      mouse events delivered per iteration
    derives     how many times the navigation graph had to be worked out, and its cost

Whichever line is large is the answer. Run it with the same command you normally use, or the
numbers describe a different situation than the one that is slow.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

import calibration                                            # noqa: E402
import road_network                                           # noqa: E402
from camera import open_camera                                # noqa: E402
from spot_editor import SpotEditor                            # noqa: E402

WINDOW = calibration.WINDOW


class Tally:
    """Per-iteration timings, plus where the drawing time went."""

    def __init__(self) -> None:
        self.rows = []
        self.events = 0
        self.derives = 0
        self.derive_ms = 0.0

    def watch_derive(self):
        """Count and time every derive, including the cached path."""
        original = road_network.RoadNetwork.derive
        state = self

        def wrapper(inner_self, *args, **kwargs):
            t0 = time.perf_counter()
            try:
                return original(inner_self, *args, **kwargs)
            finally:
                state.derives += 1
                state.derive_ms += (time.perf_counter() - t0) * 1000

        road_network.RoadNetwork.derive = wrapper
        return original


def main() -> int:
    ap = argparse.ArgumentParser(description="Where did the drag's time go?")
    ap.add_argument("--source", default="http://192.168.3.118/stream")
    ap.add_argument("--frame", help="a still image instead of the camera, as calibration.py does")
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "parking_spots.json"))
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after N seconds (default: until q)")
    args = ap.parse_args()

    still = None
    if args.frame:
        still = cv2.imread(args.frame, cv2.IMREAD_COLOR)
        if still is None:
            print(f"error: could not read {args.frame}", file=sys.stderr)
            return 2
        get_frame = lambda: still                             # noqa: E731
        shape = still.shape[:2]
        print(f"  source: {args.frame} ({shape[1]}x{shape[0]}) - a still frame, like --frame")
    else:
        cam = open_camera(args.source)
        frame = None
        for _ in range(80):
            frame = cam.read()
            if frame is not None:
                break
        if frame is None:
            print(f"error: no frame from {args.source}", file=sys.stderr)
            return 2
        get_frame = cam.read
        shape = frame.shape[:2]
        print(f"  source: {args.source} ({shape[1]}x{shape[0]}) - a live camera read per frame")

    editor = SpotEditor(shape)
    if not editor.load(args.config):
        print(f"error: could not load {args.config}", file=sys.stderr)
        return 2
    app = calibration.CalibrationApp(editor, args.config, template=still)
    app.detector_params = calibration.DetectorParams()

    tally = Tally()
    original_derive = tally.watch_derive()

    # Time the two drawing halves separately: which one is slow is the whole question.
    spot_ms = [0.0]
    road_ms = [0.0]
    real_spots, real_roads = app._draw_spots, app._draw_roads

    def timed_spots(canvas, faint):
        t0 = time.perf_counter()
        real_spots(canvas, faint)
        spot_ms[0] += (time.perf_counter() - t0) * 1000

    def timed_roads(canvas, faint):
        t0 = time.perf_counter()
        real_roads(canvas, faint)
        road_ms[0] += (time.perf_counter() - t0) * 1000

    app._draw_spots, app._draw_roads = timed_spots, timed_roads

    def on_mouse(event, x, y, flags, param):
        tally.events += 1
        app.on_mouse(event, x, y, flags, param)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    cv2.imshow(WINDOW, app.render(still if still is not None else frame))
    cv2.waitKey(200)

    print("\n  drag a bay corner (or a whole bay, or a road point) for a few seconds;")
    print("  press q when the drag is done.\n")

    started = time.time()
    drag_frames = 0
    while True:
        tf0 = time.perf_counter()
        fresh = get_frame()
        tf1 = time.perf_counter()
        if still is None and fresh is None:
            continue
        canvas = still if fresh is None else fresh
        app._frame = canvas

        derives_before = tally.derives
        spot_ms[0] = road_ms[0] = 0.0
        tr0 = time.perf_counter()
        picture = app.render(canvas)
        tr1 = time.perf_counter()
        cv2.imshow(WINDOW, picture)
        tw0 = time.perf_counter()
        key = cv2.waitKey(5) & 0xFF
        tw1 = time.perf_counter()

        dragging = app.ed.dragging() or app._access_drag is not None
        if dragging:
            drag_frames += 1
            tally.rows.append({
                "frame": (tf1 - tf0) * 1000, "render": (tr1 - tr0) * 1000,
                "spots": spot_ms[0], "roads": road_ms[0],
                "imshow": (tw0 - tr1) * 1000, "wait": (tw1 - tw0) * 1000,
                "total": (tw1 - tf0) * 1000, "events": tally.events,
                "derives": tally.derives - derives_before, "layer": app.layer,
                # Growth is the difference between "slow" and "explosive": if any of these
                # climbs during one drag, something is accumulating rather than just costing.
                "nodes": len(app.road.net.nodes), "edges": len(app.road.net.edges),
                "bays": len(app.ed.spots), "undo": len(app.ed._undo),
                "derived_nodes": len(app.road._derived.nodes) if app.road._derived else 0,
            })
            tally.events = 0

        if key == ord("q") or key == 27:
            break
        if args.seconds and time.time() - started > args.seconds:
            break

    cv2.destroyAllWindows()
    road_network.RoadNetwork.derive = original_derive

    if not tally.rows:
        print("  no frames were captured while a drag was held - nothing to report.")
        print("  (grab a corner and move it, then press q)")
        return 0

    def med(field):
        return statistics.median(r[field] for r in tally.rows)

    print(f"  === {len(tally.rows)} frames with a drag held ===")
    print(f"  {'get_frame':>10} {med('frame'):8.1f} ms   (a camera read shows up here)")
    print(f"  {'render':>10} {med('render'):8.1f} ms   "
          f"= bays {med('spots'):.1f} + roads {med('roads'):.1f} + overlay/hud "
          f"{med('render') - med('spots') - med('roads'):.1f}")
    print(f"  {'imshow':>10} {med('imshow'):8.1f} ms   (handing the picture to the window)")
    print(f"  {'waitKey':>10} {med('wait'):8.1f} ms   "
          f"(the event pump - mouse moves are only serviced here)")
    print(f"  {'TOTAL':>10} {med('total'):8.1f} ms   => {1000 / med('total'):.0f} fps")
    print(f"  {'derives':>10} {med('derives'):8.1f} per frame, "
          f"{tally.derive_ms / max(tally.derives, 1):.1f} ms each, {tally.derives} in total")
    print(f"  events per frame: median {med('events'):.1f}")
    print(f"  layer: {tally.rows[-1]['layer']}")

    worst = sorted(tally.rows, key=lambda r: -r["total"])[:5]
    print("\n  slowest frames:")
    for r in worst:
        print(f"    total {r['total']:7.1f}  frame {r['frame']:7.1f}  render {r['render']:6.1f} "
              f"(bays {r['spots']:5.1f} + roads {r['roads']:6.1f})  imshow {r['imshow']:6.1f}  "
              f"wait {r['wait']:6.1f}  derives {r['derives']:.0f}  events {r['events']:.0f}")

    # Does it get worse as the drag goes on? "Explosive" and "constant" are different bugs.
    half = max(len(tally.rows) // 2, 1)
    first = statistics.median(r["total"] for r in tally.rows[:half])
    second = statistics.median(r["total"] for r in tally.rows[half:])
    grows = [f"{k} {tally.rows[0][k]} -> {tally.rows[-1][k]}" for k in
             ("nodes", "edges", "bays", "undo", "derived_nodes")
             if tally.rows[-1][k] != tally.rows[0][k]]
    print(f"\n  first half {first:.1f} ms  vs  second half {second:.1f} ms  "
          f"=> {'WORSE OVER TIME' if second > first * 1.3 else 'steady'}")
    print(f"  state during the drag: " + (", ".join(grows) if grows else "nothing grew"))
    print("\n  the largest of these columns is the mechanism.")

    # Written down as well, because the terminal and I are not the same window.
    out_dir = os.path.join(ROOT, "out")
    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "editor_drag_probe.txt")
        lines = [f"source: {args.frame or args.source}", f"frames with a drag held: {len(tally.rows)}",
                 f"layer: {tally.rows[-1]['layer']}"]
        for field in ("frame", "render", "spots", "roads", "imshow", "wait", "total",
                      "events", "derives"):
            lines.append(f"{field:>10} median {med(field):9.2f}  worst "
                         f"{max(r[field] for r in tally.rows):9.2f}")
        lines.append(f"derives total {tally.derives}, "
                     f"{tally.derive_ms / max(tally.derives, 1):.2f} ms each")
        lines.append(f"first half {first:.2f} ms, second half {second:.2f} ms")
        lines.append("state: " + (", ".join(grows) if grows else "nothing grew"))
        lines.append("")
        lines.append("per-frame: " + " ".join(
            f"{r['total']:.0f}" for r in tally.rows[:200]))
        lines.append("render per-frame: " + " ".join(
            f"{r['render']:.0f}" for r in tally.rows[:200]))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"  written to {path}")
    except OSError as exc:
        print(f"  could not write the report: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
