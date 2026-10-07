"""Watch the hero car's detection box and catch the frame where it shortens.

Reported: the box is a rectangle, but now and then the long side collapses. Measured from the
outside over 60 s (554 frames) and it did not happen once - the long side stayed 65..70 - so this
tool exists to catch it where it does happen, with the mask that produced it.

Run it with main.py stopped, so it gets every frame rather than a share of them:

    python tests/watch_vehicle_box.py --seconds 120

It writes every shortened frame plus its red pieces to out/vehicle_watch/, and prints a summary
line per event: the box, how many red pieces the car was in at that moment, and their geometry.
That is what says whether the car was cut in two (the join failing) or simply detected smaller.

Not a test: it needs the camera.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

import vehicle                                              # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="http://192.168.3.118/stream")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--ratio", type=float, default=0.8,
                    help="a frame counts as shortened below this fraction of the running median")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "vehicle_watch"))
    args = ap.parse_args()

    from camera import open_camera
    cap = open_camera(args.source)
    frame = None
    t0 = time.time()
    while frame is None and time.time() - t0 < 15:
        frame = cap.read()
    if frame is None:
        print("no frame: is the camera free? (stop main.py first)")
        return 1

    cfg = vehicle.vehicle_config()
    print(f"  watching {args.seconds:.0f}s; open_kernel={cfg['open_kernel']} "
          f"close={cfg['close_kernel']} min_area={cfg['min_area']}")
    os.makedirs(args.out, exist_ok=True)

    lengths, events, frames = [], 0, 0
    started = time.time()
    while time.time() - started < args.seconds:
        f = cap.read()
        if f is None:
            continue
        pose = vehicle.locate(f)
        if pose is None:
            continue
        frames += 1
        long_side = max(pose.bbox[2], pose.bbox[3])
        lengths.append(long_side)
        median = float(np.median(lengths[-60:]))
        if len(lengths) > 20 and long_side < args.ratio * median:
            events += 1
            hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
            mask = vehicle.red_mask(hsv, open_kernel=cfg["open_kernel"])
            n, _lab, stats, _c = cv2.connectedComponentsWithStats(mask, 8)
            pieces = sorted(((int(stats[i, 4]), int(stats[i, 2]), int(stats[i, 3]),
                              int(stats[i, 0]), int(stats[i, 1]))
                             for i in range(1, n) if stats[i, 4] >= 8), reverse=True)
            name = f"short_{events:02d}_long{long_side}.jpg"
            cv2.imwrite(os.path.join(args.out, name), f)
            print(f"  [{frames:5d}] SHORT: box {pose.bbox} long={long_side} median={median:.0f} "
                  f"pieces={len(pieces)} {pieces[:4]} -> {name}")
    rate = frames / max(time.time() - started, 1e-6)
    print(f"\n  {frames} frames in {time.time() - started:.0f}s ({rate:.1f} fps), {events} shortened")
    if rate < 15:
        # The camera gives each client a share of its frames, so a client running beside main.py
        # sees a subset - and a fault that alternates frame to frame would hide in the gaps.
        print("  NOTE: only {:.0f} fps, so this client is sharing the camera. Stop main.py and run\n"
              "        this again: at a share of the frames, an alternating fault is aliased away."
              .format(rate))
    if lengths:
        print(f"  long side: median {np.median(lengths):.0f} min {min(lengths)} max {max(lengths)}")
    if not events:
        print("  nothing shortened. If the window still shows it, the box the window draws is not "
              "this value - say so and it will be chased in main.py's own drawing.")
    else:
        print(f"  frames saved in {args.out} - those are the evidence to look at.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
