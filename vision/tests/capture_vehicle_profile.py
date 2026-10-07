"""Sample the real car into the front/rear profile template.

The sign classifier cannot have its template made up - the pattern has to come from this car.
This is the tool that measures it, in two runs that are meant to prove the sign separately:

  1.  python tests/capture_vehicle_profile.py --build
      Leave the car where it is. It captures ~60 frames, works out which way the existing sign
      says the car is facing, averages the profiles into one rear->front template, and prints
      the forward/reverse scores: on these frames forward must beat reverse clearly.

  2.  Turn the car round 180 degrees, then:
      python tests/capture_vehicle_profile.py --verify
      reverse must now beat forward.

Nothing about the position or the axis is touched; this only writes config/vehicle_profile.json.
Manual, like car_locate.py, which is why it is not a test_*.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

import vehicle                                              # noqa: E402

OUT_PATH = vehicle.PROFILE_TEMPLATE_PATH


def raw_profiles(source: str, n: int, seconds: float):
    """Capture frames and return (profile, existing evidence sign) per located frame."""
    cam = vehicle  # noqa
    from camera import open_camera
    cap = open_camera(source)
    rows = []
    t0 = time.time()
    while time.time() - t0 < seconds and len(rows) < n:
        frame = cap.read()
        if frame is None:
            continue
        pose = vehicle.locate(frame)
        if pose is None:
            continue
        radians = np.radians(pose.heading)
        mask = vehicle.red_mask(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        i = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        ys, xs = np.where(labels == i)
        points = np.column_stack([xs, ys]).astype(np.float32)
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[ys, xs].astype(np.float64)
        along = points[:, 0] * np.cos(radians) + points[:, 1] * np.sin(radians)
        profile = vehicle.profile_of(along, grey)
        if profile is not None:
            rows.append((profile, pose.front_evidence, pose.heading, pose.bbox))
    return rows


def normalize_profile(profile: np.ndarray) -> np.ndarray:
    """Shape only: zero mean, unit spread, so the template is pattern not lighting."""
    v = np.asarray(profile, dtype=np.float64)
    v = v - v.mean()
    spread = v.std()
    return v / spread if spread > 1e-9 else v


def build(rows):
    # The existing ends sign says which physical end is the front. Majority over the capture:
    # the single-frame sign is right ~90%, sixty frames make the orientation certain. Frames
    # with no sign are skipped.
    signs = [1 if ev > 0 else -1 for _, ev, _, _ in rows if ev is not None]
    if not signs:
        print("  no frames carried even the old sign; nothing to orient the template with")
        return 1
    front_is_plus = sum(signs) > 0
    print(f"  existing sign: front at the {'+axis' if front_is_plus else '-axis'} end "
          f"({sum(s > 0 for s in signs)}/{len(signs)} frames)")

    oriented = []
    for profile, _, _, _ in rows:
        p = normalize_profile(profile)
        # profile bins read low->high along +axis. Front at the +axis end means it already
        # reads rear->front; front at the -axis end means it reads front->rear: flip it.
        oriented.append(p if front_is_plus else p[::-1])

    template = np.mean(oriented, axis=0)
    template = normalize_profile(template)

    scores = [vehicle.score_profile(p, template) for p, _, _, _ in rows]
    fwd = np.median([s[0] for s in scores])
    rev = np.median([s[1] for s in scores])
    agree = np.mean([(s[0] > s[1]) == front_is_plus for s in scores])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump({"template": [round(float(v), 6) for v in template],
                   "bins": len(template), "frames": len(rows),
                   "front_is_plus_when_built": front_is_plus,
                   "median_forward": round(float(fwd), 3),
                   "median_reverse": round(float(rev), 3)}, fh, indent=2)

    print(f"  wrote {OUT_PATH}")
    print(f"  PROOF (same orientation): forward {fwd:+.3f}  reverse {rev:+.3f}  "
          f"margin {fwd-rev:+.3f}")
    print(f"  frames agreeing with the direction: {agree*100:.0f}%")
    if not (fwd > rev and fwd - rev > 0.2 and agree >= 0.9):
        print("  !! the correct direction did not win clearly - do not trust this template")
        return 1
    print("  OK: the visual sign recognises the front.")
    return 0


def verify(rows):
    template = vehicle.vehicle_template()
    if template is None:
        print("  no template yet; run --build first")
        return 1
    scores = [vehicle.score_profile(p, template) for p, _, _, _ in rows]
    fwd = np.median([s[0] for s in scores])
    rev = np.median([s[1] for s in scores])
    plus = np.mean([s[3] for s in scores])
    print(f"  frames: {len(scores)}")
    print(f"  forward {fwd:+.3f}  reverse {rev:+.3f}  margin {fwd-rev:+.3f}")
    print(f"  frames judged front_is_plus: {plus*100:.0f}%")
    if rev > fwd and rev - fwd > 0.2 and plus <= 0.1:
        print("  OK: with the car turned round, the reverse direction wins.")
        return 0
    print("  !! reverse did not win clearly; is the car actually turned 180 degrees?")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="http://192.168.3.118/stream")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--seconds", type=float, default=12.0)
    args = ap.parse_args()
    if not (args.build or args.verify):
        ap.error("say --build or --verify")

    rows = raw_profiles(args.source, args.frames, args.seconds)
    print(f"  captured profiles from {len(rows)} frames")
    if not rows:
        print("  no usable frames")
        return 1
    return build(rows) if args.build else verify(rows)


if __name__ == "__main__":
    raise SystemExit(main())
