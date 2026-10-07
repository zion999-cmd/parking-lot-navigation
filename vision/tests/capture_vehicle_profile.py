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
    ap.add_argument("--record", action="store_true",
                    help="record one orientation's full multi-channel profile")
    ap.add_argument("--finalize", action="store_true",
                    help="build config/vehicle_profile.json from vehicle_<label>.json recordings")
    ap.add_argument("--label", help="orientation label for --record (e.g. right)")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--seconds", type=float, default=12.0)
    args = ap.parse_args()
    if not (args.build or args.verify or args.record or args.finalize):
        ap.error("say --build, --verify, --record or --finalize")
    if args.record and not args.label:
        ap.error("--record needs a --label")
    if args.finalize:
        return finalize()

    if args.record:
        return record(args.source, args.label, args.frames, args.seconds)

    rows = raw_profiles(args.source, args.frames, args.seconds)
    print(f"  captured profiles from {len(rows)} frames")
    if not rows:
        print("  no usable frames")
        return 1
    return build(rows) if args.build else verify(rows)




# ---------------------------------------------------------------------------
# Record one orientation, all channels
# ---------------------------------------------------------------------------
RECORD_BINS = 24
CHANNELS = ("gray", "saturation", "red_fraction", "width")


def record(source: str, label: str, want_frames: int, seconds: float) -> int:
    """Capture one facing of the car across channels, without using the old sign to orient it.

    Everything is binned along the current heading's +axis as-is; the label the person gives is
    the only orientation fact, so the measurement cannot be contaminated by the weak sign. Four
    channels are recorded because the gray profile proved not to survive a real turn:

      gray           brightness (the old, fragile channel)
      saturation     glass is low-sat, red paint high-sat - survives reflections
      red_fraction   1 where the filled blob is red, 0 at an enclosed window: hole geometry
      width          silhouette width per bin (the taper structure)
    """
    from camera import open_camera
    cap = open_camera(source)
    collected = {c: [] for c in CHANNELS}
    headings, bboxes, elongations = [], [], []
    t0 = time.time()
    while time.time() - t0 < seconds and min((len(v) for v in collected.values()),
                                             default=0) < want_frames:
        frame = cap.read()
        if frame is None:
            continue
        pose = vehicle.locate(frame)
        if pose is None:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        filled = vehicle.red_mask(hsv)
        raw_red = np.zeros(frame.shape[:2], np.uint8)
        for lo, hi in vehicle.RED_HUE_RANGES:
            raw_red |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        raw_red = cv2.morphologyEx(raw_red, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        count, labels, stats, _ = cv2.connectedComponentsWithStats(filled, 8)
        i = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        ys, xs = np.where(labels == i)
        points = np.column_stack([xs, ys]).astype(np.float32)
        r = np.radians(pose.heading)
        along = points[:, 0] * np.cos(r) + points[:, 1] * np.sin(r)
        across = -points[:, 0] * np.sin(r) + points[:, 1] * np.cos(r)
        lo, hi = float(along.min()), float(along.max())
        if hi - lo < 12:
            continue
        index = np.clip(((along - lo) / (hi - lo) * RECORD_BINS).astype(int),
                        0, RECORD_BINS - 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[ys, xs].astype(np.float64)
        sat = hsv[:, :, 1][ys, xs].astype(np.float64)
        is_raw_red = (raw_red[ys, xs] > 0).astype(np.float64)

        g = np.zeros(RECORD_BINS); s = np.zeros(RECORD_BINS)
        f = np.zeros(RECORD_BINS); w = np.zeros(RECORD_BINS)
        gc = np.zeros(RECord_bins if False else RECORD_BINS)
        for b in range(RECORD_BINS):
            m = index == b
            if not m.any():
                continue
            g[b] = gray[m].mean()
            s[b] = sat[m].mean()
            f[b] = is_raw_red[m].mean()
            w[b] = across[m].max() - across[m].min()
        collected["gray"].append(g); collected["saturation"].append(s)
        collected["red_fraction"].append(f); collected["width"].append(w)
        headings.append(pose.heading); bboxes.append(pose.bbox)
        elongations.append(pose.elongation)

    n = min((len(v) for v in collected.values()), default=0)
    if not n:
        print("  no usable frames captured")
        return 1
    averaged = {c: np.mean(collected[c][:n], axis=0) for c in CHANNELS}

    out_path = os.path.join(os.path.dirname(OUT_PATH), f"vehicle_{label}.json")
    payload = {"label": label, "frames": n, "bins": RECORD_BINS,
               "heading_deg": round(float(np.median(headings)), 1),
               "bbox": [int(v) for v in np.median(bboxes, axis=0)],
               "elongation": round(float(np.median(elongations)), 2),
               "channels": {c: [round(float(v), 3) for v in averaged[c]]
                            for c in CHANNELS}}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"  recorded '{label}' from {n} frames, heading {payload['heading_deg']} deg")
    print(f"  wrote {out_path}")

    def bar(values, dark_is_low=True):
        v = np.asarray(values, float); v = (v - v.mean())
        v = v / max(v.std(), 1e-9); v = (v - v.min()) / (v.max() - v.min())
        return "".join(("█" if x > .66 else ("▒" if x > .33 else "·")) for x in v)

    print("  gray        " + bar(averaged["gray"]))
    print("  saturation  " + bar(averaged["saturation"]))
    print("  red_fraction " + bar(averaged["red_fraction"]) + "   (· = window/hole)")
    print("  width       " + bar(averaged["width"]))
    print("  ready: rotate the car 180 degrees, then record again with the new label.")
    return 0




def finalize() -> int:
    """Build the sign template from the two orientation recordings.

    vehicle_right.json and vehicle_left.json were recorded along heading ~0 (axis points +x),
    bins low->high from the -x end. Facing right, the front is +x so the bins read rear->front;
    facing left they read front->rear, so the left recording is reversed before averaging.

    The template keeps the two channels the head-to-head test picked - saturation (separation
    0.94) and red_fraction (0.81):
    """
    base = os.path.dirname(OUT_PATH)
    paths = {lbl: os.path.join(base, f"vehicle_{lbl}.json") for lbl in ("right", "left")
             }
    for lbl, path in paths.items():
        if not os.path.exists(path):
            print(f"  missing {path}; record --label {lbl} first")
            return 1
    rec = {lbl: json.load(open(path, encoding="utf-8")) for lbl, path in paths.items()}
    r, l = rec["right"]["channels"], rec["left"]["channels"]
    template = {}
    for c in ("saturation", "red_fraction"):
        right_c = np.asarray(r[c], dtype=float)
        left_c = np.asarray(l[c], dtype=float)[::-1]     # canonical rear->front
        template[c] = (right_c + left_c) / 2.0
    payload = {"bins": len(template["saturation"]),
               "orientation": "rear->front, bin0 at the boot",
               "sources": {lbl: rec[lbl]["frames"] for lbl in rec},
               "template": {c: [round(float(v), 3) for v in template[c]]
                            for c in template}}
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  wrote {OUT_PATH}")
    for c in template:
        v=template[c]; v=(v-v.mean())/max(v.std(),1e-9); v=(v-v.min())/(v.max()-v.min())
        print(f"    {c:<14} " + "".join("█" if x>.66 else ("▒" if x>.33 else "·") for x in v))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
