"""Production mask against the candidate mask, on the same frames, in one process.

    python tools/hero_lab/compare.py                 # saved frames + synthetic sweep
    python tools/hero_lab/compare.py --live 60       # add live frames from the camera

Nothing is written to vision/. The candidate is injected by swapping vehicle.red_mask for the
duration of the experiment, so the product's own feature stage - PCA, minAreaRect, taper, the
painted-end front rule, all of it - consumes both masks unchanged. That is the compatibility claim
being tested as much as the accuracy is: if a mask needs the feature stage modified, it is not a
mask candidate.

What it reports, per group, with no picking and choosing:

    mask bbox, area, ratio to the paint, elongation, front      per saved frame
    worst angle error and worst frame-to-frame jump             whole synthetic circle, 5 deg steps
    detection rate and mask ratio                               live frames, if --live is given
    milliseconds per frame                                      both groups

It also checks the copy: with both switches off, candidate_red_mask must reproduce the product's
mask pixel for pixel on every frame it sees. A copy that has drifted makes every number below
meaningless, so it fails loudly instead.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "vision", "src"))
sys.path.insert(0, os.path.join(ROOT, "vision", "tests"))
sys.path.insert(0, HERE)

import vehicle                                                        # noqa: E402
import candidate_mask as lab                                          # noqa: E402

SEED = os.path.join(ROOT, "vision", "data", "labels", "seed")


def measure(frame: np.ndarray, provider=None) -> dict:
    """One reading of the car through the product's own pipeline, optionally with a mask provider."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if provider is None:
        mask = vehicle.red_mask(hsv)
    else:
        mask = provider(hsv)
    paint = paint_area(frame)
    n, _l, st, _c = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return {"mask": None, "paint": paint, "poses": [], "hsv": hsv, "raw": mask}
    i = 1 + int(np.argmax(st[1:, 4]))
    bbox = (int(st[i, 0]), int(st[i, 1]), int(st[i, 2]), int(st[i, 3]))
    if provider is None:
        poses = vehicle.locate_all(frame)
    else:
        with lab.mask_provider(provider):
            poses = vehicle.locate_all(frame)
    return {"mask": (bbox, int(st[i, 4])), "paint": paint, "poses": poses, "hsv": hsv, "raw": mask}


def paint_area(frame: np.ndarray) -> int:
    """The paint blob as the pipeline sees it: config colours, opened, largest component."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    raw = np.zeros(frame.shape[:2], np.uint8)
    for lo, hi in vehicle.vehicle_config()["hue_ranges"]:
        raw |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    blob, _ = vehicle._largest_component(raw)
    return int(np.count_nonzero(blob))


def describe(tag: str, frame: np.ndarray, provider=None) -> dict:
    r = measure(frame, provider)
    poses = r["poses"]
    if r["mask"] is None:
        print(f"  {tag:<12} mask: none                 poses 0")
        return {"mask": None, "poses": 0, "elongation": 0.0, "front": None}
    (x, y, w, h), area = r["mask"]
    pose = poses[0] if poses else None
    front = pose.as_dict().get("front_heading") if pose else None
    el = pose.elongation if pose else 0.0
    print(f"  {tag:<12} mask {w:>3}x{h:<3}={area:<5} {area / max(r['paint'], 1):>5.2f}x paint  "
          f"elong {el:>4.2f}  front {str(front):>6}  poses {len(poses)}")
    return {"mask": (w, h, area), "poses": len(poses), "elongation": el, "front": front}


def drift_guard(frames) -> bool:
    """The copy must equal the product when both switches are off."""
    bad = 0
    for name, frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        a = vehicle.red_mask(hsv)
        b = lab.candidate_red_mask(hsv, drop_small=False, near_field=False)
        if not np.array_equal(a, b):
            bad += 1
            print(f"  DRIFT: {name}: {int(np.count_nonzero(a != b))} differing pixels")
    print(f"  copy check: {'ok, identical on all frames' if not bad else f'{bad} FRAMES DIFFER'}")
    return bad == 0


def synthetic_sweep(provider=None, provider_name: str = "A") -> dict:
    import test_paint_side as synth
    worst, jumps, prev, errors = 0.0, [], None, []
    for angle in range(0, 360, 15):
        frame, truth = synth._base_car(angle)
        if provider is None:
            poses = vehicle.locate_all(frame)
        else:
            with lab.mask_provider(provider):
                poses = vehicle.locate_all(frame)
        if not poses:
            errors.append((angle, None))
            continue
        front = poses[0].as_dict()["front_heading"]
        if front is None:
            errors.append((angle, None))
            continue
        err = abs((front - truth + 180) % 360 - 180)
        errors.append((angle, err))
        worst = max(worst, err)
        if prev is not None:
            jumps.append(abs((front - prev + 180) % 360 - 180))
        prev = front
    missing = [a for a, e in errors if e is None]
    print(f"  [{provider_name}] worst angle error {worst:>5.1f} deg   worst step "
          f"{max(jumps) if jumps else 0:>5.1f} deg   no front at {len(missing)}/24 angles")
    return {"worst": worst, "missing": len(missing)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", type=int, default=0, help="also read this many live frames")
    ap.add_argument("--source", default="http://192.168.3.118/stream")
    args = ap.parse_args()

    frames = []
    for name in ("live", "turned", "right_up"):
        path = os.path.join(SEED, f"{name}.jpg")
        img = cv2.imread(path)
        if img is not None:
            frames.append((name, img))
    print(f"=== copy check ({len(frames)} saved frames) ===")
    ok = drift_guard(frames)

    print("\n=== saved frames: A = production, B = candidate ===")
    for name, frame in frames:
        print(f"  {name}:")
        describe("A production", frame, None)
        describe("B candidate", frame, lab.candidate_red_mask)

    print("\n=== synthetic circle, 5-degree steps (truth = 180 - angle) ===")
    a = synthetic_sweep(None, "A")
    b = synthetic_sweep(lab.candidate_red_mask, "B")

    if args.live:
        print(f"\n=== live: {args.live} frames ===")
        from camera import open_camera
        try:
            cam = open_camera(args.source)
            hit = {"A": 0, "B": 0}
            ratios = {"A": [], "B": []}
            timings = {"A": [], "B": []}
            seen = 0
            while seen < args.live:
                frame = cam.read()
                if frame is None:
                    continue
                seen += 1
                for tag, provider in (("A", None), ("B", lab.candidate_red_mask)):
                    t0 = time.perf_counter()
                    r = measure(frame, provider)
                    timings[tag].append((time.perf_counter() - t0) * 1000.0)
                    if r["poses"]:
                        hit[tag] += 1
                    if r["mask"] is not None and r["paint"]:
                        ratios[tag].append(r["mask"][1] / r["paint"])
            cam.close()
            for tag in ("A", "B"):
                rr = sorted(ratios[tag])
                med = rr[len(rr) // 2] if rr else 0.0
                print(f"  [{tag}] detected {hit[tag]}/{seen} frames   mask/paint median {med:.2f} "
                      f"min {min(rr) if rr else 0:.2f}   {np.median(timings[tag]):.1f} ms/frame")
        except Exception as exc:                                     # camera busy or offline
            print(f"  live frames skipped: {exc}")

    print(f"\ncopy identical to production: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
