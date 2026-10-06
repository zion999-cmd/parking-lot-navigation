"""Locate the hero car by colour and shape. A prototype, and it measures before it decides.

    python tests/car_locate.py                       # whole frame, 10 frames, save a debug image
    python tests/car_locate.py --bay B4              # look inside one bay as well
    python tests/car_locate.py --hues                # what colours are actually in the frame

Why this exists rather than a marker: the job is to know where the car is and which way it
points. A marker gives that precisely, but needs to be nearly half the car's width and flat
towards the camera to be readable at 800x600 across a whole sandbox - and it is not being
read. Colour and shape need no hardware and no reference frame, so they are unaffected by the
two things that were wrecking measurements here: lighting changes and sub-pixel shifts of the
painted bay lines.

What it cannot do, and this is the honest limit: a car's long axis has no front. PCA gives the
axis, not the direction of travel, so the heading is known modulo 180 degrees. If navigation
needs to know which end is the front, that needs something on the car that is not symmetric -
which is exactly what a marker is for.

Nothing here is wired into the contract yet.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from camera import open_camera                                  # noqa: E402
from road_network import RoadNetwork                            # noqa: E402
from spot_config import load_spots                              # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Red wraps around the hue circle, so it takes two ranges. Saturation and value floors keep
# washed-out and very dark pixels out - grey floor is unsaturated, so this does most of the
# work before any shape reasoning.
# Chosen from the frame, not guessed: the red car measures H=0..2 and 176..179 with S>=90
# and V>=60, while the orange-red strip along the top edge of the enclosure sits at H=11..15.
# The first thresholds used H<=12 and swallowed it, which is how a strip of trim at the top of
# the picture became the largest "red object" in the frame.
RED_RANGES = (((0, 90, 60), (3, 255, 255)), ((176, 90, 60), (180, 255, 255)))


def red_mask(hsv: np.ndarray) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in RED_RANGES:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    # Close with a big kernel: the car's red area breaks into two pieces wherever a dark
    # band crosses it (a shadow along the side, the join between body and roof), and a
    # centroid that jumps between two halves is what the x jitter of +/-7px was. Nothing
    # else red is nearby, so joining pieces a dozen pixels apart costs nothing.
    big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, big)
    # Fill holes. A car's windows, its shadowed side and any dark trim drop out of the colour
    # mask and leave holes inside the blob; those holes are what made the axis jump between
    # two orientations from frame to frame, because PCA weighs every pixel and the holes come
    # and go. Flood filling from the border and inverting keeps the outline and loses the holes.
    holes = mask.copy()
    cv2.floodFill(holes, np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8), (0, 0), 255)
    return mask | cv2.bitwise_not(holes)


def touches_border(bbox, shape, margin: int = 2) -> bool:
    """True when a blob runs into the edge of the picture.

    The lighting along the top of the enclosure is orange-red and shows up as a big blob.
    A car parked in the lot does not sit half outside the frame, so this is cheap and safe.
    """
    x, y, w, h = bbox
    return x <= margin or y <= margin or x + w >= shape[1] - margin or y + h >= shape[0] - margin


def describe(mask: np.ndarray, min_area: int = 80) -> List[dict]:
    """Every red blob big enough to be a car, with its position and its axis."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, count):
        area = int(stats[i, 4])
        if area < min_area:
            continue
        x, y, w, h = (int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 2]), int(stats[i, 3]))
        ys, xs = np.where(labels == i)
        points = np.column_stack([xs, ys]).astype(np.float32)
        centre = points.mean(axis=0)
        cov = np.cov((points - centre).T)
        values, vectors = np.linalg.eigh(cov)
        major = vectors[:, int(np.argmax(values))]
        heading = float(np.degrees(np.arctan2(major[1], major[0]))) % 180.0
        elongation = float(np.sqrt(max(values) / max(min(values), 1e-6)))
        rect = cv2.minAreaRect(points)
        (rdx, rdy), (rw, rh), rangle = rect
        # Long axis of the fitted rectangle, as an angle in 0..180. Usually steadier than PCA
        # on a blob whose outline flickers: the rectangle depends on the extremes, PCA on
        # every pixel.
        rect_heading = (rangle if rw >= rh else rangle + 90.0) % 180.0
        out.append({
            "area": area, "bbox": (x, y, w, h),
            "centre": (float(centre[0]), float(centre[1])),
            "heading": heading, "elongation": elongation,
            "rect_heading": float(rect_heading),
            "rect_size": (float(rect[1][0]), float(rect[1][1])),
            "rect_angle": float(rect[2]),
            "fill": area / max(w * h, 1),
        })
    out.sort(key=lambda b: -b["area"])
    return out


def bay_containing(spots, x: float, y: float) -> Optional[str]:
    for spot in spots:
        if cv2.pointPolygonTest(np.asarray(spot.polygon, np.float32), (x, y), False) >= 0:
            return spot.id
    return None


def hue_report(hsv: np.ndarray) -> None:
    """Which hues are actually present, so the thresholds are chosen from data."""
    sat = hsv[:, :, 1].astype(int)
    strong = sat >= 90
    print(f"  饱和像素 (S>=90): {strong.mean() * 100:.1f}%")
    if not strong.any():
        return
    hist = np.bincount(hsv[:, :, 0][strong].ravel(), minlength=180)
    top = np.argsort(hist)[::-1][:10]
    print("  色相分布 (H 值 0-180，红色在 0 附近和 168-180):")
    for h in sorted([int(t) for t in top if hist[t] > 0]):
        print(f"    H={h:3d}  {hist[h]:6d} px")


def watch(args) -> int:
    """Print the reading once a second until interrupted.

    This is how the prototype is meant to be checked: pick the car up, and see whether the
    numbers follow it. A single reading cannot show that a measurement tracks anything.
    """
    cam = open_camera(args.source)
    spots = load_spots(args.config, (600, 800)) if os.path.exists(args.config) else []
    roads = RoadNetwork.load(args.config, (600, 800)) if os.path.exists(args.config) else None
    print("每秒输出一次；挪车看数字跟不跟。Ctrl-C 结束。\n")
    print(f"{'时间':>6}  {'位置':>14}  {'朝向':>7}  {'伸长':>5}  {'面积':>6}  位置")
    started = time.time()
    try:
        while True:
            frame = None
            for _ in range(20):
                frame = cam.read()
                if frame is not None:
                    break
            if frame is None:
                print("  没有帧")
                time.sleep(1)
                continue
            blobs = [b for b in describe(red_mask(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)),
                                         args.min_area)
                     if args.keep_border or not touches_border(b["bbox"], frame.shape)]
            if not blobs:
                print(f"{time.time()-started:6.0f}  {'-':>14}  {'-':>7}  {'-':>5}  {'-':>6}  "
                      f"看不到红车 (visible: false)")
            else:
                b = blobs[0]
                where = bay_containing(spots, *b["centre"])
                if where:
                    loc = f"车位 {where}"
                elif roads is not None and roads.nodes:
                    near = roads.closest_point(b["centre"][0], b["centre"][1], skip_stubs=True)
                    loc = (f"路上 (距路 {near.distance:.0f}px)" if near and near.distance < 35
                           else f"空地上 (距路 {near.distance:.0f}px)" if near else "?")
                else:
                    loc = "?"
                print(f"{time.time()-started:6.0f}  ({b['centre'][0]:6.1f},{b['centre'][1]:6.1f})  "
                      f"{b.get('rect_heading', b['heading']):6.1f}°  {b['elongation']:5.2f}  "
                      f"{b['area']:6d}  {loc}")
                if len(blobs) > 1:
                    print(f"        另有 {len(blobs)-1} 个红块（面积 {[x['area'] for x in blobs[1:4]]}）")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n结束")
    finally:
        cam.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Find the hero car by colour and shape")
    ap.add_argument("--source", default="http://192.168.3.118/stream")
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "parking_spots.json"))
    ap.add_argument("--bay", default=None, help="also report what is inside this bay")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--hues", action="store_true", help="report the hues present and exit")
    ap.add_argument("--min-area", type=int, default=80)
    ap.add_argument("--keep-border", action="store_true",
                    help="also report blobs that run into the edge of the frame (default: "
                         "ignore them, they are the enclosure's trim, not a car)")
    ap.add_argument("--save", default=os.path.join(ROOT, "data", "debug", "car_locate.jpg"))
    ap.add_argument("--watch", action="store_true",
                    help="keep printing where the car is, once a second, until Ctrl-C - move "
                         "the car and watch the numbers follow")
    args = ap.parse_args()

    cam = open_camera(args.source)
    frames = []
    started = time.time()
    while len(frames) < max(args.frames, 1) and time.time() - started < 90:
        f = cam.read()
        if f is not None:
            frames.append(f)
    cam.close()
    if not frames:
        print("error: no frames arrived", file=sys.stderr)
        return 2

    if args.watch:
        return watch(args)

    reference = frames[0]
    hsv0 = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
    print(f"{len(frames)} 帧 {reference.shape[1]}x{reference.shape[0]}")

    if args.hues:
        hue_report(hsv0)
        return 0

    spots = load_spots(args.config, reference.shape[:2]) if os.path.exists(args.config) else []
    roads = (RoadNetwork.load(args.config, reference.shape[:2])
             if os.path.exists(args.config) else None)

    mask = red_mask(hsv0)
    print(f"红色像素: {mask.mean() / 255 * 100:.2f}% of the frame")
    blobs = describe(mask, args.min_area)
    dropped = [b for b in blobs if touches_border(b["bbox"], reference.shape)]
    if not args.keep_border:
        blobs = [b for b in blobs if not touches_border(b["bbox"], reference.shape)]
        if dropped:
            print(f"  (忽略了 {len(dropped)} 个贴着画面边缘的红块 —— 那是围栏的颜色，不是车)")
    print(f"\n=== 红色块 (面积 >= {args.min_area}) ===")
    if not blobs:
        print("  没有找到红色块 —— 把 --hues 的输出发我，我按实际色相调阈值")
    for b in blobs[:6]:
        x, y, w, h = b["bbox"]
        where = bay_containing(spots, *b["centre"]) if spots else None
        print(f"  面积 {b['area']:6d}  {w:3d}x{h:3d}px  @({b['centre'][0]:.0f},{b['centre'][1]:.0f})")
        print(f"     朝向 {b['heading']:6.1f}° (mod 180)  伸长比 {b['elongation']:.2f}  "
              f"填满度 {b['fill']:.2f}  外接旋转框 {b['rect_size'][0]:.0f}x{b['rect_size'][1]:.0f}")
        if where:
            print(f"     → 在车位 {where}")
        elif roads is not None and roads.nodes:
            near = roads.closest_point(b["centre"][0], b["centre"][1], skip_stubs=True)
            if near is not None:
                on = "在路上" if near.distance < 35 else "不在路上也不在车位里"
                print(f"     → {on}（距最近的路 {near.distance:.0f}px）")

    if args.bay and spots:
        target = [s for s in spots if s.id == args.bay]
        if target:
            poly = np.asarray(target[0].polygon, np.float32)
            x0, y0 = poly.min(axis=0).astype(int)
            x1, y1 = poly.max(axis=0).astype(int)
            inside = mask[y0:y1, x0:x1]
            print(f"\n=== {args.bay} 区内 (x {x0}-{x1}, y {y0}-{y1}) ===")
            print(f"  红色像素 {int(inside.sum() / 255)} / {inside.size} "
                  f"= {inside.mean() / 255 * 100:.1f}%")
            for b in describe(np.pad(inside, ((y0, 0), (x0, 0))), args.min_area)[:2]:
                print(f"  区内红块 面积 {b['area']}  朝向 {b['heading']:.1f}°  "
                      f"伸长比 {b['elongation']:.2f}")
        else:
            print(f"\n  车位 {args.bay} 不在配置里")

    # How steady is it while nothing moves? That is the accuracy the navigator would get.
    if len(frames) > 2:
        # Track the object that was identified, not simply the biggest one each time: the
        # biggest red thing in the frame is trim along the top edge, and measuring its
        # comings and goings says nothing about how steady the car's reading is.
        anchor = None
        if args.bay and spots:
            target = [s for s in spots if s.id == args.bay]
            if target:
                anchor = tuple(np.asarray(target[0].polygon, float).mean(axis=0))
        elif blobs:
            anchor = blobs[0]["centre"]
        track = []
        for f in frames[1:]:
            m = red_mask(cv2.cvtColor(f, cv2.COLOR_BGR2HSV))
            cands = [b for b in describe(m, args.min_area)
                     if args.keep_border or not touches_border(b["bbox"], reference.shape)]
            if not cands:
                continue
            b = (min(cands, key=lambda b: (b["centre"][0] - anchor[0]) ** 2
                     + (b["centre"][1] - anchor[1]) ** 2) if anchor else cands[0])
            track.append((b["centre"][0], b["centre"][1], b["heading"], b["area"],
                          b.get("rect_heading", b["heading"])))
        if len(track) > 2:
            a = np.array(track)

            def circ_std(deg):
                rad = np.radians(deg * 2)
                return np.degrees(np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * rad)))))) / 2

            print(f"\n=== 静止时的抖动（{len(track)} 帧，车没动；填洞之后的读数）===")
            print(f"  位置 x±{a[:,0].std():.2f}px  y±{a[:,1].std():.2f}px")
            print(f"  面积 中位 {int(np.median(a[:,3]))}px² (±{a[:,3].std() / max(np.median(a[:,3]), 1) * 100:.1f}%)")
            print(f"  PCA      朝向 ±{circ_std(a[:,2]):.2f}°   读数 {[round(v,1) for v in a[:,2]]}")
            print(f"  minAreaRect 朝向 ±{circ_std(a[:,4]):.2f}°   读数 {[round(v,1) for v in a[:,4]]}")
            best = min(circ_std(a[:,2]), circ_std(a[:,4]))
            print(f"\n  对照导航需求 (位置 ±5px, 朝向 ±15°): "
                  + ("✅ 达标" if a[:,0].std() <= 5 and a[:,1].std() <= 5 and best <= 15
                     else "❌ 还差一点"))

    try:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        draw = reference.copy()
        for b in blobs[:4]:
            x, y, w, h = b["bbox"]
            cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 255, 0), 1)
            c = (int(b["centre"][0]), int(b["centre"][1]))
            length = max(w, h) * 0.6
            rad = np.radians(b["heading"])
            tip = (int(c[0] + length * np.cos(rad)), int(c[1] + length * np.sin(rad)))
            cv2.arrowedLine(draw, c, tip, (0, 0, 255), 2, tipLength=0.25)
            rect_deg = b.get("rect_heading", b["heading"])
            cv2.putText(draw, f"PCA {b['heading']:.0f} / rect {rect_deg:.0f}", (x, max(y - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(args.save, draw)
        print(f"\n标注图: {os.path.abspath(args.save)}（绿框=红色块, 红箭头=长轴方向）")
    except OSError as exc:
        print(f"  could not save: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
