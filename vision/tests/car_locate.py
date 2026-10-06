"""Check the hero car's localisation, using the same code main.py uses.

    python tests/car_locate.py                 # one reading, plus how steady it is
    python tests/car_locate.py --watch         # once a second until Ctrl-C: move the car
    python tests/car_locate.py --front         # the two cues that could tell front from back
    python tests/car_locate.py --hues          # what hues are actually in the frame

This deliberately contains no detection code of its own. It calls src/vehicle.py, the same
module main.py calls, so the tool cannot drift away from what the product does - which is what
happened when this file was first written, before that module existed.

`--hues` is the exception and it is the reason the thresholds are what they are: the red car
measures hue 0..2 and 176..179, while the orange-red trim along the top of the enclosure
measures 11..15. Reading the hues off a real frame is how those numbers were chosen.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import vehicle                                                  # noqa: E402
from camera import open_camera                                  # noqa: E402
from road_network import RoadNetwork                            # noqa: E402
from spot_config import load_spots                              # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def grab(source: str, count: int, timeout: float = 90.0):
    cam = open_camera(source)
    frames, started = [], time.time()
    try:
        while len(frames) < count and time.time() - started < timeout:
            f = cam.read()
            if f is not None:
                frames.append(f)
    finally:
        cam.close()
    return frames


def hue_report(frame: np.ndarray) -> None:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    strong = hsv[:, :, 1] >= 90
    print(f"  饱和像素 (S>=90): {strong.mean() * 100:.1f}%")
    if not strong.any():
        return
    hist = np.bincount(hsv[:, :, 0][strong].ravel(), minlength=180)
    print("  色相分布 (红在 0 附近和 168-180)：")
    for h in sorted(int(v) for v in np.argsort(hist)[::-1][:10] if hist[v] > 0):
        print(f"    H={h:3d}  {hist[h]:6d} px")


def describe_pose(pose, location) -> str:
    where = ""
    if location:
        if location["kind"] == "spot":
            where = f"  在车位 {location['id']}"
        elif location["kind"] == "road":
            where = f"  在路上 (距路 {location['distance_to_road']:.0f}px)"
        elif location["kind"] == "floor":
            where = f"  在空地 (距路 {location['distance_to_road']:.0f}px)"
    agree = min(abs(pose.heading - pose.heading_pca),
                180 - abs(pose.heading - pose.heading_pca))
    trust = ("可信" if pose.elongation >= 1.6 and agree <= 10
             else "形状可疑 —— 长轴读数不可信")
    return (f"  ({pose.x:6.1f},{pose.y:6.1f})  轴 {pose.heading:5.1f}°  "
            f"伸长比 {pose.elongation:.2f}  PCA {pose.heading_pca:5.1f}° (差 {agree:.1f}°)  "
            f"面积 {pose.area}  {trust}{where}")


def front_signals(frames, args) -> int:
    """Measure the two cues that could tell the car's front from its back.

    Both are measured *inside* the red mask, so neither can be a shadow lying on the ground.
    The only question left is whether they belong to the car or to the lighting. A design
    feature - a windscreen, a tapered nose - stays with the car when it is turned around; a
    lighting effect stays with the world. So the test is: turn the car 180 degrees and run this
    again.
    """
    bins = 24
    rows = []
    for frame in frames:
        pose = vehicle.locate(frame, min_area=args.min_area, keep_border=args.keep_border)
        if pose is None:
            continue
        mask = vehicle.red_mask(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = (frame.shape[1] * 2, frame.shape[0] * 2)
        rot = cv2.getRotationMatrix2D((pose.x, pose.y), pose.heading, 1.0)
        aligned = cv2.warpAffine(mask, rot, size)
        aligned_gray = cv2.warpAffine(gray, rot, size)
        ys, xs = np.where(aligned > 0)
        if not len(xs):
            continue
        sub = aligned[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        subg = aligned_gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        length = sub.shape[1]
        widths = (sub > 0).sum(axis=0).astype(float)
        prof = np.array([widths[int(i * length / bins):int((i + 1) * length / bins)].mean()
                         for i in range(bins)])
        taper = ((prof[-(bins // 4):].mean() - prof[:bins // 4].mean())
                 / max(prof[-(bins // 4):].mean() + prof[:bins // 4].mean(), 1e-6))
        tones = []
        for i in range(bins):
            seg = subg[:, int(i * length / bins):int((i + 1) * length / bins)]
            m = sub[:, int(i * length / bins):int((i + 1) * length / bins)] > 0
            tones.append(seg[m].mean() if m.any() else np.nan)
        tones = np.array(tones)
        rows.append((taper, int(np.nanargmin(tones)) / bins,
                     float(np.nanmax(tones) - np.nanmin(tones))))
    if not rows:
        print("没有找到红车")
        return 1
    a = np.array(rows)
    print(f"{len(a)} 帧:")
    print(f"  形状锥度 (右端比左端宽多少): 均值 {a[:,0].mean():+.3f}  "
          f"标准差 {a[:,0].std():.3f}  "
          f"{'✅ 符号稳定' if a[:,0].std() < abs(a[:,0].mean()) else '⚠️ 噪声比信号大'}")
    print(f"  最暗一段位于轴上            : 均值 {a[:,1].mean()*100:.0f}%  "
          f"标准差 {a[:,1].std()*100:.0f}%")
    print(f"  明暗对比                    : {a[:,2].mean():.1f} 灰阶")
    print()
    print("把车**原地转 180°** 再运行一次，对比：")
    print("  形状锥度**符号翻转**  → 车身形状 ✅ 可用")
    print("  形状锥度**符号不变**  → 光照切掉了一段红漆 ✗ 不可用")
    print("  暗带**变成 ~33%**     → 车的设计特征（挡风玻璃）✅ 可用")
    print("  暗带**仍在 ~67%**     → 光照效果（背光侧暗）✗ 不可用")
    return 0


def watch(args) -> int:
    """Print the reading once a second until interrupted.

    This is how the localisation is meant to be checked: pick the car up and see whether the
    numbers follow it. A single reading cannot show that a measurement tracks anything.
    """
    cam = open_camera(args.source)
    spots = load_spots(args.config, (600, 800)) if os.path.exists(args.config) else []
    roads = RoadNetwork.load(args.config) if os.path.exists(args.config) else None
    print("每秒输出一次；挪车看数字跟不跟。Ctrl-C 结束。\n")
    print(f"{'时间':>6}  {'位置':>15}  {'轴':>7}  {'伸长':>5}  {'面积':>6}  位置")
    started = time.time()
    try:
        while True:
            frame = None
            for _ in range(20):
                frame = cam.read()
                if frame is not None:
                    break
            if frame is None:
                time.sleep(1)
                continue
            pose = vehicle.locate(frame, min_area=args.min_area,
                                  keep_border=args.keep_border)
            if pose is None:
                print(f"{time.time()-started:6.0f}  {'-':>15}  {'-':>7}  {'-':>5}  {'-':>6}  "
                      f"看不到红车 (visible: false)")
            else:
                loc = vehicle.location_of(pose, spots, roads)
                where = (f"车位 {loc['id']}" if loc["kind"] == "spot"
                         else f"{loc['kind']} (距路 {loc['distance_to_road']}px)"
                         if loc["distance_to_road"] is not None else loc["kind"])
                print(f"{time.time()-started:6.0f}  ({pose.x:6.1f},{pose.y:6.1f})  "
                      f"{pose.heading:6.1f}°  {pose.elongation:5.2f}  {pose.area:6d}  {where}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n结束")
    finally:
        cam.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Check the hero car localisation")
    ap.add_argument("--source", default="http://192.168.3.118/stream")
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "parking_spots.json"))
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--min-area", type=int, default=vehicle.DEFAULT_MIN_AREA)
    ap.add_argument("--keep-border", action="store_true",
                    help="do not ignore red blobs that run into the edge of the picture")
    ap.add_argument("--hues", action="store_true", help="report the hues in the frame and stop")
    ap.add_argument("--front", action="store_true",
                    help="measure the two cues that could tell front from back")
    ap.add_argument("--watch", action="store_true", help="print once a second until Ctrl-C")
    ap.add_argument("--save", default=os.path.join(ROOT, "data", "debug", "car_locate.jpg"))
    args = ap.parse_args()

    if args.watch:
        return watch(args)

    frames = grab(args.source, max(args.frames, 1))
    if not frames:
        print("error: no frames arrived", file=sys.stderr)
        return 2
    frame = frames[0]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(f"{len(frames)} 帧 {frame.shape[1]}x{frame.shape[0]}  "
          f"亮度 {gray.mean():.1f} 对比度 {gray.std():.1f} "
          f"清晰度 {cv2.Laplacian(gray, cv2.CV_64F).var():.0f}")

    if args.hues:
        hue_report(frame)
        return 0
    if args.front:
        return front_signals(frames, args)

    spots = load_spots(args.config, frame.shape[:2]) if os.path.exists(args.config) else []
    roads = RoadNetwork.load(args.config) if os.path.exists(args.config) else None

    pose = vehicle.locate(frame, min_area=args.min_area, keep_border=args.keep_border)
    print(f"\n=== 红车 ===")
    if pose is None:
        print("  找不到红车 —— visible: false（这是正确答案，不是失败）")
    else:
        print(describe_pose(pose, vehicle.location_of(pose, spots, roads) if roads else None))
    extra = vehicle.locate_all(frame, min_area=args.min_area,
                               keep_border=args.keep_border)
    if len(extra) > 1:
        print(f"  ⚠️ 还有 {len(extra)-1} 个红块（面积 {[p.area for p in extra[1:4]]}）—— "
              f"如果有两个红车，'最大的那个'就是个武断的规则")

    if len(frames) > 2 and pose is not None:
        rows = []
        for f in frames[1:]:
            p = vehicle.locate(f, min_area=args.min_area, keep_border=args.keep_border)
            if p is not None:
                rows.append((p.x, p.y, p.heading, p.area))
        if len(rows) > 2:
            a = np.array(rows)
            angles = np.radians(a[:, 2] * 2)
            std = np.degrees(np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * angles)))))) / 2
            print(f"\n=== 静止时的抖动（{len(a)} 帧，车没动）===")
            print(f"  位置 x±{a[:,0].std():.2f}px  y±{a[:,1].std():.2f}px")
            print(f"  轴向 ±{std:.2f}°   面积 中位 {int(np.median(a[:,3]))}px²")
            print(f"  对照导航需求 (位置 ±5px, 朝向 ±15°): "
                  + ("✅ 达标" if a[:,0].std() <= 5 and a[:,1].std() <= 5 and std <= 15
                     else "❌ 不足"))

    try:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        draw = frame.copy()
        for p in extra:
            x, y, w, h = p.bbox
            cv2.rectangle(draw, (x, y), (x + w, y + h), (255, 120, 0), 2)
            rad = np.radians(p.heading)
            length = max(w, h) * 0.7 / 2
            p0 = (int(p.x - length * np.cos(rad)), int(p.y - length * np.sin(rad)))
            p1 = (int(p.x + length * np.cos(rad)), int(p.y + length * np.sin(rad)))
            cv2.line(draw, p0, p1, (255, 120, 0), 2)
            for q in (p0, p1):
                cv2.circle(draw, q, 5, (255, 120, 0), 2)
            cv2.putText(draw, f"({p.x:.0f},{p.y:.0f}) axis {p.heading:.0f}",
                        (max(x - 20, 6), max(y - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 120, 0), 1)
        cv2.imwrite(args.save, draw)
        cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(args.save)), "car_mask.png"),
                    vehicle.red_mask(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)))
        print(f"\n标注图: {os.path.abspath(args.save)}（橙框 = 检测到的车）")
    except OSError as exc:
        print(f"  could not save: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
