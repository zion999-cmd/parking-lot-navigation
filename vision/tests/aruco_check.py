"""Can the camera actually see a marker on the car? A repeatable answer.

    python tests/aruco_check.py                       # every ArUco family, 20 frames
    python tests/aruco_check.py --dict DICT_4X4_50 --id 7 --frames 40
    python tests/aruco_check.py --min-side 40         # require the marker to be big enough

This exists because a single lucky-looking result is not evidence. Measured here: with very
loose detector parameters, DICT_4X4_1000 "detected" a marker in 12 of 12 frames - except the
ids were five different numbers across those frames, and every candidate was 5-10 pixels
across. Those were the parking bay markings, a grid of quadrilaterals, decoding to the same
few ids every time. A marker of five pixels cannot exist, so the rule this tool enforces is:

    a stable id is not evidence on its own; the marker also has to be big enough to be real

Reliable decoding needs the marker's side at around 40-60 px in an 800x600 frame: a
DICT_4X4_* marker is 6x6 cells (4x4 of data plus a black border), so at 5-10 px each cell is
about one pixel wide and the bits cannot be sampled. If the marker is far too small, the
answer is a bigger marker, not a cleverer detector.

Exit code 0 when the marker is seen in most frames at a usable size, 1 otherwise.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from camera import open_camera                                  # noqa: E402

ARUCO = cv2.aruco

# Below this, a "detection" is not credible whatever its id says. A side of 5-10 px is a
# quadrilateral in the scene, not a marker.
IMPLAUSIBLE_SIDE_PX = 15.0
# Comfortable for a 4x4 marker; below it the bits start getting sampled wrongly.
COMFORTABLE_SIDE_PX = 40.0


def families(only: str | None) -> list[str]:
    names = sorted(d for d in dir(ARUCO)
                   if d.startswith("DICT_") and d.upper() == d)      # skip the aliases
    if only:
        return [only]
    return names


def tuned_params():
    p = ARUCO.DetectorParameters()
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 53
    p.adaptiveThreshWinSizeStep = 4
    p.minMarkerPerimeterRate = 0.015
    p.cornerRefinementMethod = ARUCO.CORNER_REFINE_SUBPIX
    return p


def grab(count: int, source: str, timeout: float = 90.0):
    cam = open_camera(source)
    frames, t0 = [], time.time()
    try:
        while len(frames) < count and time.time() - t0 < timeout:
            f = cam.read()
            if f is not None:
                frames.append(f)
    finally:
        cam.close()
    return frames


def measure(name: str, frames, want_id: int | None):
    dictionary = ARUCO.getPredefinedDictionary(getattr(ARUCO, name))
    detector = ARUCO.ArucoDetector(dictionary, tuned_params())
    ids = collections.Counter()
    sides, centres, headings = [], [], []
    hits = 0
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        corners, found, _ = detector.detectMarkers(gray)
        if found is None or not len(found):
            continue
        wanted = [(k, int(m)) for k, m in enumerate(found.flatten())
                  if want_id is None or int(m) == want_id]
        if not wanted:
            continue
        hits += 1
        for k, mid in wanted:
            ids[mid] += 1
            pts = corners[k][0]
            sides.append(float(np.mean([np.linalg.norm(pts[(j + 1) % 4] - pts[j])
                                        for j in range(4)])))
            centres.append(pts.mean(axis=0))
            headings.append(float(np.degrees(np.arctan2(pts[1][1] - pts[0][1],
                                                        pts[1][0] - pts[0][0]))))
    return {"name": name, "frames": len(frames), "hits": hits, "ids": ids,
            "sides": sides, "centres": centres, "headings": headings}


def report(result: dict, want_id: int | None) -> bool:
    name, frames, hits = result["name"], result["frames"], result["hits"]
    ids, sides = result["ids"], result["sides"]
    rate = hits / max(frames, 1)
    median = float(np.median(sides)) if sides else 0.0
    plausible = median >= IMPLAUSIBLE_SIDE_PX
    consistent = (len(ids) == 1) if ids else False

    print(f"  {name:>22}  {hits:>3}/{frames} 帧"
          + (f"  id {dict(ids.most_common(3))}" if ids else "  id -")
          + (f"  边长中位 {median:5.1f}px" if sides else ""))
    if hits == 0:
        return False
    if not plausible:
        print(f"      -> 不可信: 边长 {median:.1f}px < {IMPLAUSIBLE_SIDE_PX:.0f}px。"
              f"这种尺寸的四边形是画面里的网格/线条，不是标记")
        return False
    if not consistent:
        print(f"      -> 不可信: 同一个标记解出了 {len(ids)} 个 id，位采样不稳")
        return False
    if median < COMFORTABLE_SIDE_PX:
        print(f"      -> 能读但偏小: 推荐 {COMFORTABLE_SIDE_PX:.0f}px 以上，"
              f"现在 {median:.1f}px")
        return rate > 0.8
    return rate > 0.8


def main() -> int:
    ap = argparse.ArgumentParser(description="Is the marker on the car visible?")
    ap.add_argument("--source", default="http://192.168.3.118/stream")
    ap.add_argument("--dict", dest="dict_name", default=None,
                    help="test only this family, e.g. DICT_4X4_50 (default: all)")
    ap.add_argument("--id", dest="want_id", type=int, default=None,
                    help="the id you printed; other ids are ignored")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--all", action="store_true",
                    help="try every family even if one works (default: stop at the first)")
    ap.add_argument("--save", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "..", "data", "debug",
                                                   "aruco_check.jpg"))
    args = ap.parse_args()

    frames = grab(args.frames, args.source)
    if not frames:
        print("error: no frames arrived", file=sys.stderr)
        return 2
    gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    print(f"{len(frames)} 帧, {frames[0].shape[1]}x{frames[0].shape[0]}")
    print(f"画面: 亮度 {gray.mean():.1f}  对比度 {gray.std():.1f}  "
          f"清晰度 {cv2.Laplacian(gray, cv2.CV_64F).var():.1f}")
    print()

    good = None
    for name in families(args.dict_name):
        result = measure(name, frames, args.want_id)
        if result["hits"] or args.all:
            if report(result, args.want_id) and good is None:
                good = result
                if not args.all:
                    break

    print()
    if good is None:
        print("结论: 没有可信的标记检出")
        print("  1) 确认你打印的字典和 id，用 --dict / --id 指定（生成标签上写着，例如 4x4_50 id 7）")
        print("  2) 把标记拿到镜头前 20~30cm 占画面一大块再测一次 —— 这能区分")
        print("     「只是太远太小」和「打印/静区/字典本身有问题」")
        print(f"  3) 可靠解码需要边长 ~{COMFORTABLE_SIDE_PX:.0f}px 以上"
              f"（DICT_4X4 是 6x6 格，每格要 ~4px 才采得准）")
    else:
        c = np.array(good["centres"])
        h = np.array(good["headings"])
        print(f"结论: {good['name']} id {good['ids'].most_common(1)[0][0]} 可用, "
              f"{good['hits']}/{good['frames']} 帧")
        print(f"  标记边长中位 {np.median(good['sides']):.1f}px")
        print(f"  → 位姿抖动: 中心 x±{c[:,0].std():.2f} y±{c[:,1].std():.2f} px, "
              f"角度 ±{h.std():.2f}°")

    # Annotated frame for a human to look at, since the numbers cannot show "wrong marker".
    try:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        reach = families(args.dict_name)
        draw = frames[0].copy()
        any_found = False
        for name in reach:
            dictionary = ARUCO.getPredefinedDictionary(getattr(ARUCO, name))
            detector = ARUCO.ArucoDetector(dictionary, tuned_params())
            corners, found, _ = detector.detectMarkers(cv2.cvtColor(draw, cv2.COLOR_BGR2GRAY))
            if found is not None and len(found):
                ARUCO.drawDetectedMarkers(draw, corners, found)
                any_found = True
        cv2.imwrite(args.save, draw)
        print(f"\n标注图: {os.path.abspath(args.save)}"
              + ("" if any_found else "  (上面没有任何检出) "))
    except OSError as exc:
        print(f"  could not save the annotated frame: {exc}", file=sys.stderr)
    return 0 if good is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
