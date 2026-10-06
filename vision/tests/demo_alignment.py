"""Side-by-side demo: what the drift compensation actually changes.

Builds a montage from a real frame and a real calibration, so the effect can be
looked at instead of taken on trust:

    python tests/demo_alignment.py --frame data/debug/frame.jpg \
        --config config/parking_spots.json --shift 14,-9 \
        --out data/debug/align_demo.png

Left  : the frame after a known nudge, with the ROIs where they were drawn.
Right : the same nudge, compensated, with the ROIs back on their bays.
Both rows show the red "changed pixels" overlay the occupancy detector computes,
and the bottom row magnifies the same crop 3x so the misalignment is obvious.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from alignment import Aligner                                    # noqa: E402
from detector import DetectorParams, detect_bays                 # noqa: E402
from occupancy import OccupancyDetector, OccupancyParams          # noqa: E402
from spot_config import load_spots                                # noqa: E402

PANEL_BG = (28, 28, 28)
LABEL = (255, 255, 255)


def caption(img: np.ndarray, text: str, sub: str = "") -> np.ndarray:
    h, w = img.shape[:2]
    band = np.full((58, w, 3), PANEL_BG, np.uint8)
    cv2.putText(band, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, LABEL, 1)
    if sub:
        cv2.putText(band, sub, (10, 46), cv2.FONT_HERSHEY_PLAIN, 1.15, (170, 230, 170), 1)
    return np.vstack([band, img])


def annotate(frame: np.ndarray, spots, results) -> np.ndarray:
    """Draw the ROIs, the ids and the changed pixels (same style as main.py)."""
    canvas = frame.copy()
    for s in spots:
        res = results.get(s.id)
        if res is None or res.diff_mask is None:
            continue
        x0, y0, x1, y1 = s.roi
        roi = canvas[y0:y1, x0:x1]
        hot = res.diff_mask > 0
        if hot.any():
            tint = np.zeros_like(roi)
            tint[:] = (0, 0, 255)
            roi[hot] = ((roi[hot].astype(np.uint16) + tint[hot].astype(np.uint16)) // 2
                        ).astype(np.uint8)
    for s in spots:
        res = results.get(s.id)
        state = res.state if res is not None else "?"
        ratio = res.changed_ratio if res is not None else 0.0
        colour = (0, 0, 255) if state == "OCCUPIED" else (0, 220, 0)
        cv2.polylines(canvas, [np.round(s.polygon).astype(np.int32)], True, (0, 255, 255), 2)
        if s.inner_points is not None:
            cv2.polylines(canvas, [np.round(s.inner_points).astype(np.int32)],
                          True, (0, 170, 0), 1)
        cx, cy = s.polygon.mean(axis=0).astype(int)
        cv2.putText(canvas, s.id, (cx - 12, cy + 4), cv2.FONT_HERSHEY_PLAIN, 1.0,
                    (0, 0, 0), 3)
        cv2.putText(canvas, s.id, (cx - 12, cy + 4), cv2.FONT_HERSHEY_PLAIN, 1.0,
                    colour, 1)
    return canvas


def roi_vs_bay_lines(frame: np.ndarray, spots) -> Tuple[float, int]:
    """Mean distance from a bay's left/right edge to the nearest detected divider.

    This is the number that matches the actual complaint. ``changed_ratio`` barely
    moves for a sub-bay nudge on a flat floor (0.004 vs 0.000), so it cannot show
    whether the ROIs still sit on the bays. Measuring the ROIs against the lines the
    detector finds in the frame does show it: a 14 px nudge leaves the ROIs ~14 px
    off the lines, and compensation puts them back at ~1 px.

    Returns (mean distance in px, number of edges measured).
    """
    res = detect_bays(frame, DetectorParams())
    lines = []
    for row in res.plot:
        y0, y1 = row["y"]
        for x in row["positions"]:
            lines.append((float(x), float(y0), float(y1)))

    distances = []
    for spot in spots:
        pts = np.asarray(spot.polygon, dtype=np.float64)
        y_mid = float(pts[:, 1].mean())
        near = [x for x, y0, y1 in lines if y0 - 6 <= y_mid <= y1 + 6]
        if not near:
            continue
        for edge in (pts[:, 0].min(), pts[:, 0].max()):
            distances.append(min(abs(edge - x) for x in near))
    return (float(np.mean(distances)) if distances else float("nan")), len(distances)


def evaluate(frame: np.ndarray, reference: np.ndarray, spots, params: OccupancyParams):
    det = OccupancyDetector(params)
    det.set_reference(reference, spots)
    return det.evaluate(frame, spots, want_debug=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "debug", "frame.jpg"))
    ap.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "config", "parking_spots.json"))
    ap.add_argument("--shift", default="14,-9", help="the nudge to simulate, dx,dy")
    ap.add_argument("--zoom", type=float, default=3.0)
    ap.add_argument("--crop", help="crop as x0,y0,x1,y1 (default: around the first bays)")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "debug", "align_demo.png"))
    args = ap.parse_args()

    reference = cv2.imread(args.frame, cv2.IMREAD_COLOR)
    if reference is None:
        print(f"error: cannot read {args.frame}", file=sys.stderr)
        return 2
    spots = load_spots(args.config, reference.shape[:2])
    dx, dy = (float(v) for v in args.shift.split(","))

    h, w = reference.shape[:2]
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    nudged = cv2.warpAffine(reference, m, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)

    params = OccupancyParams()
    plain = evaluate(nudged, reference, spots, params)

    aligner = Aligner(reference, motion="translation",
                      bay_pitch=float(np.median([min(np.linalg.norm(s.polygon[1] - s.polygon[0]),
                                                     np.linalg.norm(s.polygon[3] - s.polygon[0]))
                                                 for s in spots])))
    res = aligner.estimate(nudged)
    if res.ok:
        fixed_frame = aligner.warp_frame(nudged, res.matrix)
        fixed = evaluate(fixed_frame, reference, spots, params)
    else:
        fixed_frame, fixed = nudged, plain

    def mean_ratio(r):
        return float(np.mean([v.changed_ratio for v in r.values()]))

    left = annotate(nudged, spots, plain)
    right = annotate(fixed_frame, spots, fixed)

    off_plain, n_plain = roi_vs_bay_lines(nudged, spots)
    off_fixed, n_fixed = roi_vs_bay_lines(fixed_frame, spots)
    info = (f"estimated dx={res.dx:+.1f} dy={res.dy:+.1f}  corr={res.correlation:.3f}"
            if res.ok else f"alignment refused: {res.reason}")

    left = caption(
        left,
        f"WITHOUT compensation   ROI edges sit {off_plain:.1f} px off the bay lines",
        f"picture nudged {dx:+.0f},{dy:+.0f} px after the bays were drawn"
        f"   (changed_ratio {mean_ratio(plain):.4f})")
    right = caption(
        right,
        f"WITH compensation      ROI edges sit {off_fixed:.1f} px off the bay lines",
        f"{info}   (changed_ratio {mean_ratio(fixed):.4f})")

    # Crop around one bay and its neighbours: a 14 px offset on an 800 px frame is
    # invisible in a full view, and obvious at 3x on two or three bays.
    if args.crop:
        x0, y0, x1, y1 = (int(v) for v in args.crop.split(","))
    else:
        anchor = spots[0]
        pts = np.asarray(anchor.polygon, dtype=np.float64)
        bw = float(min(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[3] - pts[0])))
        bh = float(max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[3] - pts[0])))
        cx, cy = pts.mean(axis=0)
        x0, y0 = int(max(cx - 1.6 * bw, 0)), int(max(cy - 0.8 * bh, 0))
        x1, y1 = int(min(cx + 1.6 * bw, w)), int(min(cy + 0.8 * bh, h))

    row1 = np.hstack([left, right])
    half = row1.shape[1] // 2
    crop_w, crop_h = max(x1 - x0, 1), max(y1 - y0, 1)
    scale = min(args.zoom, half / float(crop_w))          # never wider than a panel

    linemap = detect_bays(nudged, DetectorParams())

    def zoom(img, title, sub):
        crop = img[y0:y1, x0:x1].copy()
        for row in linemap.plot:
            for x in row["positions"]:
                if x0 <= x <= x1:
                    cv2.line(crop, (int(x - x0), 0), (int(x - x0), crop.shape[0]),
                             (0, 0, 255), 1)
        out = cv2.resize(crop, (int(crop_w * scale), int(crop_h * scale)),
                         interpolation=cv2.INTER_NEAREST)
        if out.shape[1] < half:
            pad = np.full((out.shape[0], half - out.shape[1], 3), PANEL_BG, np.uint8)
            out = np.hstack([out, pad])
        return caption(out, title, sub)

    zl = zoom(nudged, f"zoomed {scale:.1f}x - red line = where the bay line is",
              f"ROI (yellow) is {off_plain:.0f} px off it")
    zr = zoom(fixed_frame, f"zoomed {scale:.1f}x - compensated",
              f"ROI (yellow) is {off_fixed:.0f} px off it")
    montage = np.vstack([row1, np.hstack([zl, zr])])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, montage)
    print(f"wrote {args.out}  ({montage.shape[1]}x{montage.shape[0]})")
    print(f"  without compensation: mean changed_ratio {mean_ratio(plain):.4f}")
    print(f"  with compensation   : mean changed_ratio {mean_ratio(fixed):.4f}")
    print(f"  {info}")
    print(f"  zoom crop: x {x0}..{x1}  y {y0}..{y1}  at {scale:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
