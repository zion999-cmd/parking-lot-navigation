"""Geometric parking bay candidate detection. No AI, no learned model.

The sandbox has one very strong structural feature: many regular, parallel,
evenly spaced bay divider lines. So instead of free-form rectangle detection the
pipeline is:

    grayscale -> CLAHE contrast -> blur -> Canny edges
        -> HoughLinesP segments
        -> keep segments that are close to perpendicular to the row direction
        -> merge collinear-ish segments into "dividers"
        -> group dividers into rows by shared vertical extent
        -> bays are the gaps between consecutive dividers of a row
        -> keep the plausible ones

Working per row is what makes this stable on a regular car park: a missed line
costs one bay, it does not desynchronise the rest. Nothing here is required to be
perfect - the editor is expected to be the final authority, and
``DetectorResult.diagnostics`` says what was found and what was thrown away.

Horizontal rows are handled by rotating the frame 90 degrees and running exactly
the same code, then mapping the polygons back. One algorithm, two orientations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class DetectorParams:
    # contrast / edges
    clahe_clip: float = 2.5
    clahe_grid: int = 8
    blur_ksize: int = 3
    canny_lo: int = 40
    canny_hi: int = 130

    # Hough
    hough_threshold: int = 28
    hough_min_len: int = 26
    hough_max_gap: int = 10

    # divider selection
    max_angle_deg: float = 32.0     # measured from perpendicular to the row
    x_tolerance: float = 14.0       # px: how close two segments must be to merge
    y_overlap_frac: float = 0.40    # px: how much vertical extent two dividers must share
    min_row_members: int = 2

    # bay plausibility. These are deliberately permissive: their job is to throw
    # out absurd shapes (a 3 px sliver, a 40:1 stripe), not to decide what a real
    # bay looks like. min_aspect 0.35 once silently discarded every 44x130 bay in
    # a synthetic row that had been detected perfectly.
    min_width: int = 14
    max_width: int = 400
    min_height: int = 20
    min_aspect: float = 0.15        # width / height
    max_aspect: float = 4.5

    # Regularity is the strongest signal a car park has: dividers sit on an
    # even grid. A row is only turned into bays when its dividers fit such a
    # grid; the outliers are dropped instead of producing nonsense bays.
    require_grid: bool = True
    grid_tolerance: float = 0.18    # fraction of the spacing
    min_aligned: int = 3

    orientation: str = "auto"       # auto | vertical | horizontal
    method: str = "profile"         # profile (energy + autocorrelation) | segments (Hough)

    # profile method
    band_smooth_sigma: float = 6.0     # smooths the row-energy curve before banding
    band_threshold: float = 0.30       # fraction between mean and max energy
    min_band_height: int = 26
    peak_prominence: float = 0.35      # fraction of the profile's std
    autocorr_min: float = 0.12         # minimum normalised autocorrelation peak


@dataclass
class Divider:
    """One bay divider line: x = a * y + b, valid for y in [y0, y1]."""

    a: float
    b: float
    y0: float
    y1: float
    members: int = 1
    strength: float = 0.0

    def x_at(self, y: float) -> float:
        return self.a * y + self.b

    def mid_y(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass
class Row:
    dividers: List[Divider]
    y0: float
    y1: float

    def mid_row_y(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def spacing(self) -> Tuple[float, float]:
        xs = sorted(d.x_at(self.mid_row_y()) for d in self.dividers)
        if len(xs) < 2:
            return 0.0, 0.0
        gaps = np.diff(xs)
        return float(np.median(gaps)), float(np.std(gaps))


@dataclass
class GridFit:
    """A regular comb fitted to divider positions: x ~ origin + k * spacing."""

    spacing: float
    origin: float
    kept: List[int]             # indices into the row's divider list
    dropped: List[int]
    score: float                # fraction of dividers that fit


def fit_grid(xs: Sequence[float], min_spacing: float, tolerance: float) -> Optional[GridFit]:
    """Find the even spacing that explains the most divider positions.

    This is the 1-D version of "the bays are all the same width": try every
    plausible spacing (including a gap divided by 2 or 3, because one or two
    dividers can be missed), phase-align a comb to the data, and keep the comb
    that covers the most dividers. Outliers fall between the teeth and are
    reported as dropped rather than being turned into bogus bays.
    """
    if len(xs) < 3:
        return None
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    sorted_x = [float(xs[i]) for i in order]

    gaps = {round(b - a, 2) for a, b in zip(sorted_x, sorted_x[1:])
            if b - a >= min_spacing}
    if not gaps:
        return None
    candidates = set()
    for g in gaps:
        for k in (1, 2, 3):
            s = g / k
            if s >= min_spacing:
                candidates.add(round(s, 2))
    if not candidates:
        return None

    best: Optional[Tuple[float, float, List[int]]] = None
    for s in sorted(candidates):
        tol = max(1.0, tolerance * s)
        for origin in sorted_x:
            kept = [i for i, x in enumerate(sorted_x)
                    if abs(((x - origin + s / 2.0) % s) - s / 2.0) <= tol]
            # Prefer a comb that covers more dividers, then a wider spacing.
            key = (len(kept), s)
            if best is None or key > (len(best[2]), best[0]):
                best = (s, origin, kept)
    if best is None:
        return None

    s, origin, kept = best
    # Re-derive the grid from the surviving dividers so the spacing is not
    # anchored to one possibly-noisy line.
    if len(kept) >= 2:
        seq = [sorted_x[i] for i in kept]
        idx = np.arange(len(seq))
        spacing, origin = np.polyfit(idx, seq, 1)
        spacing = float(spacing)
        origin = float(origin)

    tol = max(1.0, tolerance * spacing)
    kept_final = [i for i, x in enumerate(sorted_x)
                  if abs(((x - origin + spacing / 2.0) % spacing) - spacing / 2.0) <= tol]
    dropped = [i for i in range(len(sorted_x)) if i not in set(kept_final)]
    return GridFit(spacing=spacing, origin=origin, kept=kept_final,
                   dropped=dropped, score=len(kept_final) / len(sorted_x))


def comb_from_spacing(peaks: Sequence[float], spacing: float, tolerance: float
                      ) -> Tuple[List[float], int, int]:
    """Phase-align a comb of known pitch to the peaks, and fill in the gaps.

    This is the payoff of the bays being evenly spaced: when one divider was not
    detected, the comb still knows where it should have been, so the bay is
    reconstructed instead of being merged with its neighbour.

    Returns (positions, n_aligned, n_missing).
    """
    if not peaks or spacing <= 0:
        return list(peaks), 0, 0
    tol = max(1.0, tolerance * spacing)
    best_aligned, best_origin = -1, peaks[0]
    for origin in peaks:
        n = sum(1 for x in peaks
                if abs(((x - origin + spacing / 2.0) % spacing) - spacing / 2.0) <= tol)
        if n > best_aligned:
            best_aligned, best_origin = n, origin

    lo, hi = min(peaks), max(peaks)
    xs = np.asarray(sorted(peaks), dtype=np.float64)

    # Derive the comb indices from the peaks themselves rather than from
    # (hi-lo)/spacing: peak positions carry ~0.5 px of jitter, and a naive floor
    # would decide the outermost divider is one tooth short and silently drop the
    # last bay of the row.
    idx = [int(round((x - best_origin) / spacing)) for x in xs]
    k0, k1 = min(idx), max(idx)
    if k1 <= k0:
        return list(peaks), best_aligned, 0

    positions: List[float] = []
    missing = 0
    for k in range(k0, k1 + 1):
        nominal = best_origin + k * spacing
        near = xs[np.abs(xs - nominal) <= tol]
        if near.size:
            positions.append(float(near.mean()))
        else:
            positions.append(float(nominal))
            missing += 1
    return positions, best_aligned, missing


@dataclass
class DetectorResult:
    polygons: List[List[List[float]]] = field(default_factory=list)
    rows: List[Row] = field(default_factory=list)
    edges: Optional[np.ndarray] = field(default=None, repr=False)
    diagnostics: Dict[str, object] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.polygons)


def _preprocess(frame: np.ndarray, p: DetectorParams) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=p.clahe_clip,
                            tileGridSize=(p.clahe_grid, p.clahe_grid))
    gray = clahe.apply(gray)
    k = p.blur_ksize if p.blur_ksize % 2 == 1 else p.blur_ksize + 1
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    return gray


def _edges_and_segments(gray: np.ndarray, p: DetectorParams) -> Tuple[np.ndarray, np.ndarray]:
    edges = cv2.Canny(gray, p.canny_lo, p.canny_hi)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=p.hough_threshold,
                            minLineLength=p.hough_min_len, maxLineGap=p.hough_max_gap)
    if lines is None:
        return edges, np.zeros((0, 4), dtype=np.float64)
    return edges, lines.reshape(-1, 4).astype(np.float64)


def _is_perpendicular(x1, y1, x2, y2, max_angle_deg: float) -> bool:
    """True when the segment is closer to vertical than max_angle_deg."""
    dx, dy = x2 - x1, y2 - y1
    if abs(dy) < 1e-6:
        return False
    ang = abs(math.degrees(math.atan2(dx, dy)))     # 0 = vertical
    ang = min(ang, 180 - ang)
    return ang <= max_angle_deg


def _merge_dividers(segs: np.ndarray, p: DetectorParams) -> List[Divider]:
    """Group near-vertical segments into one line per divider."""
    cands = []
    for x1, y1, x2, y2 in segs:
        if not _is_perpendicular(x1, y1, x2, y2, p.max_angle_deg):
            continue
        if y2 < y1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        length = math.hypot(x2 - x1, y2 - y1)
        cands.append((x1, y1, x2, y2, length))

    # Longest first, so a long line absorbs the short fragments that belong to it.
    cands.sort(key=lambda c: -c[4])
    clusters: List[List[Tuple[float, float, float, float, float]]] = []
    for c in cands:
        cx = (c[0] + c[2]) / 2.0
        placed = False
        for cl in clusters:
            ref = cl[0]
            rx = (ref[0] + ref[2]) / 2.0
            oy0, oy1 = max(ref[1], c[1]), min(ref[3], c[3])
            overlap = max(0.0, oy1 - oy0)
            shorter = min(ref[3] - ref[1], c[3] - c[1])
            if abs(cx - rx) <= p.x_tolerance and shorter > 0 and overlap / shorter >= p.y_overlap_frac:
                cl.append(c)
                placed = True
                break
        if not placed:
            clusters.append([c])

    dividers: List[Divider] = []
    for cl in clusters:
        ys, xs, w = [], [], []
        for x1, y1, x2, y2, length in cl:
            ys.extend([y1, y2])
            xs.extend([x1, x2])
            w.append(length)
        ys_a, xs_a = np.asarray(ys), np.asarray(xs)
        if ys_a.max() - ys_a.min() < 5:
            continue
        a, b = np.polyfit(ys_a, xs_a, 1)
        dividers.append(Divider(a=float(a), b=float(b),
                                y0=float(ys_a.min()), y1=float(ys_a.max()),
                                members=len(cl), strength=float(sum(w))))
    return dividers


def _group_rows(dividers: List[Divider], p: DetectorParams) -> List[Row]:
    """Union dividers that share vertical extent into rows."""
    if not dividers:
        return []
    parent = list(range(len(dividers)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(dividers)):
        for j in range(i + 1, len(dividers)):
            di, dj = dividers[i], dividers[j]
            oy0, oy1 = max(di.y0, dj.y0), min(di.y1, dj.y1)
            overlap = max(0.0, oy1 - oy0)
            shorter = min(di.y1 - di.y0, dj.y1 - dj.y0)
            if shorter > 0 and overlap / shorter >= p.y_overlap_frac:
                union(i, j)

    buckets: Dict[int, List[Divider]] = {}
    for i, d in enumerate(dividers):
        buckets.setdefault(find(i), []).append(d)

    rows: List[Row] = []
    for members in buckets.values():
        if len(members) < p.min_row_members:
            continue
        # Use the median extent of the row so one stray short line cannot skew it.
        y0 = float(np.median([m.y0 for m in members]))
        y1 = float(np.median([m.y1 for m in members]))
        if y1 - y0 < p.min_height:
            continue
        members.sort(key=lambda m: m.x_at((y0 + y1) / 2))
        rows.append(Row(dividers=members, y0=y0, y1=y1))
    rows.sort(key=lambda r: (r.y0, min(d.x_at(r.mid_row_y()) for d in r.dividers)))
    return rows


def _row_quad(left: Divider, right: Divider, row: Row) -> List[List[float]]:
    y0, y1 = row.y0, row.y1
    return [
        [left.x_at(y0), y0],
        [right.x_at(y0), y0],
        [right.x_at(y1), y1],
        [left.x_at(y1), y1],
    ]


def _plausible(quad: Sequence[Sequence[float]], p: DetectorParams,
               frame_w: int, frame_h: int) -> bool:
    pts = np.asarray(quad, dtype=np.float64)
    if not (0 <= pts[:, 0].min() and pts[:, 0].max() <= frame_w
            and 0 <= pts[:, 1].min() and pts[:, 1].max() <= frame_h):
        return False
    area = abs(cv2.contourArea(pts.astype(np.float32)))
    if area <= 0:
        return False
    w = float(np.linalg.norm(pts[1] - pts[0]))
    h = float(np.linalg.norm(pts[3] - pts[0]))
    if not (p.min_width <= w <= p.max_width) or h < p.min_height:
        return False
    aspect = w / max(h, 1e-6)
    return p.min_aspect <= aspect <= p.max_aspect


def _detect_vertical(frame: np.ndarray, p: DetectorParams) -> DetectorResult:
    h, w = frame.shape[:2]
    gray = _preprocess(frame, p)
    edges, segs = _edges_and_segments(gray, p)
    dividers = _merge_dividers(segs, p)
    rows = _group_rows(dividers, p)

    polys: List[List[List[float]]] = []
    rejected = 0
    rowinfo = []

    for row in rows:
        xs = [d.x_at(row.mid_row_y()) for d in row.dividers]
        med, sd = row.spacing()

        fit = fit_grid(xs, p.min_width, p.grid_tolerance) if p.require_grid else None
        if fit is not None and len(fit.kept) >= p.min_aligned:
            kept = sorted((row.dividers[i] for i in fit.kept),
                          key=lambda d: d.x_at(row.mid_row_y()))
            spacing = fit.spacing
        else:
            # Not regular enough to trust a grid: fall back to every divider, so
            # a genuinely irregular bay can still be picked up by hand.
            kept = list(row.dividers)
            spacing = med

        if len(kept) < 2:
            continue

        # Recompute the row extent from the surviving lines.
        y0 = float(np.median([d.y0 for d in kept]))
        y1 = float(np.median([d.y1 for d in kept]))
        row_kept = Row(dividers=kept, y0=y0, y1=y1)

        bays_here = 0
        for left, right in zip(kept, kept[1:]):
            gap = right.x_at(row_kept.mid_row_y()) - left.x_at(row_kept.mid_row_y())
            if spacing > 0 and abs(gap - spacing) > 0.4 * spacing:
                # A gap in the comb, i.e. a missed divider. A double-width bay is
                # almost certainly a false positive, so skip it and say so.
                rejected += 1
                continue
            quad = _row_quad(left, right, row_kept)
            if _plausible(quad, p, w, h):
                polys.append(quad)
                bays_here += 1
            else:
                rejected += 1

        rowinfo.append({
            "y": [round(y0, 1), round(y1, 1)],
            "dividers": len(row.dividers),
            "bays": bays_here,
            "spacing_median": round(med, 1),
            "spacing_std": round(sd, 1),
            "grid_spacing": round(spacing, 1),
            "grid_kept": len(kept),
            "grid_dropped": len(row.dividers) - len(kept),
            "grid_score": round((len(kept) / len(row.dividers)) if row.dividers else 0.0, 2),
        })

    return DetectorResult(
        polygons=polys, rows=rows, edges=edges,
        diagnostics={
            "orientation": "vertical",
            "segments": int(len(segs)),
            "dividers": len(dividers),
            "rows": rowinfo,
            "rejected_bays": rejected,
        })


def _rotate_back(quads: List[List[List[float]]], orig_h: int) -> List[List[List[float]]]:
    """Map polygons found in a 90-degree-rotated frame back to the original."""
    out = []
    for quad in quads:
        out.append([[float(y), float(orig_h - 1 - x)] for x, y in quad])
    return out


def detect_bays(frame: np.ndarray, params: Optional[DetectorParams] = None
                ) -> DetectorResult:
    """Find parking bay candidates. Never raises on a hard image; returns empty."""
    p = params or DetectorParams()
    if frame is None or frame.size == 0:
        return DetectorResult(diagnostics={"error": "empty frame"})

    if p.orientation == "vertical":
        return _run(frame, p, "vertical")
    if p.orientation == "horizontal":
        rot = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        res = _run(rot, p, "vertical")
        res.polygons = _rotate_back(res.polygons, frame.shape[0])
        res.diagnostics["orientation"] = "horizontal"
        return res

    # auto: run both orientations and keep whichever produced more candidates
    vert = _run(frame, p, "vertical")
    rot = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    horiz = _run(rot, p, "vertical")
    horiz.polygons = _rotate_back(horiz.polygons, frame.shape[0])
    horiz.diagnostics["orientation"] = "horizontal"
    vert.diagnostics["auto_considered"] = {"vertical": vert.count,
                                           "horizontal": horiz.count}
    if horiz.count > vert.count:
        return horiz
    return vert


def _run(frame: np.ndarray, p: DetectorParams, orientation: str) -> DetectorResult:
    """Run the selected method, falling back to the other one if it finds nothing.

    A detector is allowed to be wrong; it is not allowed to take the tool down
    with it. If one method returns nothing, the other gets a turn.
    """
    order = ("profile", "segments") if p.method != "segments" else ("segments", "profile")
    first: Optional[DetectorResult] = None
    for name in order:
        res = (_detect_profile(frame, p) if name == "profile"
               else _detect_vertical(frame, p))
        res.diagnostics["method"] = name
        if first is None:
            first = res
        if res.count > 0:
            if first is not res:
                first.diagnostics["fallback_from"] = order[0]
            return res
    return first if first is not None else DetectorResult(
        diagnostics={"error": "no candidates"})


# --------------------------------------------------------------------------- #
# Profile method: vertical-edge energy -> bands -> periodic peaks
# --------------------------------------------------------------------------- #

def _vertical_energy(gray: np.ndarray) -> np.ndarray:
    """|d/dx| - how much of a vertical edge passes through each pixel."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    return np.abs(gx)


def _smooth1d(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return a
    return cv2.GaussianBlur(a.reshape(1, -1), (0, 0), sigma)[0]


def _find_bands(energy: np.ndarray, p: DetectorParams) -> List[Tuple[int, int]]:
    """Horizontal bands where vertical edges are dense: one per row of bays."""
    rows = energy.mean(axis=1)
    rows = _smooth1d(rows, p.band_smooth_sigma)
    thr = rows.mean() + p.band_threshold * (rows.max() - rows.mean())
    mask = rows > thr

    bands: List[Tuple[int, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        if j - i + 1 >= p.min_band_height:
            bands.append((i, j))
        i = j + 1
    return bands


def _dominant_period(profile: np.ndarray, min_spacing: float, max_spacing: float,
                     min_corr: float) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """Autocorrelation of the (detrended) profile -> the bay pitch, in pixels."""
    base = _smooth1d(profile, max(min_spacing * 0.75, 8.0))
    d = np.clip(profile - base, 0.0, None).astype(np.float64)
    if d.sum() <= 0:
        return None, None
    d = d - d.mean()
    n = len(d)
    f = np.fft.rfft(d, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    if ac[0] <= 0:
        return None, None
    ac = ac / ac[0]

    lo = int(max(min_spacing, 3))
    hi = int(min(max_spacing, n // 2))
    if hi <= lo + 2:
        return None, ac
    seg = ac[lo:hi]
    best = int(np.argmax(seg)) + lo
    if ac[best] < min_corr:
        return None, ac
    # Sub-pixel: use the autocorrelation centroid around the peak.
    w = 2
    a, b = max(best - w, 1), min(best + w + 1, n)
    weights = np.clip(ac[a:b], 0, None)
    if weights.sum() > 0:
        best = float((np.arange(a, b) * weights).sum() / weights.sum())
    return float(best), ac


def _profile_peaks(profile: np.ndarray, p: DetectorParams) -> List[float]:
    """Sub-pixel x of each divider: local maxima of the detrended profile."""
    base = _smooth1d(profile, max(p.min_width * 0.75, 8.0))
    d = np.clip(profile - base, 0.0, None)
    if d.max() <= 0:
        return []
    thr = d.mean() + p.peak_prominence * d.std()
    peaks: List[float] = []
    for x in range(1, len(d) - 1):
        if d[x] >= d[x - 1] and d[x] > d[x + 1] and d[x] > thr:
            peaks.append(float(x))
    # Merge peaks closer than half a bay, keeping a weighted centroid.
    merged: List[float] = []
    for x in peaks:
        if merged and x - merged[-1] < p.min_width * 0.5:
            merged[-1] = 0.5 * (merged[-1] + x)
        else:
            merged.append(x)
    return merged


def _refine_x(energy: np.ndarray, x: float, y: int, search: int) -> float:
    """Snap x to the strongest vertical edge nearby on this row."""
    lo = max(int(x) - search, 0)
    hi = min(int(x) + search + 1, energy.shape[1])
    if hi <= lo:
        return x
    row = energy[y, lo:hi]
    return float(lo + int(np.argmax(row)))


def _detect_profile(frame: np.ndarray, p: DetectorParams) -> DetectorResult:
    h, w = frame.shape[:2]
    gray = _preprocess(frame, p)
    energy = _vertical_energy(gray)

    bands = _find_bands(energy, p)
    polys: List[List[List[float]]] = []
    rowinfo = []
    rejected = 0

    for (by0, by1) in bands:
        band = energy[by0:by1 + 1, :]
        profile = band.mean(axis=0)

        peaks = _profile_peaks(profile, p)
        if len(peaks) < 2:
            rowinfo.append({"y": [by0, by1], "peaks": len(peaks), "bays": 0,
                            "note": "too few dividers"})
            continue

        spacing, _ = _dominant_period(profile, p.min_width, p.max_width,
                                      p.autocorr_min)
        missing = 0
        aligned = len(peaks)
        if spacing is not None:
            # A missed divider can make autocorrelation lock onto 2x or 3x the
            # real pitch. Try the sub-multiples and keep whichever explains the
            # most peaks; on a tie prefer the larger spacing so we do not invent
            # dividers that were never there.
            best = None
            for div in (1, 2, 3):
                s = spacing / div
                if s < p.min_width:
                    continue
                pos, al, mi = comb_from_spacing(peaks, s, p.grid_tolerance)
                key = (al, s)
                if best is None or key > best[0]:
                    best = (key, s, pos, al, mi)
            if best is None:
                spacing = None
            else:
                _, spacing, kept, aligned, missing = best
        if spacing is None:
            fit = fit_grid(peaks, p.min_width, p.grid_tolerance)
            if fit is not None and len(fit.kept) >= p.min_aligned:
                kept = sorted(peaks[i] for i in fit.kept)
                spacing = fit.spacing
                aligned = len(fit.kept)
            else:
                kept, spacing = peaks, float(np.median(np.diff(peaks)))

        y0, y1 = by0 + 2, by1 - 2
        if y1 - y0 < p.min_height:
            y0, y1 = by0, by1

        bays_here = 0
        for xl, xr in zip(kept, kept[1:]):
            if spacing > 0 and abs((xr - xl) - spacing) > 0.4 * spacing:
                rejected += 1        # a gap in the comb: a missed divider
                continue
            # Follow the real lines instead of assuming perfectly vertical ones.
            tl = _refine_x(energy, xl, y0, max(3, int(p.min_width * 0.25)))
            tr = _refine_x(energy, xr, y0, max(3, int(p.min_width * 0.25)))
            bl = _refine_x(energy, xl, y1, max(3, int(p.min_width * 0.25)))
            br = _refine_x(energy, xr, y1, max(3, int(p.min_width * 0.25)))
            quad = [[tl, float(y0)], [tr, float(y0)], [br, float(y1)], [bl, float(y1)]]
            if _plausible(quad, p, w, h):
                polys.append(quad)
                bays_here += 1
            else:
                rejected += 1

        rowinfo.append({
            "y": [by0, by1],
            "peaks": len(peaks),
            "grid_spacing": round(spacing, 1) if spacing else None,
            "grid_aligned": aligned,
            "grid_filled": missing,
            "dividers_used": len(kept),
            "bays": bays_here,
        })

    return DetectorResult(
        polygons=polys, rows=[], edges=None,
        diagnostics={
            "orientation": "vertical",
            "bands": len(bands),
            "rows": rowinfo,
            "rejected_bays": rejected,
        })


def sort_reading_order(polys: List[List[List[float]]]) -> List[List[List[float]]]:
    """Top to bottom, then left to right - the numbering the user expects."""
    def key(poly):
        pts = np.asarray(poly, dtype=np.float64)
        return (round(float(pts[:, 1].mean()) / 60.0), float(pts[:, 0].mean()))
    return sorted(polys, key=key)


def draw_debug(frame: np.ndarray, result: DetectorResult) -> np.ndarray:
    """Overlay the detected line structure, for the 'why did it find that' view."""
    canvas = frame.copy()
    if result.edges is not None:
        canvas[result.edges > 0] = (0, 255, 0)
    for row in result.rows:
        for d in row.dividers:
            cv2.line(canvas,
                     (int(d.x_at(d.y0)), int(d.y0)),
                     (int(d.x_at(d.y1)), int(d.y1)),
                     (0, 165, 255), 2)
        cv2.line(canvas, (0, int(row.y0)), (canvas.shape[1], int(row.y0)),
                 (255, 0, 255), 1)
        cv2.line(canvas, (0, int(row.y1)), (canvas.shape[1], int(row.y1)),
                 (255, 0, 255), 1)
    return canvas
