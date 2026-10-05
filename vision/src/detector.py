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
    # A divider cannot be taller than this fraction of the frame, and a row cannot
    # hold more dividers than could physically fit across the frame. Both are
    # sanity caps for noisy images, where Hough segments chain into one giant
    # cluster (204 "dividers" 1.9 px apart, observed on a noise-only frame).
    max_divider_height_frac: float = 0.6

    # Regularity is the strongest signal a car park has: dividers sit on an
    # even grid. A row is only turned into bays when its dividers fit such a
    # grid; the outliers are dropped instead of producing nonsense bays.
    require_grid: bool = True
    grid_tolerance: float = 0.18    # fraction of the spacing
    min_aligned: int = 3

    orientation: str = "vertical"   # vertical | horizontal | auto
    # Defaults verified against the real sandbox on 2026-10-05:
    #   method=profile, orientation=vertical, band_threshold=0.05
    # finds two rows at y 36..133 and y 197..390 with a consistent pitch of
    # 46-47 px, matching the rows that were visually confirmed in the camera image.
    # "segments" needs the same rows but reads the two edges of each white line as
    # two dividers, so it reports a bogus ~21 px pitch in the lower row, and on a
    # noisy frame it chains 1367 segments into 204 "dividers".
    method: str = "profile"         # profile (energy+autocorr) | segments (Hough)

    # profile method
    band_smooth_sigma: float = 6.0     # smooths the row-energy curve before banding
    # 0.05, not 0.30: with 0.30 the upper row of the real sandbox was missed
    # entirely. Bands are refined later by the pitch fit, so being inclusive here
    # costs little and missing a row costs a whole row of bays.
    band_threshold: float = 0.05       # fraction between mean and max energy
    min_band_height: int = 26
    peak_prominence: float = 0.35      # fraction of the profile's std
    # Periodicity confidence. Measured on 2026-10-05: the normalised
    # autocorrelation peak of the band profile is 0.53-0.67 on the real sandbox
    # and 0.85 on a clean synthetic row, but 0.068-0.128 on a noise-only frame. A
    # threshold of 0.25 therefore sits about a factor of two away from both, and
    # it is what keeps a random image from producing a hundred bogus bays once
    # band_threshold is permissive enough to find real rows.
    autocorr_min: float = 0.25
    # A comb must also explain a decent share of the peaks it was fitted to.
    min_aligned_fraction: float = 0.5


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


def comb_from_spacing(peaks: Sequence[float], spacing: float, tolerance: float,
                      max_fill_fraction: float = 0.4
                      ) -> Tuple[List[float], int, int]:
    """Phase-align a comb of known pitch to the peaks, and fill in the gaps.

    This is the payoff of the bays being evenly spaced: when one divider was not
    detected, the comb still knows where it should have been, so the bay is
    reconstructed instead of being merged with its neighbour.

    The phase is fitted by least squares over the peaks that matched, not just
    taken from whichever peak happened to tie first. Taking a peak as the origin
    left the whole comb shifted by up to half a bay on real images - visible as
    boxes that were "slightly off".

    A comb that would have to invent more than ``max_fill_fraction`` of its teeth
    is rejected outright: at that point the "pitch" is fiction, and returning an
    empty list makes the caller fall back to the raw peaks instead of generating
    bays that do not exist.

    Returns (positions, n_aligned, n_missing); positions == [] means rejected.
    """
    if not peaks or spacing <= 0:
        return list(peaks), 0, 0
    tol = max(1.0, tolerance * spacing)
    xs = np.asarray(sorted(peaks), dtype=np.float64)

    best_aligned, best_origin = -1, float(xs[0])
    for origin in xs:
        n = sum(1 for x in xs
                if abs(((x - origin + spacing / 2.0) % spacing) - spacing / 2.0) <= tol)
        if n > best_aligned:
            best_aligned, best_origin = n, float(origin)
    if best_aligned <= 0:
        return [], 0, 0

    # Least-squares phase over the peaks that matched, iterated once so the
    # matching itself is done with the refined phase.
    origin = best_origin
    for _ in range(2):
        k = np.round((xs - origin) / spacing)
        near = np.abs(xs - (origin + k * spacing)) <= tol
        if not near.any():
            break
        k_near = np.round((xs[near] - origin) / spacing)
        origin = float(np.mean(xs[near] - k_near * spacing))

    k_all = np.round((xs - origin) / spacing)
    aligned_mask = np.abs(xs - (origin + k_all * spacing)) <= tol
    aligned = int(aligned_mask.sum())

    idx = [int(round((x - origin) / spacing)) for x in xs[aligned_mask]]
    if not idx:
        return [], 0, 0
    k0, k1 = min(idx), max(idx)
    if k1 <= k0:
        return [], 0, 0

    total = k1 - k0 + 1
    positions: List[float] = []
    missing = 0
    for k in range(k0, k1 + 1):
        nominal = origin + k * spacing
        near = xs[np.abs(xs - nominal) <= tol]
        if near.size:
            positions.append(float(near.mean()))
        else:
            positions.append(float(nominal))
            missing += 1

    if missing > max_fill_fraction * total:
        return [], aligned, missing
    return positions, aligned, missing


def choose_comb(peaks: Sequence[float], base_spacing: float, min_spacing: float,
                tolerance: float) -> Optional[Tuple[float, List[float], int, int]]:
    """Pick the pitch that best explains the peaks, allowing a coarser harmonic.

    A finer comb can trivially "align" more points, so counting aligned peaks
    alone always drifts towards a pitch that is too small and then invents the
    dividers in between. Reconstructed teeth are therefore charged a penalty, and
    a comb that would have to invent most of itself is discarded.

    Returns (spacing, positions, aligned, missing) or None.
    """
    if not peaks or base_spacing <= 0:
        return None
    best = None
    # Sub-multiples for a missed divider, multiples for the opposite mistake: a
    # white line has two edges, and if Hough returned both, the median gap is half
    # the real bay pitch. Whichever pitch explains the peaks better wins.
    for div in (1, 2, 3, 0.5, 1 / 3.0):
        s = base_spacing / div
        if s < min_spacing:
            continue
        positions, aligned, missing = comb_from_spacing(peaks, s, tolerance)
        if not positions:
            continue
        score = aligned - 0.35 * missing
        key = (score, s)
        if best is None or key > best[0]:
            best = (key, s, positions, aligned, missing)
    if best is None:
        return None
    _, s, positions, aligned, missing = best
    return s, positions, aligned, missing


@dataclass
class DetectorResult:
    polygons: List[List[List[float]]] = field(default_factory=list)
    rows: List[Row] = field(default_factory=list)
    edges: Optional[np.ndarray] = field(default=None, repr=False)
    diagnostics: Dict[str, object] = field(default_factory=dict)
    # Per-band working data, kept so the whole thing can be rendered as text.
    # A text overlay is how alignment gets checked when no one can look at a PNG.
    plot: List[Dict] = field(default_factory=list, repr=False)

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


def _merge_dividers(segs: np.ndarray, p: DetectorParams,
                    frame_w: int = 1, frame_h: int = 1) -> List[Divider]:
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

    # Drop implausibly tall "dividers" - on a noisy frame many short segments chain
    # into one cluster spanning half the image.
    max_h = max(frame_h * p.max_divider_height_frac, p.min_height)
    dividers = [d for d in dividers if (d.y1 - d.y0) <= max_h]
    # And a row cannot contain more dividers than fit across the frame.
    max_per_row = max(int(frame_w / max(p.min_width, 1)), 2)
    if len(dividers) > max_per_row * 8:
        # Far beyond anything a car park could hold: treat as noise.
        return []
    return dividers


def _group_rows(dividers: List[Divider], p: DetectorParams,
                frame_w: int = 0) -> List[Row]:
    """Union dividers that share vertical extent into rows."""
    if not dividers:
        return []
    # A row cannot hold more dividers than could physically fit across the frame.
    # Noise frames otherwise chain thousands of segments into one giant row.
    max_per_row = max(int(frame_w / max(p.min_width, 1)), 4) if frame_w else 10 ** 6
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
        if len(members) > max_per_row:
            continue            # impossible for a parking row: treat as noise
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
    # The profile is only used for the text overlay, but having it for this method
    # too is what makes "are the dividers on the real lines?" answerable in text.
    energy = _vertical_energy(gray)
    dividers = _merge_dividers(segs, p, w, h)
    rows = _group_rows(dividers, p, w)

    polys: List[List[List[float]]] = []
    rejected = 0
    rowinfo = []
    plot: List[Dict] = []

    for row in rows:
        xs = [d.x_at(row.mid_row_y()) for d in row.dividers]
        med, sd = row.spacing()

        # Same trick as the profile method: measure the pitch, then let it
        # complete the row. A divider that was never seen as a segment is still
        # placed by the comb, and the positions are snapped onto the grid, which
        # is what removes the "slightly offset" jitter of individual segments.
        chosen = choose_comb(xs, med, p.min_width, p.grid_tolerance) if p.require_grid else None
        if chosen is not None:
            spacing, kept_positions, aligned, missing = chosen
        else:
            spacing, kept_positions, aligned, missing = med, list(xs), len(xs), 0

        tolerance = max(1.0, p.grid_tolerance * max(spacing, 1.0))

        def divider_for(x: float) -> Optional[Divider]:
            near = [d for d in row.dividers if abs(d.x_at(row.mid_row_y()) - x) <= tolerance]
            if not near:
                return None
            return min(near, key=lambda d: abs(d.x_at(row.mid_row_y()) - x))

        if len(kept_positions) < 2:
            continue

        y0, y1 = float(row.y0), float(row.y1)
        bays_here = 0
        for px, nx in zip(kept_positions, kept_positions[1:]):
            if spacing > 0 and abs((nx - px) - spacing) > 0.4 * spacing:
                rejected += 1        # a gap in the comb: a missed divider
                continue
            # Use the real line where one was detected (keeps the slant), and a
            # straight line at the comb position where one was reconstructed.
            dl, dr = divider_for(px), divider_for(nx)
            tl = dl.x_at(y0) if dl is not None else px
            bl = dl.x_at(y1) if dl is not None else px
            tr = dr.x_at(y0) if dr is not None else nx
            br = dr.x_at(y1) if dr is not None else nx
            quad = [[tl, y0], [tr, y0], [br, y1], [bl, y1]]
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
            "grid_aligned": aligned,
            "grid_filled": missing,
            "dividers_used": len(kept_positions),
        })
        by0, by1 = int(max(y0, 0)), int(min(y1, h))
        if by1 - by0 >= 2:
            plot.append({"y": (by0, by1),
                         "profile": energy[by0:by1, :].mean(axis=0),
                         "peaks": xs, "positions": list(kept_positions),
                         "pitch": spacing,
                         "radius": tolerance})

    return DetectorResult(
        polygons=polys, rows=rows, edges=edges, plot=plot,
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
    # Canny only feeds the debug overlay, but it must be populated or the overlay
    # comes out as a blank camera frame and looks like a detection failure.
    edges = cv2.Canny(gray, p.canny_lo, p.canny_hi)
    bands = _find_bands(energy, p)
    polys: List[List[List[float]]] = []
    rowinfo = []
    plot: List[Dict] = []
    rejected = 0

    for (by0, by1) in bands:
        # Measure the profile over the middle of the band, not all of it. In a
        # slanted row a divider drifts in x across the band, and averaging the
        # whole height smears the peak until the pitch is misread (a slanted 2x5
        # row came back as 2 bays of double width).
        band_h = by1 - by0 + 1
        my0 = by0 + band_h // 4
        my1 = max(by1 - band_h // 4, my0 + 1)
        profile = energy[my0:my1, :].mean(axis=0)

        peaks = _profile_peaks(profile, p)
        if len(peaks) < 2:
            rowinfo.append({"y": [by0, by1], "peaks": len(peaks), "bays": 0,
                            "note": "too few dividers"})
            continue

        spacing, ac = _dominant_period(profile, p.min_width, p.max_width,
                                       p.autocorr_min)
        missing = 0
        aligned = len(peaks)
        chosen = None
        if spacing is not None:
            chosen = choose_comb(peaks, spacing, p.min_width, p.grid_tolerance)
            if chosen is not None and chosen[2] < p.min_aligned_fraction * len(peaks):
                chosen = None            # the comb only explains a few of the peaks
        if chosen is None:
            mid = float(np.median(np.diff(peaks))) if len(peaks) > 1 else 0.0
            alt = choose_comb(peaks, mid, p.min_width, p.grid_tolerance)
            if alt is not None and alt[2] >= p.min_aligned_fraction * len(peaks):
                chosen = alt
        if chosen is not None:
            spacing, kept, aligned, missing = chosen
        elif p.require_grid:
            # No trustworthy periodic structure here. On a noise-only frame this is
            # what stops a hundred bogus bays from being proposed: with
            # band_threshold at 0.05 bands are cheap to find, so the grid is what
            # decides. A row that really is irregular can still be calibrated by
            # hand (n, then g).
            rowinfo.append({"y": [by0, by1], "peaks": len(peaks), "bays": 0,
                            "note": "no periodic grid - skipped"})
            continue
        else:
            # require_grid off: offer the raw peaks so a hand correction can start.
            spacing = float(np.median(np.diff(peaks))) if len(peaks) > 1 else 0.0
            kept = list(peaks)

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
        plot.append({"y": (by0, by1), "profile": profile, "peaks": peaks,
                     "positions": kept, "pitch": spacing or 0.0,
                     "radius": max(1.0, p.grid_tolerance * (spacing or 1.0))})

    return DetectorResult(
        polygons=polys, rows=[], edges=edges, plot=plot,
        diagnostics={
            "orientation": "vertical",
            "bands": len(bands),
            "rows": rowinfo,
            "rejected_bays": rejected,
        })


def render_profile_ascii(result: DetectorResult, cols: int = 118) -> List[str]:
    """Draw each band's edge profile with the fitted comb underneath it.

    This is the alignment check in text form: the '|' row is the measured
    vertical-edge energy, the '^' row marks dividers that matched a real peak,
    '+' marks teeth where no divider was found (reconstructed from the pitch) and
    'o' marks a detected peak the comb ignored. If the '^'/'+' marks do not sit
    under the tall bars, the pitch or the phase is wrong, and that is visible
    without opening an image.
    """
    out: List[str] = []
    for i, row in enumerate(result.plot or []):
        prof = np.asarray(row["profile"], dtype=np.float64)
        if prof.size == 0:
            continue
        y0, y1 = row["y"]
        positions = list(row["positions"])
        radius = float(row.get("radius", 3.0))

        idx = np.linspace(0, prof.size - 1, cols).astype(int)
        sampled = prof[idx]
        base = sampled.min()
        span = max(sampled.max() - base, 1e-6)
        levels = " .:-=+*#%@"

        scale = cols / float(prof.size)
        bars = "".join(levels[int((s - base) / span * (len(levels) - 1))] for s in sampled)
        comb = [" "] * cols
        peaks = row.get("peaks", [])
        for j, x in enumerate(positions):
            c = int(round(x * scale))
            if not (0 <= c < cols):
                continue
            near = any(abs(p - x) <= radius for p in peaks)
            comb[c] = "^" if near else "+"
        for x in peaks:
            c = int(round(x * scale))
            if 0 <= c < cols and comb[c] == " " and not any(
                    abs(x - q) <= radius for q in positions):
                comb[c] = "o"

        hi = sampled.max()
        out.append(f"  band {i}  y={y0}..{y1}  pitch={row.get('pitch', 0):.1f}px  "
                   f"peaks={len(peaks)}  comb={len(positions)}")
        out.append(f"    energy higher->wider      max={hi:.0f}")
        out.append(f"    |{bars}|")
        out.append(f"    |{''.join(comb)}|   ^ matched peak   + reconstructed   o ignored peak")
    return out


def sort_reading_order(polys: List[List[List[float]]]) -> List[List[List[float]]]:
    """Top to bottom, then left to right - the numbering the user expects."""
    def key(poly):
        pts = np.asarray(poly, dtype=np.float64)
        return (round(float(pts[:, 1].mean()) / 60.0), float(pts[:, 0].mean()))
    return sorted(polys, key=key)


def draw_debug(frame: np.ndarray, result: DetectorResult) -> np.ndarray:
    """Overlay what the detector locked onto, for the 'why did it find that' view.

    Must work for BOTH methods. It used to draw only ``result.rows``, which the
    profile method leaves empty, so the overlay came out as a blank camera frame
    and looked like a total detection failure.
    """
    canvas = frame.copy()
    if result.edges is not None:
        canvas[result.edges > 0] = (0, 255, 0)

    for row in result.rows:                      # segments method
        for d in row.dividers:
            cv2.line(canvas,
                     (int(d.x_at(d.y0)), int(d.y0)),
                     (int(d.x_at(d.y1)), int(d.y1)),
                     (0, 165, 255), 2)
        cv2.line(canvas, (0, int(row.y0)), (canvas.shape[1], int(row.y0)),
                 (255, 0, 255), 1)
        cv2.line(canvas, (0, int(row.y1)), (canvas.shape[1], int(row.y1)),
                 (255, 0, 255), 1)

    for row in result.plot:                      # profile method
        y0, y1 = int(row["y"][0]), int(row["y"][1])
        cv2.line(canvas, (0, y0), (canvas.shape[1], y0), (255, 0, 255), 1)
        cv2.line(canvas, (0, y1), (canvas.shape[1], y1), (255, 0, 255), 1)
        radius = float(row.get("radius", 3.0))
        peaks = row.get("peaks", [])
        for x in row["positions"]:
            matched = any(abs(p - x) <= radius for p in peaks)
            colour = (0, 165, 255) if matched else (0, 0, 255)   # orange / red
            cv2.line(canvas, (int(x), y0), (int(x), y1), colour, 2)
        for x in peaks:                          # peaks the comb threw away
            if not any(abs(x - q) <= radius for q in row["positions"]):
                cv2.circle(canvas, (int(x), (y0 + y1) // 2), 4, (255, 255, 0), 1)
    return canvas
