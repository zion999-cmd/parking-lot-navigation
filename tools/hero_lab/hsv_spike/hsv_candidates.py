#!/usr/bin/env python3
"""Phase 1 of the visual-marker spike: HSV red candidates, nothing else.

The question is deliberately narrow. On the live 800x600 stream, can an HSV red mask plus
area/size/aspect filtering compress each frame into a handful of candidates that reliably
contain the car's red arrow? No shape matching, no template, no tracker, no HOG - those are
later phases, and mixing them in now would make it impossible to say which layer did what.

Everything is written out so the answer can be checked rather than believed:

  candidates.csv   one row per frame per candidate: frame, n, index, bbox, centre, area
  frames.csv       one row per frame: n candidates, and the largest one's geometry
  overlay_XXXX.png every --overlay-every frames, with every candidate drawn
  stdout           the distribution of n, and how much the largest candidate wanders

Deliberately not used: parking_spots.json, vision/data/labels, tools/hero_lab/data. The
measurement below stands on its own frames.

Usage:
    python hsv_candidates.py --frames 240 --out out/phase1
    python hsv_candidates.py --video clip.mp4 --out out/phase1_from_clip
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import statistics
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "config.json")
RELAY = "http://127.0.0.1:8021/stream"


# --------------------------------------------------------------------------- config


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


# --------------------------------------------------------------------------- phase 1


def red_mask(frame: np.ndarray, cfg: dict) -> np.ndarray:
    """Two-arc red mask. OpenCV's H axis wraps, so red lives at both ends of it."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low = cv2.inRange(hsv, (cfg["h1_min"], cfg["s_min"], cfg["v_min"]),
                      (cfg["h1_max"], 255, 255))
    high = cv2.inRange(hsv, (cfg["h2_min"], cfg["s_min"], cfg["v_min"]),
                       (cfg["h2_max"], 255, 255))
    mask = cv2.bitwise_or(low, high)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (cfg["open_kernel"], cfg["open_kernel"]))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (cfg["close_kernel"], cfg["close_kernel"]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    x, y, w, h = cfg["roi"]
    keep = np.zeros_like(mask)
    keep[y:y + h, x:x + w] = mask[y:y + h, x:x + w]
    return keep


def candidates(frame: np.ndarray, cfg: dict) -> tuple[np.ndarray, list[dict]]:
    """Red blobs that survive the geometric filters, largest first."""
    mask = red_mask(frame, cfg)
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    found = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < cfg["min_area"] or area > cfg["max_area"]:
            continue
        if w < cfg["min_width"] or h < cfg["min_height"]:
            continue
        aspect = w / float(h)
        if aspect < cfg["min_aspect"] or aspect > cfg["max_aspect"]:
            continue
        cx, cy = centroids[i]
        found.append({
            "bbox": (x, y, w, h),
            "center_x": round(float(cx), 2),
            "center_y": round(float(cy), 2),
            "area": area,
            "width": w,
            "height": h,
            "aspect": round(aspect, 3),
        })
    found.sort(key=lambda c: c["area"], reverse=True)
    return mask, found


def overlay(frame: np.ndarray, mask: np.ndarray, found: list[dict]) -> np.ndarray:
    """Every candidate gets a box; the largest gets a heavier one and its number."""
    vis = frame.copy()
    tint = np.zeros_like(vis)
    tint[mask > 0] = (0, 0, 255)
    vis = cv2.addWeighted(vis, 1.0, tint, 0.35, 0)

    for i, c in enumerate(found):
        x, y, w, h = c["bbox"]
        colour = (0, 255, 255) if i == 0 else (0, 200, 0)
        thick = 2 if i == 0 else 1
        cv2.rectangle(vis, (x, y), (x + w, y + h), colour, thick)
        cv2.putText(vis, "%d a=%d" % (i, c["area"]), (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
        if i == 0:
            cv2.circle(vis, (int(c["center_x"]), int(c["center_y"])), 4, (0, 255, 255), -1)

    cv2.putText(vis, "HSV candidates: %d" % len(found), (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


# --------------------------------------------------------------------------- frame source


def grab(url: str, timeout: float = 30.0) -> np.ndarray | None:
    """One frame off an MJPEG stream. Used only when there is no --video."""
    host, port, path = _split(url)
    s = socket.create_connection((host, port), timeout=timeout)
    try:
        s.sendall(("GET %s HTTP/1.0\r\nHost: %s\r\n\r\n" % (path, host)).encode())
        s.settimeout(timeout)
        buf = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                return None
            buf += chunk
            a = buf.find(b"\xff\xd8")
            b = buf.find(b"\xff\xd9", a)
            if a >= 0 and b > a:
                return cv2.imdecode(np.frombuffer(buf[a:b + 2], np.uint8), cv2.IMREAD_COLOR)
    finally:
        s.close()


def _split(url: str) -> tuple[str, int, str]:
    from urllib.parse import urlparse
    u = urlparse(url)
    return u.hostname or "127.0.0.1", u.port or 80, u.path or "/"


class Stream:
    """Keeps reading the MJPEG stream and hands back frames as they arrive."""

    def __init__(self, url: str):
        host, port, path = _split(url)
        self.sock = socket.create_connection((host, port), timeout=30)
        self.sock.sendall(("GET %s HTTP/1.0\r\nHost: %s\r\n\r\n" % (path, host)).encode())
        self.sock.settimeout(30)
        self.buf = b""

    def next(self) -> np.ndarray | None:
        while True:
            a = self.buf.find(b"\xff\xd8")
            b = self.buf.find(b"\xff\xd9", a)
            if a >= 0 and b > a:
                jpeg = self.buf[a:b + 2]
                self.buf = self.buf[b + 2:]
                return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            chunk = self.sock.recv(65536)
            if not chunk:
                return None
            self.buf += chunk

    def close(self):
        self.sock.close()


# --------------------------------------------------------------------------- reporting


def summarise(per_frame: list[dict], cfg: dict) -> str:
    counts = [f["n"] for f in per_frame]
    lines = []
    lines.append("frames        %d" % len(per_frame))
    lines.append("candidates/frame  mean %.2f  median %d  max %d"
                 % (statistics.mean(counts), int(statistics.median(counts)), max(counts)))
    hist = {}
    for c in counts:
        hist[c] = hist.get(c, 0) + 1
    lines.append("distribution  " + "  ".join("%d:%d" % (k, hist[k]) for k in sorted(hist)))

    empty = sum(1 for c in counts if c == 0)
    lines.append("frames with zero candidates  %d (%.1f%%)"
                 % (empty, 100.0 * empty / max(1, len(counts))))

    with_largest = [f for f in per_frame if f["n"] > 0]
    if with_largest:
        xs = [f["largest_x"] for f in with_largest]
        ys = [f["largest_y"] for f in with_largest]
        areas = [f["largest_area"] for f in with_largest]
        lines.append("largest candidate  x %.1f..%.1f (spread %.1f)  y %.1f..%.1f (spread %.1f)"
                     % (min(xs), max(xs), max(xs) - min(xs),
                        min(ys), max(ys), max(ys) - min(ys)))
        lines.append("largest candidate  area %.0f..%.0f  median %.0f"
                     % (min(areas), max(areas), statistics.median(areas)))
        lines.append("  (a still scene should show a small spread; a wandering one means the")
        lines.append("   largest red blob is not the arrow, or the mask is unstable)")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=240, help="how many frames to measure")
    ap.add_argument("--url", default=RELAY, help="MJPEG stream (default: the raw PC relay)")
    ap.add_argument("--video", help="an offline video/clip instead of the live stream")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "phase1"))
    ap.add_argument("--overlay-every", type=int, default=20,
                    help="write an overlay image every N frames (0 disables)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)
    print("config: %s" % json.dumps(cfg, sort_keys=True))
    print("source: %s" % (args.video or args.url))

    cap = cv2.VideoCapture(args.video) if args.video else None
    stream = None if args.video else Stream(args.url)

    cand_path = os.path.join(args.out, "candidates.csv")
    frame_path = os.path.join(args.out, "frames.csv")
    per_frame = []
    t_start = time.time()
    frame_no = 0

    with open(cand_path, "w", newline="", encoding="utf-8") as fc, \
            open(frame_path, "w", newline="", encoding="utf-8") as ff:
        wc = csv.writer(fc)
        wf = csv.writer(ff)
        wc.writerow(["frame", "n", "index", "x", "y", "w", "h",
                     "center_x", "center_y", "area", "aspect"])
        wf.writerow(["frame", "n", "largest_x", "largest_y", "largest_area"])

        while frame_no < args.frames:
            frame = None
            if args.video:
                ok, frame = cap.read()
                if not ok:
                    break
            else:
                frame = stream.next()
                if frame is None:
                    print("stream ended at frame %d" % frame_no)
                    break

            mask, found = candidates(frame, cfg)
            largest = found[0] if found else None
            per_frame.append({
                "n": len(found),
                "largest_x": largest["center_x"] if largest else float("nan"),
                "largest_y": largest["center_y"] if largest else float("nan"),
                "largest_area": largest["area"] if largest else 0,
            })
            for i, c in enumerate(found):
                wc.writerow([frame_no, len(found), i, c["bbox"][0], c["bbox"][1],
                             c["bbox"][2], c["bbox"][3], c["center_x"], c["center_y"],
                             c["area"], c["aspect"]])
            wf.writerow([frame_no, len(found),
                         largest["center_x"] if largest else "",
                         largest["center_y"] if largest else "",
                         largest["area"] if largest else 0])

            if args.overlay_every and frame_no % args.overlay_every == 0:
                cv2.imwrite(os.path.join(args.out, "overlay_%04d.png" % frame_no),
                            overlay(frame, mask, found))

            frame_no += 1
            if frame_no % 40 == 0:
                print("  %d/%d frames, %.1f fps so far"
                      % (frame_no, args.frames, frame_no / (time.time() - t_start)))

    if stream:
        stream.close()
    if cap:
        cap.release()

    elapsed = time.time() - t_start
    print()
    print(summarise(per_frame, cfg))
    print()
    print("measured %.1f fps over %d frames (includes JPEG decode)"
          % (frame_no / elapsed if elapsed else 0, frame_no))
    print("wrote %s, %s and the overlays in %s" % (cand_path, frame_path, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
