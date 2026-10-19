"""Can the roof arrow be found by template matching, and is the angle usable?

The arrow carries no reliable colour (measured: mean RGB inside 154,134,123 vs
134,139,123 just outside - a 20 level difference in red, nothing a threshold can
trust). Its shape does not move, so this asks the shape question: cut a template out
of one hand-annotated frame, then look for it in the others, at 24 rotations, and
report the score, the position and the angle.

Read-only: it reads saved frames and one annotation, and writes nothing into the
dataset.
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LABELS = os.path.join(ROOT, "vision", "data", "labels")
DEBUG = os.path.join(ROOT, "vision", "data", "debug")


def load_feature(prefix: str = "arrow") -> dict:
    out = None
    with open(os.path.join(LABELS, "annotations.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("kind") == "feature" and rec.get("name") == prefix:
                out = rec
    if out is None:
        raise SystemExit("no feature annotation named %r" % prefix)
    return out


def template_from(rec: dict):
    img = cv2.imread(os.path.join(LABELS, rec["image"]))
    pts = np.array(rec["points"], np.int32)
    hull = cv2.convexHull(pts)
    x, y, w, h = cv2.boundingRect(hull)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [hull], 255)
    # Grey, because colour carries no signal here; the arrow still has an edge.
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Normalise the patch to its own mean so lighting changes do not decide the score.
    patch = grey[y:y + h, x:x + w].astype(np.float32)
    pmask = mask[y:y + h, x:x + w]
    patch = patch - patch[pmask > 0].mean()
    return patch, pmask, (x, y, w, h)


def rotated(patch: np.ndarray, mask: np.ndarray, deg: float):
    h, w = patch.shape
    diag = int(np.ceil(np.hypot(h, w))) + 6
    canvas = np.full((diag, diag), 0, np.float32)
    mcan = np.zeros((diag, diag), np.uint8)
    y0 = (diag - h) // 2
    x0 = (diag - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = patch
    mcan[y0:y0 + h, x0:x0 + w] = mask
    m = cv2.getRotationMatrix2D((diag / 2, diag / 2), deg, 1.0)
    rot = cv2.warpAffine(canvas, m, (diag, diag), flags=cv2.INTER_LINEAR)
    mrot = cv2.warpAffine(mcan, m, (diag, diag), flags=cv2.INTER_NEAREST)
    keep = mrot > 0
    if not keep.any():
        return None, None
    rot = rot.copy()
    rot[~keep] = rot[keep].mean()
    return rot.astype(np.float32), mrot


def search(frame: np.ndarray, patch, mask, step: float = 15.0):
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    best = (-2.0, None, None, None)
    for deg in np.arange(0, 360, step):
        t, tm = rotated(patch, mask, float(deg))
        if t is None or t.shape[0] >= grey.shape[0] or t.shape[1] >= grey.shape[1]:
            continue
        res = cv2.matchTemplate(grey, t, cv2.TM_CCOEFF_NORMED, mask=(tm > 0).astype(np.uint8) * 255)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx > best[0]:
            best = (float(mx), loc, float(deg), (t.shape[1], t.shape[0]))
    score, loc, deg, size = best
    if loc is None:
        return best
    cx = loc[0] + size[0] / 2.0
    cy = loc[1] + size[1] / 2.0
    return score, (cx, cy), deg, size


def main() -> int:
    rec = load_feature()
    patch, mask, bbox = template_from(rec)
    print("template from %s  bbox=%s  %dx%d" % (rec["id"], bbox, bbox[2], bbox[3]))

    frames = [os.path.join(LABELS, rec["image"])]
    for name in ("raw_now.jpg", "raw_now2.jpg", "arrow_check.jpg"):
        p = os.path.join(DEBUG, name)
        if os.path.exists(p):
            frames.append(p)
    seed_dir = os.path.join(LABELS, "seed")
    if os.path.isdir(seed_dir):
        for seed in sorted(os.listdir(seed_dir)):
            if seed.endswith(".jpg"):
                frames.append(os.path.join(seed_dir, seed))

    print("\n%-28s %7s %8s %8s %s" % ("frame", "score", "x", "y", "angle"))
    for path in frames:
        img = cv2.imread(path)
        if img is None:
            continue
        score, centre, deg, _size = search(img, patch, mask)
        tag = "  <-- template source" if os.path.basename(path) == rec["image"] else ""
        if centre is None:
            print("%-28s %7.3f      -       -      -" % (os.path.basename(path)[:28], score))
            continue
        print("%-28s %7.3f %8.1f %8.1f %6.0f°%s"
              % (os.path.basename(path)[:28], score, centre[0], centre[1], deg, tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
