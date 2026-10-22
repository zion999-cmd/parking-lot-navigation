"""A one-file annotation server: live frame in, car + heading out.

    python tools/label_server.py            # then open http://127.0.0.1:8030

It holds one connection to the camera stream and serves the latest frame. Saving writes the JPEG
and one line of JSON beside it, nothing else. No recording step, no frame extraction step, no
intermediate format - if you can see the car on the page you can label it.

    GET  /                 the page
    GET  /snapshot         the current frame as JPEG (live), or ?from=<name> for a saved one
    GET  /frames           frames available to label: live, plus the seed images on disk
    POST /save             {quad, tip, status, notes, from} -> writes the image and the label
    GET  /list             everything labelled so far, newest first
    GET  /image/<id>       the saved JPEG behind a label

Notes that matter:

  * The camera gives each client a share of its frames, so while this page is open the vision
    service sees fewer. For a labelling session that is fine, and `./start.sh stop vision` gives
    full rate if you want it.
  * Coordinates on the page are image pixels, scaled from whatever the browser displayed, so the
    numbers saved are directly comparable with the ones in the contract (origin top left).
  * The heading is derived from the quad, not clicked: corner 1 is the car's RIGHT-FRONT corner and
    the rest follow clockwise, so p0->p1 runs along the car's side pointing rearward and the front
    is the other way. The page still allows a click to override, and records which happened as
    tip_source, so the two can be compared later - a systematic disagreement would mean the
    convention is being applied wrongly somewhere, which is worth knowing before the labels are
    trusted.
  * Saved under vision/data/labels/, which is not gitignored: labels are work. Nothing here
    deletes anything - the tool only appends. If a test needs to write, point it at a throwaway
    directory with --labels, because the two annotation images lost on 2026-10-17 were deleted by a
    cleanup command that globbed ann-*.jpg instead of removing the one file it had just written.
    The quads survived in annotations.jsonl; the pictures did not, and they came from the live
    stream, so they are gone.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "vision", "src"))

LABELS = os.path.join(ROOT, "vision", "data", "labels")
SEED = os.path.join(LABELS, "seed")
ANNOTATIONS = os.path.join(LABELS, "annotations.jsonl")
UI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_ui.html")

# Largest accepted POST body. A quad plus notes is well under a kilobyte; 64 KiB leaves
# room for a long note without letting an unauthenticated client claim arbitrary memory.
MAX_LABEL_BYTES = 64 * 1024

DEFAULT_STREAM = "http://192.168.3.118/stream"


class Frames:
    """One background reader holding the latest frame; None when the camera is not answering."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.latest = None
        self.stamp = 0.0
        self.error = ""
        self._lock = threading.Lock()
        self._stop = False
        threading.Thread(target=self._run, name="frames", daemon=True).start()

    def _run(self) -> None:
        from camera import open_camera
        while not self._stop:
            cam = None
            try:
                cam = open_camera(self.source)
                while not self._stop:
                    frame = cam.read()
                    if frame is None:
                        time.sleep(0.05)
                        continue
                    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    if ok:
                        with self._lock:
                            self.latest, self.stamp, self.error = buf.tobytes(), time.time(), ""
            except Exception as exc:
                with self._lock:
                    self.error = str(exc)
                time.sleep(1.0)
            finally:
                if cam is not None:
                    try:
                        cam.close()
                    except Exception:
                        pass

    def current(self):
        with self._lock:
            return self.latest, self.stamp, self.error


def read_annotations() -> list:
    if not os.path.exists(ANNOTATIONS):
        return []
    out = []
    with open(ANNOTATIONS, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    return out


def seed_names() -> list:
    if not os.path.isdir(SEED):
        return []
    # Photos only: the seed folder also holds turned_mask.png and turned_paint.png, which are
    # masks this project drew, not pictures of the car - labelling one of those would be labelling
    # our own output.
    return sorted(n for n in os.listdir(SEED) if n.lower().endswith((".jpg", ".jpeg")))


class Handler(BaseHTTPRequestHandler):
    frames: Frames = None                                            # set in main()

    def log_message(self, fmt, *args):                               # quieter log
        if "snapshot" not in (args[0] if args else ""):
            sys.stderr.write("  %s\n" % (fmt % args))

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):                                                # noqa: N802
        path, _, query = self.path.partition("?")
        if path in ("/", "/index.html"):
            if not os.path.exists(UI):
                return self._send(500, b"label_ui.html is missing", "text/plain")
            with open(UI, "rb") as fh:
                return self._send(200, fh.read(), "text/html; charset=utf-8")

        if path == "/frames":
            return self._json(200, {"live": self.frames.source, "seed": seed_names(),
                                    "error": self.frames.current()[2]})

        if path == "/snapshot":
            want = ""
            for part in query.split("&"):
                if part.startswith("from="):
                    want = part[5:]
            if want:                                                 # a saved image on disk
                safe = os.path.basename(want)
                for base in (SEED, LABELS):
                    candidate = os.path.join(base, safe)
                    if os.path.exists(candidate):
                        with open(candidate, "rb") as fh:
                            ctype = "image/png" if safe.lower().endswith(".png") else "image/jpeg"
                            return self._send(200, fh.read(), ctype)
                return self._send(404, b"no such image", "text/plain")
            data, stamp, error = self.frames.current()
            if data is None:
                return self._send(503, (error or "no frame yet").encode("utf-8"), "text/plain")
            return self._send(200, data, "image/jpeg")

        if path == "/boxes":
            # The HOG dataset's own index, so the page can list what hero mode has saved.
            hero = os.path.join(ROOT, "tools", "hero_lab", "data", "boxes.jsonl")
            rows = []
            if os.path.exists(hero):
                with open(hero, encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            try:
                                rows.append(json.loads(line))
                            except ValueError:
                                pass
            return self._json(200, {"boxes": list(reversed(rows))})

        if path == "/hsv_calib":
            calib = os.path.join(ROOT, "vision", "data", "hsv_marker", "calibration.jsonl")
            rows = []
            if os.path.exists(calib):
                with open(calib, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rows.append(json.loads(line))
                        except ValueError:
                            continue
            return self._json(200, {"samples": rows[-200:], "total": len(rows)})

        if path == "/list":
            return self._json(200, {"labels": list(reversed(read_annotations()))})

        if path.startswith("/image/"):
            name = os.path.basename(path[len("/image/"):])
            candidate = os.path.join(LABELS, name)
            if os.path.exists(candidate):
                with open(candidate, "rb") as fh:
                    return self._send(200, fh.read(), "image/jpeg")
            return self._send(404, b"no such image", "text/plain")

        return self._send(404, b"not found", "text/plain")

    def do_POST(self):                                               # noqa: N802
        if self.path != "/save":
            return self._send(404, b"not found", "text/plain")
        # A label is a few hundred bytes. Without a ceiling, a request body is read into
        # memory exactly as claimed by the client - and this endpoint has no auth.
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except (TypeError, ValueError):
            return self._json(400, {"ok": False, "why": "bad Content-Length"})
        if length <= 0:
            return self._json(400, {"ok": False, "why": "empty body"})
        if length > MAX_LABEL_BYTES:
            return self._json(413, {"ok": False,
                                    "why": f"body over {MAX_LABEL_BYTES} bytes"})
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            return self._json(400, {"ok": False, "why": f"bad json: {exc}"})
        if not isinstance(payload, dict):
            return self._json(400, {"ok": False, "why": "body must be a JSON object"})

        kind = str(payload.get("kind") or "quad")

        # A feature patch is a free polygon the human clicked, not a fitted rectangle:
        # it marks something ON the car (a roof marker, say), and its shape is whatever
        # the thing looks like. Three points is the minimum that can enclose an area.
        points = payload.get("points") or []
        quad = payload.get("quad") or []

        if kind == "hsv":
            # HSV calibration carries a single click, not an outline: the server finds the blob
            # under it. Validated where it is used.
            pass
        elif kind in ("feature", "hero"):
            if len(points) < 3:
                return self._json(400, {"ok": False, "why": "a feature needs at least 3 points"})
        else:
            if len(quad) != 4:
                return self._json(400, {"ok": False, "why": "need four corners"})

        # Every coordinate below is indexed and rounded; validate the shape once here so a
        # malformed body is a 400 instead of a KeyError/TypeError traceback.
        try:
            points = [[float(p[0]), float(p[1])] for p in points]
            quad = [[float(p[0]), float(p[1])] for p in quad]
            tip_raw = payload.get("tip")
            tip = [float(tip_raw[0]), float(tip_raw[1])] if tip_raw else None
            raw_quad = [[float(p[0]), float(p[1])]
                        for p in (payload.get("raw_quad") or [])][:4]
        except (TypeError, ValueError, IndexError):
            return self._json(400, {"ok": False,
                                    "why": "points/quad/tip must be [x, y] pairs"})

        status = payload.get("status", "ok")
        if status not in ("ok", "skip", "bad"):
            # It becomes a CSS class and is echoed back into the page; keep it to the
            # three values the UI actually sends.
            return self._json(400, {"ok": False, "why": f"unknown status: {status!r}"})
        source = str(payload.get("from") or "").strip()

        os.makedirs(LABELS, exist_ok=True)
        label_id = "ann-%s" % time.strftime("%Y%m%d-%H%M%S")
        image_name = f"{label_id}.jpg"

        if source:                                                   # annotating a saved image
            safe = os.path.basename(source)
            original = None
            for base in (SEED, LABELS):
                candidate = os.path.join(base, safe)
                if os.path.exists(candidate):
                    original = candidate
                    break
            if original is None:
                return self._json(404, {"ok": False, "why": f"no such image: {safe}"})
            with open(original, "rb") as fh:
                blob = fh.read()
            origin = os.path.relpath(original, ROOT)
        else:
            blob, _stamp, error = self.frames.current()
            if blob is None:
                return self._json(503, {"ok": False, "why": error or "no frame yet"})
            origin = self.frames.source

        # The hsv branch keeps its own copy under hsv_marker/frames, where the calibration report
        # reads from; writing it into the annotation set as well would inflate that dataset with
        # frames that carry no annotation.
        if kind != "hsv":
            with open(os.path.join(LABELS, image_name), "wb") as fh:
                fh.write(blob)

        if kind == "hero":
            # A hero box for the HOG/SVM dataset, saved straight into tools/hero_lab/data:
            # the frame, the bbox, a 64x64 positive crop and automatic negatives from the
            # same frame. The `notes` field carries the capture session, which is what
            # train_hog.py splits train/validation on - so two frames from the same minute
            # cannot end up on both sides.
            import sys as _sys
            _sys.path.insert(0, os.path.join(ROOT, "tools", "hero_lab"))
            import hero_capture as _hero

            pts = [[float(p[0]), float(p[1])] for p in points]
            if len(pts) < 3:
                return self._json(400, {"ok": False, "why": "a hero box needs at least 3 points"})
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bbox = [int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys))]
            if bbox[2] < 12 or bbox[3] < 12:
                return self._json(400, {"ok": False, "why": "hero box too small"})

            frame = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                return self._json(400, {"ok": False, "why": "could not decode the frame"})

            for d in (_hero.FRAMES, _hero.POS, _hero.NEG):
                os.makedirs(d, exist_ok=True)
            stem = label_id.replace("ann-", "")
            cv2.imwrite(os.path.join(_hero.FRAMES, stem + ".jpg"), frame)
            x, y, w, h = bbox
            crop = frame[y:y + h, x:x + w]
            if crop.size:
                cv2.imwrite(os.path.join(_hero.POS, stem + ".jpg"),
                            cv2.resize(crop, (_hero.WIN_W, _hero.WIN_H),
                                       interpolation=cv2.INTER_AREA))
            negatives = _hero.auto_negatives(frame, (x, y, w, h), _hero.NEG_PER_FRAME)
            for k, (nx, ny, nw, nh) in enumerate(negatives):
                cv2.imwrite(os.path.join(_hero.NEG, "%s_%d.jpg" % (stem, k)),
                            cv2.resize(frame[ny:ny + nh, nx:nx + nw],
                                       (_hero.WIN_W, _hero.WIN_H),
                                       interpolation=cv2.INTER_AREA))
            with open(_hero.BOXES, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"frame": stem, "bbox": bbox, "negatives": negatives,
                                     "note": str(payload.get("notes", "")).strip() or "default",
                                     "at": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
            record = {"id": label_id, "image": image_name, "kind": "hero", "bbox": bbox,
                      "negatives": len(negatives), "note": str(payload.get("notes", "")),
                      "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            return self._json(200, {"ok": True, "saved": image_name, "record": record})

        if kind == "hsv":
            # HERO HSV calibration: one click on the arrow records what that arrow looks like
            # here - its colour, size and shape distance - at three saturation levels, together
            # with the frame. This exists because the reference contour the shape gate compares
            # against was sampled from a single frame at a single position, so at the dark corner
            # the very same arrow scored 1.14 and was rejected. Four positions of data turns
            # that into a curve instead of a guess.
            point = payload.get("point")
            if (not isinstance(point, (list, tuple)) or len(point) != 2
                    or not all(isinstance(v, (int, float)) for v in point)):
                return self._json(400, {"ok": False, "why": "point must be [x, y]"})

            import sys as _sys
            _sys.path.insert(0, os.path.join(ROOT, "vision", "src"))
            import cv2 as _cv2
            import numpy as _np
            import hsv_marker as _hsv

            frame = _cv2.imdecode(_np.frombuffer(blob, _np.uint8), _cv2.IMREAD_COLOR)
            if frame is None:
                return self._json(400, {"ok": False, "why": "frame did not decode"})

            px, py = float(point[0]), float(point[1])
            pool = _hsv.candidates(frame, centre=(px, py), radius=70.0, max_shape=9.9)
            if not pool:
                return self._json(422, {"ok": False,
                                        "why": "no red blob near that point"})
            # The blob the click actually landed on, not merely the best-scoring one.
            def _dist(c):
                d = abs(c["bbox"][0] - px) + abs(c["bbox"][1] - py)
                inside = (c["bbox"][0] <= px <= c["bbox"][0] + c["bbox"][2]
                          and c["bbox"][1] <= py <= c["bbox"][1] + c["bbox"][3])
                return (0 if inside else 1, d)
            pool.sort(key=_dist)
            hit = pool[0]

            CALIB = os.path.join(ROOT, "vision", "data", "hsv_marker")
            FRAMES_DIR = os.path.join(CALIB, "frames")
            os.makedirs(FRAMES_DIR, exist_ok=True)
            with open(os.path.join(FRAMES_DIR, image_name), "wb") as fh:
                fh.write(blob)

            record = {
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "frame": image_name,
                "from": origin,
                "batch": str(payload.get("notes", ""))[:60],
                "click": [round(px, 1), round(py, 1)],
                "bbox": list(hit["bbox"]),
                "centre": [round(hit["centre"][0], 2), round(hit["centre"][1], 2)],
                "area": hit["area"],
                "width": hit["width"], "height": hit["height"],
                "aspect": round(hit["aspect"], 3),
                "shape_score": hit["shape_score"],
                "s_min_used": hit["s_min"],
                "h": round(hit["h"], 1), "s": round(hit["s"], 1), "v": round(hit["v"], 1),
                # Every level's view of the same arrow, so the report can say which threshold
                # actually fitted it here without re-running the detection later.
                "levels": [{"s_min": c["s_min"], "area": c["area"], "shape": c["shape_score"],
                            "w": c["width"], "h": c["height"]} for c in pool[:6]],
            }
            with open(os.path.join(CALIB, "calibration.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            n = sum(1 for _ in open(os.path.join(CALIB, "calibration.jsonl"), encoding="utf-8"))
            return self._json(200, {"ok": True, "saved": image_name, "record": record,
                                    "total": n})

        if kind == "feature":
            record = {
                "id": label_id,
                "image": image_name,
                "from": origin,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "kind": "feature",
                "name": str(payload.get("name") or "feature")[:40],
                "points": [[int(round(p[0])), int(round(p[1]))] for p in points],
                "closed": bool(payload.get("closed", True)),
                "status": status,
                "notes": str(payload.get("notes", ""))[:200],
            }
            with open(ANNOTATIONS, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            return self._json(200, {"ok": True, "saved": image_name, "record": record})

        record = {
            "id": label_id,
            "image": image_name,
            "from": origin,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "kind": "quad",
            "quad": [[int(round(p[0])), int(round(p[1]))] for p in quad],   # clockwise, p0 at the nose
            # The quad is a fitted rectangle; raw_quad is what the hand actually clicked, kept
            # because a fit cannot be un-fitted later and the clicks cannot be recovered from it.
            "raw_quad": [[int(round(p[0])), int(round(p[1]))] for p in raw_quad],
            "tip": [int(round(tip[0])), int(round(tip[1]))] if tip else None,
            "tip_source": str(payload.get("tip_source", ""))[:12],   # auto | manual
            "status": status,                    # ok | skip | invalid
            "notes": str(payload.get("notes", ""))[:200],
        }
        with open(ANNOTATIONS, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return self._json(200, {"ok": True, "saved": image_name, "record": record})


def main() -> int:
    # Declared before anything reads them: `default=LABELS` below counts as a use, and Python
    # rejects a `global` that comes after a use in the same function.
    global LABELS, ANNOTATIONS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8030)
    ap.add_argument("--source", default=DEFAULT_STREAM)
    ap.add_argument("--labels", default=LABELS,
                    help="where to write labels; point at a temporary directory when testing so a "
                         "test can never write to, or clean up, the real dataset")
    args = ap.parse_args()

    LABELS = os.path.abspath(args.labels)
    ANNOTATIONS = os.path.join(LABELS, "annotations.jsonl")
    os.makedirs(LABELS, exist_ok=True)
    Handler.frames = Frames(args.source)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    # The camera is a single MJPEG stream with a fixed budget: every extra client takes a share of
    # its frames. Measured on this rig, one extra permanent client took the vision service from
    # ~12 fps down to 3.5, main.py then hit "no frames for 20 s" and exited, and the supervisor
    # restarted it every 2 s - 12,907 times. So say it before it happens.
    busy = False
    try:
        import urllib.request
        urllib.request.urlopen("http://127.0.0.1:8010/out/parking_state.json", timeout=1.5).read(1)
        busy = True
    except Exception:
        busy = False
    if busy:
        print("  ! the vision service is running and holds the camera stream")
        print("    each extra client takes a share of the camera's frames - measured, it drops the")
        print("    service from ~12 fps to ~3.5, after which main.py exits on 'no frames for 20 s'")
        print("    and the supervisor restarts it in a loop. For a labelling session run:")
        print("        ./start.sh stop vision        (and ./start.sh start vision afterwards)")
    print(f"  标注页: http://127.0.0.1:{args.port}")
    print(f"  取帧:   {args.source}（后台保持一路连接）")
    print(f"  保存到: {LABELS}/  （图片 + annotations.jsonl）")
    print("  Ctrl-C 结束")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
