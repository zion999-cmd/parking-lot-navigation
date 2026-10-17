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
  * Saved under vision/data/labels/, which is not gitignored: labels are work.
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

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "vision", "src"))

LABELS = os.path.join(ROOT, "vision", "data", "labels")
SEED = os.path.join(LABELS, "seed")
ANNOTATIONS = os.path.join(LABELS, "annotations.jsonl")
UI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_ui.html")

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
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except ValueError as exc:
            return self._json(400, {"ok": False, "why": f"bad json: {exc}"})

        quad = payload.get("quad") or []
        if len(quad) != 4:
            return self._json(400, {"ok": False, "why": "need four corners"})
        status = payload.get("status", "ok")
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

        with open(os.path.join(LABELS, image_name), "wb") as fh:
            fh.write(blob)

        record = {
            "id": label_id,
            "image": image_name,
            "from": origin,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "quad": [[int(round(p[0])), int(round(p[1]))] for p in quad],   # clockwise, p0 at the nose
            # The quad is a fitted rectangle; raw_quad is what the hand actually clicked, kept
            # because a fit cannot be un-fitted later and the clicks cannot be recovered from it.
            "raw_quad": [[int(round(p[0])), int(round(p[1]))]
                         for p in (payload.get("raw_quad") or [])][:4],
            "tip": [int(round(payload["tip"][0])), int(round(payload["tip"][1]))]
            if payload.get("tip") else None,
            "tip_source": str(payload.get("tip_source", ""))[:12],   # auto | manual
            "status": status,                    # ok | skip | invalid
            "notes": str(payload.get("notes", ""))[:200],
        }
        with open(ANNOTATIONS, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return self._json(200, {"ok": True, "saved": image_name, "record": record})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8030)
    ap.add_argument("--source", default=DEFAULT_STREAM)
    args = ap.parse_args()

    os.makedirs(LABELS, exist_ok=True)
    Handler.frames = Frames(args.source)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
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
