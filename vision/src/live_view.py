"""Publish the analysed picture as an MJPEG stream, for the board page to show.

Why this exists rather than the page using the camera's own ``/stream``: the camera gives every
connected client a share of its frames, so a browser tab on the raw stream halves what the analysis
gets - measured, 35 fps split two ways. This serves the *analysed* frames instead, so the board is
the only client, and it shows what the system sees rather than what the sensor sees.

The encoding happens here, on this server's own thread, never in the loop that produced the frame.
That is the lesson ``camera.py`` paid for: a producer must not be slowed down by whoever is
watching it. Encoding 800x600 costs about 10 ms, which at 15 frames a second is 15% of a core -
acceptable to spend, but not acceptable to spend inside the detection loop, so publishing only ever
stores a reference. A client that cannot keep up loses frames instead of delaying the pipeline, and
so does a client that connects while nobody is looking at the picture.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional

import cv2
import numpy as np

BOUNDARY = b"parkinglotviewboundary"

# Encoded frames per second at most. The analysis runs at 10-17 fps; there is nothing to see above
# that, and every encode is a real cost on a laptop that is also running the detection.
DEFAULT_MAX_FPS = 15.0


class _Client:
    """One viewer: a slot holding the newest encoded frame it has not been sent yet."""

    def __init__(self) -> None:
        self.available = threading.Event()
        self._jpeg: Optional[bytes] = None
        self._lock = threading.Lock()
        self.dropped = 0
        self.closed = False

    def offer(self, jpeg: bytes) -> None:
        with self._lock:
            if self._jpeg is not None:
                self.dropped += 1          # the viewer is behind; it gets the present instead
            self._jpeg = jpeg
        self.available.set()

    def take(self, timeout: float) -> Optional[bytes]:
        if not self.available.wait(timeout):
            return None
        with self._lock:
            jpeg, self._jpeg = self._jpeg, None
            self.available.clear()
        return jpeg


class LiveView:
    """Serves the newest published frame as ``multipart/x-mixed-replace``."""

    def __init__(self, port: int = 8020, host: str = "0.0.0.0",
                 quality: int = 80, max_fps: float = DEFAULT_MAX_FPS) -> None:
        self.port = port
        self.host = host
        self.quality = quality
        self.interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self.frames_encoded = 0
        self.frames_published = 0
        self._frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._stop = threading.Event()
        self._clients: List[_Client] = []
        self._clients_lock = threading.Lock()
        self._server: Optional[ThreadingHTTPServer] = None
        self._threads: List[threading.Thread] = []

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> str:
        handler = type("LiveHandler", (BaseHTTPRequestHandler,), {
            "view": self,
            "log_message": lambda *a, **k: None,
            "do_GET": _LiveHandler.do_GET,
        })
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self.port = self._server.server_address[1]          # 0 means "any free port"
        for target in (self._pump, self._serve):
            thread = threading.Thread(target=target, daemon=True, name=f"live-{target.__name__}")
            thread.start()
            self._threads.append(thread)
        return self.url

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}/stream"

    def stop(self) -> None:
        self._stop.set()
        self._new_frame.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads.clear()

    # -- the producing side --------------------------------------------------

    def publish(self, frame: np.ndarray) -> None:
        """Hand over the newest picture. Copies, because the caller draws on its canvas after this.

        Cheap - about half a millisecond for 800x600 - and it is what lets the loop carry on
        without ever waiting for an encoder or a viewer.
        """
        with self._lock:
            self._frame = frame.copy()
            self._frame_id += 1
            self.frames_published += 1
        self._new_frame.set()

    @property
    def clients(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    # -- the serving side ----------------------------------------------------

    def _serve(self) -> None:
        assert self._server is not None
        self._server.serve_forever(poll_interval=0.2)

    def _pump(self) -> None:
        """Encode the newest frame at most `max_fps` times a second, once for all viewers."""
        last_id, last_at = -1, 0.0
        while not self._stop.is_set():
            if not self._new_frame.wait(0.1):
                continue
            self._new_frame.clear()
            if self.clients == 0:
                continue                                   # nobody is watching: not a cost worth paying
            now = time.time()
            if now - last_at < self.interval:
                continue
            with self._lock:
                frame, frame_id = self._frame, self._frame_id
            if frame is None or frame_id == last_id:
                continue
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
            if not ok:
                continue
            jpeg = buf.tobytes()
            last_id, last_at = frame_id, now
            self.frames_encoded += 1
            with self._clients_lock:
                clients = list(self._clients)
            for client in clients:
                client.offer(jpeg)

    # -- the HTTP side -------------------------------------------------------

    def _add_client(self) -> _Client:
        client = _Client()
        with self._clients_lock:
            self._clients.append(client)
        self._new_frame.set()
        return client

    def _remove_client(self, client: _Client) -> None:
        with self._clients_lock:
            if client in self._clients:
                self._clients.remove(client)

    def stats(self) -> Dict[str, object]:
        with self._clients_lock:
            dropped = sum(c.dropped for c in self._clients)
        return {"clients": self.clients, "published": self.frames_published,
                "encoded": self.frames_encoded, "dropped": dropped}


class _LiveHandler(BaseHTTPRequestHandler):
    """One GET: either the stream, or a page saying what this is."""

    protocol_version = "HTTP/1.0"

    def do_GET(self):                                    # noqa: N802  (stdlib naming)
        view: LiveView = self.view                       # type: ignore[attr-defined]
        if self.path.split("?")[0] not in ("/stream", "/", "/index.html"):
            self.send_error(404)
            return
        if self.path.split("?")[0] != "/stream":
            body = (b"<!doctype html><meta charset=utf-8><title>analysed stream</title>"
                    b"<h1>analysed stream</h1><p><img src=\"/stream\" "
                    b"style=\"max-width:100%\"></p>")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=" + BOUNDARY.decode())
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        client = view._add_client()
        try:
            while not view._stop.is_set():
                jpeg = client.take(1.0)
                if jpeg is None:
                    continue
                head = (b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n")
                self.wfile.write(head + jpeg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            view._remove_client(client)
