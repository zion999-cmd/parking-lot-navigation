"""Camera input for the parking vision PoC.

Two backends, chosen because the ESP32-S3-CAM's MJPEG stream is not readable by
OpenCV's bundled FFmpeg here:

* :class:`OpenCvVideoCapture` - plain ``cv2.VideoCapture``. Tried first, because
  it is the obvious answer and it does work for files and for some cameras.
* :class:`MjpegHttpReader` - a hand-rolled ``multipart/x-mixed-replace`` reader:
  parse the boundary, read ``Content-Length``, hand the JPEG bytes to
  ``cv2.imdecode``. Nothing but the standard library and OpenCV.

``open_camera()`` tries the first and falls back to the second, so callers do not
need to care which one is in use.

Measured on 2026-10-05 against http://192.168.3.118/stream:
``cv2.VideoCapture`` reports ``isOpened() == False`` ("Couldn't read video stream
from file"), while :class:`MjpegHttpReader` delivers ~13-14 fps at 800x600.

The reader drains the socket on its own thread, because a caller's cadence must not be able to
starve the connection - see :meth:`MjpegHttpReader._pump` for the numbers that made that matter.
"""

from __future__ import annotations

import re
import socket
import threading
import time
from collections import deque
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

# The firmware sends this boundary; used only if the HTTP response does not
# advertise one.
DEFAULT_BOUNDARY = b"parkinglotcamboundary"

# Refuse to buffer more than this while hunting for a frame. A 800x600 JPEG is
# ~25 KB, so 4 MB is several frames of slack before we resynchronise.
MAX_BUFFER = 4 * 1024 * 1024

# How much to take off the socket in one go. The stream is about 282 KB/s in 25 KB frames, so a
# 64 KB read is two or three frames and one syscall each, instead of 8192 bytes and four syscalls
# per frame - which is what used to leave this reader 40% behind the stream.
READ_SIZE = 64 * 1024

# A single chunk cannot contain more than a few frames; a bound keeps a corrupted buffer from
# spinning the reader thread forever.
MAX_FRAMES_PER_READ = 8

# How many decoded frames may wait for the caller. Two: enough that a burst is not thrown away,
# short enough that the caller is never more than a frame or two behind the camera.
MAX_PENDING = 2


@dataclass
class StreamStats:
    """Lightweight counters so the caller can report real fps and stalls."""

    frames: int = 0              # decoded, whether or not the caller was quick enough to take it
    skipped: int = 0             # decoded but replaced by a newer one before the caller asked
    reconnects: int = 0
    decode_failures: int = 0
    started_at: float = field(default_factory=time.time)
    last_frame_at: float = 0.0

    def fps(self) -> float:
        dt = max(time.time() - self.started_at, 1e-6)
        return self.frames / dt

    def summary(self) -> str:
        return (f"frames={self.frames} fps={self.fps():.1f} skipped={self.skipped} "
                f"reconnects={self.reconnects} decode_failures={self.decode_failures}")


class CameraError(RuntimeError):
    pass


class OpenCvVideoCapture:
    """Thin wrapper over ``cv2.VideoCapture``."""

    name = "opencv"

    def __init__(self, source: str):
        self.source = source
        self.stats = StreamStats()
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            self._cap.release()
            raise CameraError(f"cv2.VideoCapture could not open {source!r}")

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        self.stats.frames += 1
        self.stats.last_frame_at = time.time()
        return frame

    def close(self) -> None:
        self._cap.release()

    @property
    def resolution(self) -> Tuple[int, int]:
        return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))


class MjpegHttpReader:
    """Read an MJPEG ``multipart/x-mixed-replace`` stream over HTTP.

    The firmware always sends ``Content-Length`` per part, which makes framing
    trivial; if a future server omits it we fall back to scanning for the next
    boundary. The connection is re-established automatically, because a camera
    node rebooting (or an OTA) must not kill the viewer.
    """

    name = "mjpeg-http"

    _HEADER_RE = re.compile(rb"Content-Type:\s*image/jpeg", re.IGNORECASE)
    _LENGTH_RE = re.compile(rb"Content-Length:\s*(\d+)", re.IGNORECASE)

    def __init__(self, url: str, timeout: float = 6.0, reconnect_delay: float = 1.0,
                 threaded: bool = True):
        self.url = url
        self.timeout = timeout
        self.reconnect_delay = reconnect_delay
        self.stats = StreamStats()
        self._buf = bytearray()
        self._resp = None
        self._boundary = DEFAULT_BOUNDARY
        self._width = 0
        self._height = 0
        self._lock = threading.Lock()
        self._pending: Deque[np.ndarray] = deque()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._connect()                     # fail here, not in the background, if the URL is bad
        if threaded:
            self._thread = threading.Thread(target=self._pump, name="mjpeg-reader", daemon=True)
            self._thread.start()

    # -- connection handling -------------------------------------------------

    def _connect(self) -> None:
        self._close()
        self._buf.clear()
        req = urllib.request.Request(self.url, headers={"User-Agent": "parking-vision/0.1"})
        try:
            self._resp = urllib.request.urlopen(req, timeout=self.timeout)
        except (urllib.error.URLError, socket.timeout, OSError) as exc:
            raise CameraError(f"could not open {self.url!r}: {exc}") from exc

        ctype = self._resp.headers.get("Content-Type", "")
        m = re.search(r"boundary=([^;]+)", ctype)
        if m:
            self._boundary = m.group(1).strip().strip('"').encode()

    def _close(self) -> None:
        if self._resp is not None:
            try:
                self._resp.close()
            except Exception:
                pass
            self._resp = None

    # -- frame extraction ---------------------------------------------------

    def _extract(self) -> Optional[bytes]:
        """Pull one JPEG out of the buffer, or None if not enough data yet."""
        start = self._HEADER_RE.search(self._buf)
        if start is None:
            # Nothing that looks like a part header: keep only a small tail.
            if len(self._buf) > MAX_BUFFER:
                del self._buf[:-4096]
            return None

        # Header block ends at the blank line.
        split = self._buf.find(b"\r\n\r\n", start.end())
        sep_len = 4
        if split < 0:
            split = self._buf.find(b"\n\n", start.end())
            sep_len = 2
        if split < 0:
            return None

        header = bytes(self._buf[start.start():split])
        body_start = split + sep_len

        m = self._LENGTH_RE.search(header)
        if m:
            length = int(m.group(1))
            if len(self._buf) - body_start < length:
                return None                      # wait for the rest
            jpeg = bytes(self._buf[body_start:body_start + length])
            del self._buf[:body_start + length]
            return jpeg

        # No Content-Length: take everything up to the next boundary.
        nxt = self._buf.find(b"--" + self._boundary, body_start)
        if nxt < 0:
            return None
        jpeg = bytes(self._buf[body_start:nxt])
        del self._buf[:nxt]
        return jpeg

    def _pump(self) -> None:
        """Drain the socket as fast as it will go, keeping only the newest frame.

        Not an optimization for its own sake. The device stops giving a client its full share when
        that client does not drain the socket, and this reader used to read 8192 bytes per call,
        once per frame of the caller's loop - 8 KB x 22 a second = 178 KB/s against a stream
        offering 282 KB/s, so it fell about 40% behind and the device slowed it down to match.
        Measured, same reader, same camera: pulling as fast as possible, 10.6 fps; pulling once
        every 45 ms, which is main.py's loop cadence, 6.2 fps. main.py was not slow because it
        computed slowly - its whole pipeline is 22 ms a frame - but because it did not read often
        enough, and its cadence should not be able to starve the connection.

        Frames wait in a queue two deep, oldest handed over first and the oldest dropped when it
        overflows. One slot would be simpler, and is what this had first, but it throws away a
        whole burst: measured against the real camera, the reader decoded 9.9 fps while a caller
        polling every 45 ms - main.py's cadence - took only 4.6, with 66 of 121 frames replaced
        before anyone asked for them. Two frames keeps the freshest of a burst and the one before
        it, at the cost of at most one frame of extra delay. `stats.skipped` counts what is lost.
        """
        while not self._stop.is_set():
            if self._resp is None and not self._reconnect():
                continue
            try:
                chunk = self._resp.read(READ_SIZE)
            except (socket.timeout, urllib.error.URLError, OSError):
                chunk = b""
            if self._stop.is_set():
                break
            if not chunk:
                self._reconnect()
                continue

            self._buf.extend(chunk)
            if len(self._buf) > MAX_BUFFER:
                del self._buf[:-MAX_BUFFER // 2]

            # A chunk can hold several frames; take them all, but bound the loop so a buffer full
            # of rubbish cannot spin here forever.
            for _ in range(MAX_FRAMES_PER_READ):
                jpeg = self._extract()
                if jpeg is None:
                    break
                frame = self._decode(jpeg)
                if frame is None:
                    continue
                with self._lock:
                    if len(self._pending) >= MAX_PENDING:
                        self._pending.popleft()
                        self.stats.skipped += 1
                    self._pending.append(frame)

    def _decode(self, jpeg: bytes) -> Optional[np.ndarray]:
        if len(jpeg) < 4 or jpeg[:2] != b"\xff\xd8":
            self.stats.decode_failures += 1
            return None
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self.stats.decode_failures += 1
            return None
        self._height, self._width = frame.shape[:2]
        self.stats.frames += 1
        self.stats.last_frame_at = time.time()
        return frame

    def read(self) -> Optional[np.ndarray]:
        """The newest frame decoded since the last call, or None if there is not a newer one.

        Returning None rather than repeating the last frame is deliberate: the caller's loop keeps
        the frame it has and works on it, so handing out the same picture twice would only double
        the detection cost for nothing.
        """
        if self._thread is None:                    # threaded=False: read inline, one chunk per call
            self._pump_once()
        with self._lock:
            frame = self._pending.popleft() if self._pending else None
        return frame

    def _pump_once(self) -> None:
        """One synchronous read, for threaded=False and for the tests."""
        if self._resp is None and not self._reconnect():
            return
        try:
            chunk = self._resp.read(READ_SIZE)
        except (socket.timeout, urllib.error.URLError, OSError):
            chunk = b""
        if not chunk:
            self._reconnect()
            return
        self._buf.extend(chunk)
        if len(self._buf) > MAX_BUFFER:
            del self._buf[:-MAX_BUFFER // 2]
        for _ in range(MAX_FRAMES_PER_READ):
            jpeg = self._extract()
            if jpeg is None:
                break
            frame = self._decode(jpeg)
            if frame is not None:
                with self._lock:
                    if len(self._pending) >= MAX_PENDING:
                        self._pending.popleft()
                        self.stats.skipped += 1
                    self._pending.append(frame)

    def _reconnect(self) -> bool:
        """Try to get the stream back. False means not now (so the pump can stop promptly)."""
        self.stats.reconnects += 1
        self._close()
        if self._stop.is_set():
            return False
        time.sleep(self.reconnect_delay)
        if self._stop.is_set():
            return False
        try:
            self._connect()
            return True
        except CameraError:
            self._resp = None
            return False

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.reconnect_delay + 1.0))
            self._thread = None
        self._close()

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self._width, self._height)


def open_camera(source: str, backend: str = "auto", timeout: float = 6.0):
    """Open ``source``.

    backend:
        ``auto``   - try cv2.VideoCapture, fall back to the HTTP reader
        ``opencv`` - only cv2.VideoCapture
        ``mjpeg``  - only the HTTP multipart reader
    """
    if backend not in ("auto", "opencv", "mjpeg"):
        raise ValueError(f"unknown backend {backend!r}")

    if backend in ("auto", "opencv"):
        try:
            cam = OpenCvVideoCapture(source)
            # An open capture that yields nothing is worse than useless, so
            # require one frame before accepting this backend.
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if cam.read() is not None:
                    return cam
            cam.close()
            if backend == "opencv":
                raise CameraError("cv2.VideoCapture opened but produced no frames")
        except CameraError:
            if backend == "opencv":
                raise

    if not source.lower().startswith(("http://", "https://")):
        raise CameraError(f"{source!r} is not an HTTP URL and cv2.VideoCapture failed")
    return MjpegHttpReader(source, timeout=timeout)
