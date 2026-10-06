"""The realtime path: the reader's cadence and the state's write cadence.

Two measured problems live here. The camera reader used to pull 8 KB per call, once per frame of
the caller's loop, which is 178 KB/s against a stream offering 282 - so the device slowed the
client down to match, and main.py's source rate read 6.7 fps against a pipeline that costs 22 ms
a frame. And the state was rewritten on a timer, so a change could wait half a second before it
existed as a document, on top of the page's own poll.

The reader tests run against a fake MJPEG server on a local socket, so they need no camera.

Run:  python tests/test_realtime.py
"""

from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import cv2                                                      # noqa: E402
import numpy as np                                              # noqa: E402

import camera                                                   # noqa: E402
import contract                                                 # noqa: E402
import main                                                     # noqa: E402

BOUNDARY = "testboundary"


def a_jpeg(grey: int, size: int = 384) -> bytes:
    """A frame big enough to matter, identifiable by its top-left corner.

    The size is the point: a few small frames fit in the socket buffer, so a reader that never
    drains would still appear to work. Noise does not compress, so each frame is around 200 KB and
    twelve of them cannot sit in any buffer - the server can only finish sending if the client is
    reading. The corner is left flat so a decoded frame can be recognised.
    """
    rng = np.random.default_rng(grey)
    img = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    img[:16, :16] = grey
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    return buf.tobytes()


class FakeCamera(BaseHTTPRequestHandler):
    """An MJPEG stream of `count` frames, each an even grey so the reader can be checked."""

    count = 12
    delay = 0.02
    sent = 0                     # frames fully written to the socket
    written_lock = threading.Lock()

    def log_message(self, *args):                # keep the test output readable
        pass

    idle_after = 2.0             # stay connected, like a camera, so nothing reconnects

    def do_GET(self):                            # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.end_headers()
        try:
            for i in range(self.count):
                jpeg = a_jpeg(40 + i * 15)
                head = (f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpeg)}\r\n\r\n").encode()
                self.wfile.write(head + jpeg + b"\r\n")
                self.wfile.flush()
                with self.written_lock:
                    type(self).sent += 1
                time.sleep(self.delay)
            time.sleep(self.idle_after)          # the stream is still open; just quiet
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


def a_server(count: int = 12, delay: float = 0.02, idle_after: float = 3.0):
    handler = type("Bound", (FakeCamera,),
                   {"count": count, "delay": delay, "sent": 0, "idle_after": idle_after})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/stream"


def wait_until(condition, timeout: float = 6.0, interval: float = 0.01) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# The reader keeps draining, whatever the caller's cadence
# ---------------------------------------------------------------------------

def test_a_slow_caller_does_not_starve_the_connection() -> None:
    """The measured bug: main.py's loop was the only thing reading the socket, so its cadence was
    the client's throughput. Here the caller takes one frame and then does nothing while the
    camera sends ten more - and all of them must still arrive and be decoded."""
    server, url = a_server(count=12, delay=0.01)
    try:
        reader = camera.MjpegHttpReader(url, reconnect_delay=0.1)
        # The server cannot finish writing 2.4 MB unless the client drains it, and the client here
        # is the reader thread - the caller asks for nothing at all until all of it has arrived.
        assert wait_until(lambda: server.RequestHandlerClass.sent == 12), \
            f"only {server.RequestHandlerClass.sent} of 12 frames got out"
        assert wait_until(lambda: reader.stats.frames == 12), reader.stats.summary()
        first = reader.read()
        assert first is not None and first.shape[:2] == (384, 384)
        # Asking once leaves everything else waiting and then replaced: the caller's cadence cost
        # frames, but it never cost the connection.
        assert reader.stats.skipped >= 9, reader.stats.summary()
        reader.close()
    finally:
        server.shutdown()
        server.server_close()


def test_frames_are_handed_over_once_and_in_order() -> None:
    server, url = a_server(count=8, delay=0.03)
    try:
        reader = camera.MjpegHttpReader(url, reconnect_delay=0.1)
        seen, deadline = [], time.time() + 6.0
        while len(seen) < 8 and time.time() < deadline:
            frame = reader.read()
            if frame is not None:
                seen.append(int(frame[0, 0, 0]))
            else:
                time.sleep(0.01)
        reader.close()
        assert len(seen) >= 7, seen                      # the newest may still be in hand
        assert seen == sorted(seen), seen                # in order, never backwards
        assert len(set(seen)) == len(seen), seen         # and never the same picture twice
    finally:
        server.shutdown()
        server.server_close()


def test_what_the_caller_did_not_take_is_accounted_for() -> None:
    """Nothing silent: every decoded frame either reaches the caller or is counted as skipped."""
    server, url = a_server(count=12, delay=0.005)
    try:
        reader = camera.MjpegHttpReader(url, reconnect_delay=0.1)
        assert wait_until(lambda: reader.stats.frames == 12), reader.stats.summary()
        time.sleep(0.1)                                   # nothing more is coming
        taken = 0
        while reader.read() is not None:
            taken += 1
        reader.close()
        assert taken + reader.stats.skipped == 12, (taken, reader.stats.skipped)
        assert reader.stats.skipped > 0, "this test no longer exercises the replacement path"
    finally:
        server.shutdown()
        server.server_close()


def test_close_stops_the_reader_thread() -> None:
    server, url = a_server(count=4, delay=0.02)
    try:
        reader = camera.MjpegHttpReader(url, reconnect_delay=0.1)
        assert reader._thread is not None and reader._thread.is_alive()
        reader.close()
        assert reader._thread is None
        assert reader.read() is None
    finally:
        server.shutdown()
        server.server_close()


def test_the_reader_survives_the_stream_ending() -> None:
    """A camera node rebooting must not kill the viewer; the counters say what happened."""
    server, url = a_server(count=3, delay=0.01, idle_after=0.0)   # the stream ends and closes
    try:
        reader = camera.MjpegHttpReader(url, reconnect_delay=0.05)
        assert wait_until(lambda: reader.stats.reconnects >= 1), reader.stats.summary()
        assert reader.read() is not None or reader.stats.frames >= 1   # it keeps working
        reader.close()
    finally:
        server.shutdown()
        server.server_close()


def test_the_reader_takes_whole_frames_off_the_socket_not_eight_kilobytes() -> None:
    """The number behind the bug: 8192 bytes per call, 22 calls a second, is 178 KB/s against a
    282 KB/s stream, so the device slows the client down to whatever it can drain."""
    assert camera.READ_SIZE >= 64 * 1024, camera.READ_SIZE
    assert camera.MAX_PENDING >= 2, "one slot throws away a whole burst"


# ---------------------------------------------------------------------------
# The state is written when it changes
# ---------------------------------------------------------------------------

def a_state_doc(hero_x: float = 100.0, hero_heading: float = 90.0,
                bay: str = "EMPTY", ratio: float = 0.0) -> dict:
    return {"schema": "parking_state/1", "timestamp": 1791283027,
            "spots": {"10": {"state": bay, "changed_ratio": ratio}},
            "vehicle": {"id": "hero", "x": hero_x, "y": 300.0, "heading": hero_heading,
                        "visible": True, "front_is_known": True, "front_heading": hero_heading}}


def test_a_still_scene_has_one_key() -> None:
    """Jitter must not look like change, or the file is rewritten for nothing every frame."""
    base = a_state_doc()
    for dx in (-0.4, 0.0, 0.3):
        for dh in (-0.4, 0.0, 0.3):
            doc = a_state_doc(hero_x=100.0 + dx, hero_heading=90.0 + dh)
            assert main.state_key(doc) == main.state_key(base), (dx, dh)
    assert main.state_key(a_state_doc(ratio=0.004)) == main.state_key(base)


def test_a_moving_or_changing_scene_has_a_new_key() -> None:
    base = a_state_doc()
    assert main.state_key(a_state_doc(hero_x=102.0)) != main.state_key(base)
    assert main.state_key(a_state_doc(hero_heading=95.0)) != main.state_key(base)
    assert main.state_key(a_state_doc(bay="OCCUPIED", ratio=0.44)) != main.state_key(base)
    assert main.state_key(a_state_doc(ratio=0.40)) != main.state_key(base)
    later = dict(base, timestamp=1791283028)             # the clock is not a change
    assert main.state_key(later) == main.state_key(base)


def test_the_state_is_written_on_change_and_on_the_heartbeat() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "parking_state.json")
        args = SimpleNamespace(state_out=path, state_every=0.5)

        # The first write happens at once, whatever the clock says.
        wrote = main.write_state(args, {}, [], False, 0.0, 0.0, None, None)
        assert wrote is not None, "the first state must be written immediately"
        when, key = wrote
        assert os.path.exists(path)

        # The same state again, a moment later: nothing to say.
        assert main.write_state(args, {}, [], False, 0.0, when, None, key) is None
        # Half the interval with the same state: still nothing.
        assert main.write_state(args, {}, [], False, 0.0, when - 0.2, None, key) is None
        # A whole interval with the same state: the heartbeat, so the page's age is honest.
        assert main.write_state(args, {}, [], False, 0.0, when - 0.6, None, key) is not None
        # A different reason after half the interval: it counts as a change.
        other = main.state_key({"spots": {}, "vehicle": None, "reason": "no frames for 3.0s"})
        assert main.write_state(args, {}, [], False, 0.0, when - 0.3, None, other) is not None
        # A different reason inside half the interval: held back, not lost.
        assert main.write_state(args, {}, [], False, 0.0, when - 0.1, None, other) is None


def test_the_document_written_is_the_p0002_contract_unchanged() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "parking_state.json")
        args = SimpleNamespace(state_out=path, state_every=0.5)
        main.write_state(args, {}, [], False, 0.0, 0.0, None, None)
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        assert doc["schema"] == contract.STATE_SCHEMA, doc["schema"]
        assert "timestamp" in doc and "spots" in doc and "vehicle" in doc


def main_entry() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:                     # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok    {name}")
    print(f"\n{len(tests)} checks, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main_entry())
