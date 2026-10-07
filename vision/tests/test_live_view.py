"""The board's stream: the analysed picture, served without slowing the analysis down.

The point of this module is the same one camera.py learned: a producer must not be slowed by
whoever is watching. Here that means publishing never waits for an encoder or a viewer, encoding
happens once for all viewers, and nothing is encoded at all when nobody is looking.

No camera and no display are needed.

Run:  python tests/test_live_view.py
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np                                              # noqa: E402

import live_view                                                # noqa: E402
import route                                                    # noqa: E402

BOARD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web", "board.html")


def a_frame(grey: int = 100, width: int = 320, height: int = 240) -> np.ndarray:
    return np.full((height, width, 3), grey, dtype=np.uint8)


def a_view(**kwargs) -> live_view.LiveView:
    view = live_view.LiveView(port=0, **kwargs)
    view.start()
    return view


def read_stream(url: str, want_frames: int, timeout: float = 5.0):
    """Fetch an MJPEG stream and return the JPEG parts, as a browser would."""
    parts = []
    with urllib.request.urlopen(url, timeout=timeout) as response:
        assert response.headers["Content-Type"].startswith("multipart/x-mixed-replace"), \
            response.headers["Content-Type"]
        buf, deadline = b"", time.time() + timeout
        while len(parts) < want_frames and time.time() < deadline:
            chunk = response.read(4096)
            if not chunk:
                break
            buf += chunk
            while True:
                start = buf.find(b"\xff\xd8\xff")
                end = buf.find(b"\xff\xd9", start + 3) if start >= 0 else -1
                if start < 0 or end < 0:
                    break
                parts.append(buf[start:end + 2])
                buf = buf[end + 2:]
                if len(parts) >= want_frames:
                    return parts                        # a chunk can carry more than one frame
    return parts


def test_nothing_is_encoded_when_nobody_is_watching() -> None:
    """Encoding 800x600 costs about 10 ms; the loop is not paying it for an empty room."""
    view = a_view()
    try:
        for i in range(20):
            view.publish(a_frame(i))
            time.sleep(0.005)
        time.sleep(0.3)
        assert view.clients == 0
        assert view.frames_published == 20, view.frames_published
        assert view.frames_encoded == 0, view.stats()
    finally:
        view.stop()


def test_a_client_receives_frames_it_can_decode() -> None:
    view = a_view()
    try:
        stop = threading.Event()
        threading.Thread(target=lambda: [view.publish(a_frame(i % 250))
                                         or time.sleep(0.02)
                                         for i in range(2000) if not stop.is_set()],
                         daemon=True).start()
        assert view._server is not None
        url = f"http://127.0.0.1:{view.port}/stream"
        parts = read_stream(url, want_frames=3)
        stop.set()
        assert len(parts) == 3, len(parts)           # the reader stops as soon as it has enough
        for part in parts:
            assert part[:3] == b"\xff\xd8\xff", part[:8]
        assert view.frames_encoded > 0, view.stats()
    finally:
        view.stop()


def test_publishing_never_waits_for_a_viewer() -> None:
    """The producer's cost is a copy; a viewer that never reads must not be able to change that."""
    view = a_view(max_fps=60.0)
    try:
        # A client that connects and then does not read, so frames pile up and are dropped.
        view._add_client()
        view.publish(a_frame())
        time.sleep(0.1)
        durations = []
        for i in range(50):
            t0 = time.perf_counter()
            view.publish(a_frame(i))
            durations.append((time.perf_counter() - t0) * 1000)
        worst = max(durations)
        assert worst < 40.0, f"a publish took {worst:.1f} ms with an idle viewer"
        time.sleep(0.4)
        assert view.stats()["dropped"] > 0, view.stats()
    finally:
        view.stop()


def test_frames_are_encoded_once_however_many_are_watching() -> None:
    view = a_view(max_fps=60.0)
    try:
        view._add_client()
        view._add_client()
        for i in range(6):
            view.publish(a_frame(i))
            time.sleep(0.03)
        time.sleep(0.3)
        assert view.frames_encoded <= view.frames_published + 1, view.stats()
    finally:
        view.stop()


def test_the_info_page_and_an_unknown_path() -> None:
    view = a_view()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{view.port}/", timeout=5) as r:
            assert r.status == 200 and b"<img" in r.read()
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{view.port}/nope", timeout=5)
            raise AssertionError("a wrong path should not be served")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        view.stop()


def test_stop_lets_the_port_go() -> None:
    view = a_view()
    port = view.port
    view.stop()
    assert view._server is None
    probe = live_view.LiveView(port=port)
    probe.start()                       # would raise if the port were still held
    probe.stop()


# ---------------------------------------------------------------------------
# The board page
# ---------------------------------------------------------------------------

def codes_the_planner_can_return() -> list:
    return [getattr(route, name) for name in
            ("NO_SUCH_SPOT", "NO_ACCESS", "NOT_FREE", "NO_HERO", "NO_ROUTE")]


def test_every_refusal_the_planner_can_give_has_a_chinese_sentence() -> None:
    """A visitor must never be shown a raw code. If route.py grows one, the board must grow a
    sentence for it - and this is where that is noticed."""
    html = open(BOARD, encoding="utf-8").read()
    body = html[html.index("const WHY = {"):html.index("};", html.index("const WHY = {"))]
    phrases = set(re.findall(r'"([a-z0-9-]+)":', body))
    missing = [code for code in codes_the_planner_can_return() if code not in phrases]
    assert not missing, f"the board has no sentence for {missing}"
    assert "endpoint-unreachable" in phrases, "the board's own failure mode needs one too"
    assert len(phrases) >= 6, phrases


def test_the_board_draws_from_the_analysed_stream_and_shows_it() -> None:
    html = open(BOARD, encoding="utf-8").read()
    assert "8020" in html, "the board must know where main.py serves the picture"
    assert "liveImg.addEventListener(\"error\"" in html, "a kiosk needs the stream to come back"
    assert "本板坐标" not in html
    for needed in ("/api/route", "POLL_MS = 250", "parking_state.json"):
        assert needed in html, needed


def main() -> int:
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
    raise SystemExit(main())
