#!/usr/bin/env python3
"""Magnetometer test page: four bay cards, one Baseline button, live delta-B.

A bench tool, not a product UI. It reuses the detector - MagPoller, MagBaseline, MagJudge from
vision/src/mag_baseline.py - without changing a line of it, so what it shows is what the vision run
would decide.

Started by tools/mag-test.sh, which finds the board and opens a browser. Standard library only.
"""

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "vision", "src"))

import mag_baseline                                              # noqa: E402

MAPPING_PATH = os.path.join(ROOT, "vision", "config", "mag_bays.json")
BASELINE_PATH = os.path.join(ROOT, "vision", "data", "mag_baseline.json")
CAMERA_CACHE = os.path.join(ROOT, "run", "camera.txt")
HTML_PATH = os.path.join(ROOT, "tools", "mag_test.html")


def build_stamp():
    newest = 0.0
    for path in (os.path.abspath(__file__), HTML_PATH):
        try:
            newest = max(newest, os.path.getmtime(path))
        except OSError:
            pass
    return time.strftime("%H:%M:%S", time.localtime(newest))


BUILD = build_stamp()
BAYS = ("B1", "B2", "B3", "B4")


# ---------------------------------------------------------------------------- device discovery

def board_from_url(url):
    """http://host:port/stream -> http://host:port , or None."""
    if not url or "://" not in url:
        return None
    return url.split("://", 1)[0] + "://" + url.split("://", 1)[1].split("/", 1)[0]


def mag_answers(base, timeout=1.5):
    try:
        with urllib.request.urlopen(f"{base}/api/mag", timeout=timeout) as response:
            return response.status == 200
    except Exception:                                        # noqa: BLE001
        return False


def probe_lan(timeout=0.6):
    """Sweep this machine's /24 for a board answering /api/mag. Parallel, about a second."""
    try:
        import subprocess
        out = subprocess.run(["ipconfig", "getifaddr", "en0"], capture_output=True, text=True,
                             timeout=3).stdout.strip()
    except Exception:                                        # noqa: BLE001
        out = ""
    if not out or out.count(".") != 3:
        return []
    prefix = out.rsplit(".", 1)[0]
    found = []
    lock = threading.Lock()

    def check(octet):
        base = f"http://{prefix}.{octet}"
        if mag_answers(base, timeout):
            with lock:
                found.append(base)

    threads = [threading.Thread(target=check, args=(octet,), daemon=True) for octet in range(1, 255)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout + 0.5)
    return found


def find_board(explicit=None):
    """Explicit address, then start.sh's camera cache, then a LAN sweep."""
    if explicit:
        base = explicit if "://" in explicit else f"http://{explicit}"
        base = base.rstrip("/")
        if base.endswith("/stream"):
            base = base[:-len("/stream")]
        return base, "given on the command line"
    try:
        with open(CAMERA_CACHE, encoding="utf-8") as handle:
            cached = board_from_url(handle.read().strip())
    except OSError:
        cached = None
    if cached and mag_answers(cached):
        return cached, f"run/camera.txt ({os.path.relpath(CAMERA_CACHE, ROOT)})"
    found = probe_lan()
    if found:
        return found[0], "probed this computer's /24 for /api/mag"
    return None, "not found"


# ---------------------------------------------------------------------------- shared state

class TestState:
    def __init__(self, board, source, mapping):
        self.board = board
        self.source = source
        self.mapping = mapping
        self.poller = mag_baseline.MagPoller(f"{board}/api/mag") if board else None
        if self.poller:
            self.poller.start()
        self.baseline = mag_baseline.MagBaseline.load(BASELINE_PATH)
        self.judge = mag_baseline.MagJudge(self.baseline)
        self.capturing = False
        self.capture_note = ""
        self.lock = threading.Lock()
        # The judge is driven on its own beat, not by page refreshes. Judged on request, the 1 s
        # window only ever holds a point or two when the page polls slowly, and a closed tab stops
        # judging altogether; both look like a hardware fault and neither is one.
        self._states = {}
        self._stop = threading.Event()
        threading.Thread(target=self._tick, daemon=True, name="mag-judge").start()

    def _tick(self, period=0.25):
        while not self._stop.is_set():
            try:
                if self.poller is not None and self.judge is not None:
                    now = time.time()
                    self._states = self.judge.update(self.poller.snapshot(now), now)
            except Exception:                                # noqa: BLE001
                pass
            self._stop.wait(period)

    def stop(self):
        self._stop.set()

    # -- mapping -------------------------------------------------------------------
    def save_mapping(self, pairs):
        for bay, channel in pairs.items():
            if bay in BAYS and (channel is None or isinstance(channel, int)):
                self.mapping[bay] = channel
        document = {
            "comment": ("Bay -> TCA9548A channel, established by measurement with "
                        "tools/bench/mag_map.py. Never assume B1=CH0."),
            "established_by": "tools/mag-test.sh (set from the test page)",
            "established_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "bays": {bay: {"channel": self.mapping.get(bay)} for bay in sorted(self.mapping)},
        }
        os.makedirs(os.path.dirname(MAPPING_PATH), exist_ok=True)
        tmp = MAPPING_PATH + ".writing"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)
            handle.write("\n")
        os.replace(tmp, MAPPING_PATH)
        # A judge built for the old mapping must not keep judging the new one.
        self.judge = mag_baseline.MagJudge(self.baseline)

    # -- baseline ------------------------------------------------------------------
    def start_capture(self, duration=3.0):
        with self.lock:
            if self.capturing:
                return False, "already sampling"
            if self.poller is None:
                return False, "no board"
            if not any(channel is not None for channel in self.mapping.values()):
                # Sampling nothing can only produce nothing. Say it here, with the fix, instead of
                # letting the capture run three seconds and report "no bay produced a baseline".
                return False, ("no bay is mapped to a channel yet - in the table below, watch which "
                               "channel moves when a car sits on B4, then pick that bay in the last "
                               "column")
            self.capturing = True
            self.capture_note = f"sampling {duration:g} s"
        threading.Thread(target=self._capture, args=(duration,), daemon=True).start()
        return True, "sampling"

    def _capture(self, duration):
        note = "failed"
        try:
            baseline = mag_baseline.MagBaseline.capture(
                self.poller, self.mapping, duration=duration, source=self.source,
                log=lambda _text: None)
            valid = sum(1 for entry in baseline.bays.values() if entry.get("valid"))
            if valid == 0:
                reasons = "; ".join(f"{bay}: {entry.get('reason') or 'invalid'}"
                                    for bay, entry in baseline.bays.items()
                                    if not entry.get("valid"))
                note = f"failed: no usable baseline, previous one kept - {reasons}"
            else:
                baseline.save(BASELINE_PATH)
                self.baseline = baseline
                self.judge = mag_baseline.MagJudge(baseline)
                note = f"captured {valid}/{len(baseline.bays)} bay(s)"
        except Exception as exc:                             # noqa: BLE001
            note = f"failed: {type(exc).__name__}: {exc}"
        with self.lock:
            self.capturing = False
            self.capture_note = note

    # -- report --------------------------------------------------------------------
    def snapshot(self):
        now = time.time()
        frame = self.poller.snapshot(now) if self.poller else {"_meta": {}}
        with self.lock:
            states = self._states
        channels = []
        for channel, reading in sorted((k, v) for k, v in frame.items() if isinstance(k, int)):
            channels.append({
                "ch": channel,
                "x": reading["x"],
                "y": reading["y"],
                "z": reading["z"],
                "field_ut": round(reading["field_ut"], 2) if "field_ut" in reading else None,
                "ok": reading.get("ok", True),
                "age": round(reading.get("age", 99), 2),
                "reads": reading.get("reads"),
            })
        bays = []
        for bay in BAYS:
            entry = states.get(bay) or {}
            delta = entry.get("delta_ut")
            threshold = self.baseline.threshold_ut if self.baseline else 3.8
            bays.append({
                "bay": bay,
                "channel": self.mapping.get(bay),
                "state": entry.get("state", "UNKNOWN"),
                "delta_ut": None if delta is None else round(delta, 2),
                "detail": entry.get("reason", ""),
                "bar": 0.0 if delta is None else min(1.0, delta / max(threshold, 0.1)),
            })
        meta = frame.get("_meta") or {}
        return {
            "board": self.board,
            "board_ok": not meta.get("error") and bool(channels),
            "error": meta.get("error"),
            "polls": meta.get("polls", 0),
            "duplicates": meta.get("duplicates", 0),
            "capturing": self.capturing,
            "capture_note": self.capture_note,
            "threshold_ut": self.baseline.threshold_ut if self.baseline else None,
            "hysteresis_ut": self.baseline.hysteresis_ut if self.baseline else None,
            "baseline": {
                "loaded": self.baseline is not None,
                "calibrated_at": self.baseline.calibrated_at if self.baseline else None,
                "source": self.baseline.source if self.baseline else None,
                "valid": sum(1 for e in (self.baseline.bays.values() if self.baseline else [])
                             if e.get("valid")),
                "total": len(self.baseline.bays) if self.baseline else 0,
            },
            "bays": bays,
            "channels": channels,
            "mapping": self.mapping,
            # Which code is running, so a stale page or a stale backend is visible on the screen
            # instead of being mistaken for a measurement.
            "build": BUILD,
        }


# ---------------------------------------------------------------------------- HTTP

def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):          # keep the terminal for real messages
            pass

        def _send(self, status, body, content_type):
            payload = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path.startswith("/api/state"):
                self._send(200, json.dumps(state.snapshot()), "application/json")
                return
            if self.path in ("/", "/index.html"):
                try:
                    with open(HTML_PATH, encoding="utf-8") as handle:
                        payload = handle.read().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Cache-Control", "no-store")   # always the current page
                    self.end_headers()
                    self.wfile.write(payload)
                except OSError as exc:
                    self._send(500, f"cannot read {HTML_PATH}: {exc}", "text/plain")
                return
            self._send(404, "not found", "text/plain")

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except ValueError:
                self._send(400, json.dumps({"ok": False, "error": "bad json"}), "application/json")
                return
            if self.path.startswith("/api/baseline"):
                ok, note = state.start_capture(float(body.get("seconds", 3.0)))
                self._send(200 if ok else 409, json.dumps({"ok": ok, "note": note}),
                           "application/json")
                return
            if self.path.startswith("/api/map"):
                pairs = {bay: (None if value in (None, "", "null") else int(value))
                         for bay, value in (body.get("pairs") or {}).items()}
                state.save_mapping(pairs)
                self._send(200, json.dumps({"ok": True, "mapping": state.mapping}),
                           "application/json")
                return
            self._send(404, json.dumps({"ok": False}), "application/json")

    return Handler


def load_mapping():
    mapping = {bay: None for bay in BAYS}
    try:
        with open(MAPPING_PATH, encoding="utf-8") as handle:
            for bay, entry in (json.load(handle).get("bays") or {}).items():
                mapping[bay] = (entry or {}).get("channel")
    except (OSError, ValueError):
        pass
    return mapping


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", help="board address, e.g. 192.168.3.118 (default: discover)")
    parser.add_argument("--port", type=int, default=8040)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)

    board, origin = find_board(args.host)
    if board is None:
        print("error: no board answering /api/mag was found.", file=sys.stderr)
        print("       pass the address explicitly, for example:", file=sys.stderr)
        print("         ./tools/mag-test.sh --host 192.168.3.118", file=sys.stderr)
        return 2
    print(f"board      {board}  ({origin})")
    print(f"build      {BUILD}")
    if not mag_answers(board):
        print(f"error: {board}/api/mag does not answer; the page will show UNKNOWN",
              file=sys.stderr)

    state = TestState(board, f"{board}/stream", load_mapping())
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"page       {url}")
    print("           clear B1-B4, press '建立 Baseline', then move one car at a time")
    if not args.no_open:
        try:
            import subprocess
            subprocess.Popen(["open", url])
        except Exception:                                    # noqa: BLE001
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        state.stop()
        if state.poller:
            state.poller.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
