"""The smallest local service that lets the page ask for a route.

    python src/serve.py
    # then open http://localhost:8010/web/index.html and click a bay

It serves the static files and these small endpoints:

    GET /api/route?spot=38      ->  the route document, as src/plan.py writes it
    GET /api/route?clear=1      ->  the same document, saying there is no target any more
    GET /api/qr                 ->  current /go URL and QR module matrix

That is the whole of it. The handler reads the two contract documents, calls route.plan() and
writes out/route.json, the same call src/plan.py makes. No vision, no bay selection, no state
logic: the refusals come from the planner with its own codes, unchanged.

The clear form exists because a route document is only ever written, never removed: the bay
lights read out/route.json to decide which pixel is yellow, so a navigation that simply stops
would otherwise leave a target lit forever. The page calls it when it cancels or arrives.

The HTTP status stays 200 when the route is refused, deliberately. The document says whether
there is a route, through ok/code/reason; wrapping that in a status code would give two places to
look for one answer. A malformed request - no bay named - is a 400, which is a different thing
and worth distinguishing.

Why one server rather than an API plus `python -m http.server`: the page and its endpoint then
share an origin, so there is no CORS to arrange, one command to remember, and no way to start
half of it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import contract
import nav_lifecycle                                            # noqa: E402
import overview                                                 # noqa: E402
import mag_baseline                                             # noqa: E402
import qr_data                                                  # noqa: E402
import route                                                    # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
API_PATH = "/api/route"
INDEX_HTML = """<!doctype html><meta charset=utf-8><title>parking lot</title>
<style>
 body{font:15px/1.6 -apple-system,"PingFang SC",system-ui;margin:2em;background:#0f1115;color:#e6e9ef;max-width:900px}
 h1{font-size:19px;margin:0 0 2px} h2{font-size:14px;color:#93a0b4;margin:26px 0 6px;font-weight:600}
 a{color:#66d9ff;text-decoration:none} a:hover{text-decoration:underline}
 td{padding:3px 14px 3px 0;vertical-align:top} code{color:#d2a8ff}
 .n{color:#93a0b4}
</style>
<h1>停车场车位导航</h1>
<div class="n">三路感知：视觉 + 地磁 + RFID　·　本页是入口索引（文字可直接选中复制）</div>

<h2>页面</h2>
<table>
<tr><td><a href="/web/overview.html">/web/overview.html</a></td><td>三路总览：B1–B4 视觉 vs 地磁、RFID 标签、链路新鲜度</td></tr>
<tr><td><a href="/web/navigation.html">/web/navigation.html</a></td><td>车载导航：点空位出路线（手机可用）</td></tr>
<tr><td><a href="/web/board.html">/web/board.html</a></td><td>展示用：分析画面 + 点车位规划</td></tr>
<tr><td><a href="/web/index.html">/web/index.html</a></td><td>工程用：地图路线调试</td></tr>
<tr><td><a href="/go">/go</a></td><td>短跳转，面板上的二维码就用它</td></tr>
</table>

<h2>实时画面（MJPEG）</h2>
<table>
<tr><td><a href="http://127.0.0.1:8020/">:8020</a></td><td>标注流：车位的框、差图、车辆位置</td></tr>
<tr><td><a href="http://127.0.0.1:8021/">:8021</a></td><td>原始流：板子画面，未标注</td></tr>
</table>
<div class="n">手机看的话把 127.0.0.1 换成这台电脑的局域网地址。</div>

<h2>接口</h2>
<table>
<tr><td><code>/api/overview</code></td><td>三路汇总 JSON（视觉 + 地磁 + RFID + 各源新鲜度）</td></tr>
<tr><td><code>/api/route?spot=&lt;id&gt;</code></td><td>规划到某个车位；返回路线，灯带跟着黄灯</td></tr>
<tr><td><code>/api/route?clear=1</code></td><td>停止导航，黄灯跟着灭</td></tr>
<tr><td><code>/api/qr</code></td><td>当前导航地址的二维码矩阵（板子面板用）</td></tr>
</table>

<h2>板子直接接口（视觉/地磁/RFID 的原始数据）</h2>
<table>
<tr><td><code>GET /api/mag</code></td><td>4 颗地磁的 XYZ、field_ut、量程、reads/errors</td></tr>
<tr><td><code>GET /api/rfid</code></td><td>EPC、RSSI、端口、读取次数、四端口统计</td></tr>
<tr><td><code>GET /stream</code></td><td>摄像头 MJPEG（只允许一个客户端：用上面的 8020/8021 转播）</td></tr>
<tr><td><code>POST /api/ota/check</code></td><td>触发一次固件升级检查</td></tr>
</table>

<h2>终端工具</h2>
<table>
<tr><td><code>./start.sh</code></td><td>起整栈（serve + vision + relay + beacon + ota），<code>status</code> / <code>stop --all</code></td></tr>
<tr><td><code>./run.sh</code></td><td>手动跑 main.py（有窗口，可按 r 采参考图）</td></tr>
<tr><td><code>./tools/rfid-test.sh</code></td><td>RFID 存在检测：当前读到哪些标签、PRESENT/ABSENT</td></tr>
<tr><td><code>./tools/mag-test.sh</code></td><td>地磁测试页（:8040）：四路卡片、一键建 Baseline</td></tr>
<tr><td><code>./tools/rfid/rssi_survey.py</code></td><td>RSSI 可分性测量（判断能不能靠 RSSI 分区）</td></tr>
<tr><td><code>./tools/bench/mag_map.py</code></td><td>测车位→地磁通道映射（不假设 B1=CH0）</td></tr>
</table>
"""

OVERVIEW_PATH = "/api/overview"
OVERVIEW_PAGE = "/web/overview.html"
MAG_BASELINE_PATH = os.path.join(ROOT, "data", "mag_baseline.json")
RFID_TOOL_DIR = os.path.abspath(os.path.join(ROOT, "..", "tools", "rfid"))
QR_PATH = "/api/qr"
GO_PATH = "/go"
NAV_PATH = "/web/navigation.html"


class OverviewSources:
    """Board-facing collectors for the overview page.

    Reuses what already works instead of adding a second implementation: mag_baseline owns the
    magnetometer baseline and judgement, tools/rfid/rfid_presence.py owns "is this tag still here".
    Both are read-only; nothing here writes to main.py's state, the baseline or the reader.
    """

    def __init__(self):
        self.board = _find_board()
        self.poller = None
        self.baseline = None
        self.judge = None
        self.mapping = {}
        self.presence = None
        self.presence_module = None
        if self.board:
            try:
                self.poller = mag_baseline.MagPoller(f"{self.board}/api/mag")
                self.poller.start()
            except Exception:                          # noqa: BLE001
                self.poller = None
        self.baseline = mag_baseline.MagBaseline.load(MAG_BASELINE_PATH)
        self.mapping = _load_mapping()
        self.judge = mag_baseline.MagJudge(self.baseline)
        # The judgement runs on its own beat, not on page requests: the 1 s window needs several
        # samples, and a page that polls once a second would otherwise show UNKNOWN forever.
        self._states = {}
        self._stop = threading.Event()
        threading.Thread(target=self._tick, daemon=True, name="overview-mag").start()
        try:
            sys.path.insert(0, RFID_TOOL_DIR)
            import rfid_presence                        # noqa: PLC0415 - optional, tools-side
            self.presence_module = rfid_presence
            self.presence = rfid_presence.Presence()
        except Exception:                              # noqa: BLE001
            self.presence = None

    def _tick(self, period=0.25):
        while not self._stop.is_set():
            try:
                if self.poller is not None and self.judge is not None:
                    now = time.time()
                    self._states = self.judge.update(self.poller.snapshot(now), now)
            except Exception:                          # noqa: BLE001
                pass
            self._stop.wait(period)

    def magnetometers(self, now):
        if self.poller is None:
            return None, self.mapping, self.baseline
        return self._states, self.mapping, self.baseline

    def rfid(self):
        if not self.board:
            return None
        document = _http_json(f"{self.board}/api/rfid")
        if self.presence is not None:
            self.presence.update(document)
        return document


def _http_json(url, timeout=2.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:                                  # noqa: BLE001
        return None


def _find_board():
    """Same discovery the other tools use: run/camera.txt first, then the usual address."""
    cache = os.path.join(os.path.dirname(ROOT), "run", "camera.txt")
    try:
        with open(cache, encoding="utf-8") as handle:
            cached = handle.read().strip()
        if "://" in cached:
            return cached.split("://", 1)[0] + "://" + cached.split("://", 1)[1].split("/", 1)[0]
    except OSError:
        pass
    return "http://192.168.3.118"


def _load_mapping():
    path = os.path.join(ROOT, "config", "mag_bays.json")
    mapping = {bay: None for bay in ("B1", "B2", "B3", "B4")}
    try:
        with open(path, encoding="utf-8") as handle:
            for bay, entry in (json.load(handle).get("bays") or {}).items():
                mapping[bay] = (entry or {}).get("channel")
    except (OSError, ValueError):
        pass
    return mapping


class Handler(SimpleHTTPRequestHandler):
    """Static files, plus /api/route. Paths are set on the class by serve()."""

    map_path = os.path.join(ROOT, "out", "parking_map.json")
    state_path = os.path.join(ROOT, "out", "parking_state.json")
    route_path = os.path.join(ROOT, "out", "route.json")

    # No caching anywhere: the page polls the state and the route, and a cached contract would
    # be the one thing worse than a stale one - a stale one you cannot tell is stale.
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        # The default writes to stderr on every icon request; this keeps one line per API call.
        if self.path.startswith(API_PATH):
            sys.stderr.write("  api %s\n" % (fmt % args))

    def do_GET(self) -> None:                                   # noqa: N802  (stdlib naming)
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            return self._index()
        if parsed.path == GO_PATH:
            return self._go()
        if parsed.path == QR_PATH:
            return self._qr()
        if parsed.path == OVERVIEW_PATH:
            return self._overview()
        if parsed.path != API_PATH:
            return super().do_GET()
        self._api_route(urllib.parse.parse_qs(parsed.query))

    def _overview(self) -> None:
        """Vision, magnetometer and RFID in one document, each with its own freshness."""
        try:
            document = self._overview_document()
        except Exception as exc:                       # noqa: BLE001 - the page must still load
            return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
        return self._json(200, document)

    def _overview_document(self) -> dict:
        now = time.time()
        try:
            with open(self.state_path, encoding="utf-8") as handle:
                state_doc = json.load(handle)
        except (OSError, ValueError):
            state_doc = None

        sources = self._sources()
        mag_states, mapping, baseline = sources.magnetometers(now)
        rfid_doc = sources.rfid()
        return overview.build(state_doc, mag_states, mapping, baseline, rfid_doc,
                              sources.presence, now=now, state_path=self.state_path)

    def _sources(self):
        """The board-facing collectors, created once per server rather than per request."""
        if getattr(self.server, "overview_sources", None) is None:
            self.server.overview_sources = OverviewSources()
        return self.server.overview_sources

    def _qr(self) -> None:
        """Return the current PC's navigation URL and its QR module matrix."""
        if not qr_data.available():
            return self._json(503, {
                "error": "the QR library is not installed in this interpreter",
                "hint": "pip install qrcode in the python that runs serve.py, "
                        "or run serve.py with the vision-dev interpreter",
                "interpreter": sys.executable})
        try:
            payload = qr_data.navigation_url(self.headers.get("Host", ""))
            document = qr_data.document(payload)
        except ValueError as exc:
            return self._json(400, {"error": str(exc),
                                    "hint": "use this PC's LAN address in the Host header"})
        self._json(200, document)

    def _go(self) -> None:
        """Short path for the QR code on the gate's panel.

        The panel holds a baked QR matrix, and a shorter payload means a lower
        version means bigger modules means a phone can read it from further away.
        /web/navigation.html is 45 characters and lands on version 3 (6 px per
        module on a 240 px panel); /go is 28 and lands on version 2 (7 px). Same
        code, one pixel larger, for a string nobody reads.

        A redirect rather than a copy of the page so there is still exactly one
        navigation page.
        """
        self.send_response(302)
        self.send_header("Location", NAV_PATH)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _index(self) -> None:
        """One page listing everything this server and its neighbours actually offer.

        It drifts out of date the moment a feature is added somewhere else, which is what happened:
        /web/overview.html and the two MJPEG relays existed but were only discoverable by reading
        the code. Everything below is a real path, and the tools are named with the command that
        runs them.
        """
        body = INDEX_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _api_route(self, query: dict) -> None:
        # "No target" has to be sayable. out/route.json is otherwise write-only, and the bay
        # lights would show the last target forever after a navigation ends.
        if (query.get("clear") or [""])[0] in ("1", "true", "yes"):
            document = route.navigation_ended().as_document()
            try:
                contract.write_json(self.route_path, document)
            except OSError as exc:
                sys.stderr.write(f"  could not write {self.route_path}: {exc}\n")
            return self._json(200, document)

        spot = (query.get("spot") or [""])[0].strip()
        if not spot:
            return self._json(400, {"error": "no spot named",
                                    "hint": f"GET {API_PATH}?spot=<bay id>"})

        map_doc = route.load_document(self.map_path)
        if not map_doc:
            return self._json(503, {"error": "no map",
                                    "hint": f"run src/emit_map.py (looked at {self.map_path})"})
        state_doc = route.load_document(self.state_path)
        if not state_doc:
            return self._json(503, {"error": "no state",
                                    "hint": f"run src/main.py (looked at {self.state_path})"})

        # The existing planning call, unchanged. Refusals are answers and are returned as they
        # are, with their own codes.
        planned = route.plan(map_doc, state_doc, spot)
        document = planned.as_document()
        try:
            contract.write_json(self.route_path, document)
        except OSError as exc:
            sys.stderr.write(f"  could not write {self.route_path}: {exc}\n")
        self._json(200, document)

    def _json(self, status: int, body: dict) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def start_lifecycle(state_path: str, route_path: str, period: float = 0.5):
    """End a navigation when the tracked vehicle settles, with or without a browser open.

    The page does this itself while it is open, and did so before this existed - which is why the
    target survived whenever nobody was looking at the navigation tab. Running it in the server makes
    the ending independent of who is watching.

    Returns the thread so a test can stop it.
    """
    lifecycle = nav_lifecycle.NavLifecycle(state_path, route_path)

    def loop():
        while True:
            try:
                outcome = lifecycle.step()
                if outcome:
                    sys.stderr.write("  nav ended: %s (%s)\n" % (outcome["ended"], outcome["reason"]))
            except Exception as exc:                   # noqa: BLE001 - never take the server down
                sys.stderr.write("  nav lifecycle: %s: %s\n" % (type(exc).__name__, exc))
            time.sleep(period)

    thread = threading.Thread(target=loop, daemon=True, name="nav-lifecycle")
    thread.lifecycle = lifecycle
    thread.start()
    return thread


def serve(port: int = 8010, root: str = ROOT, map_path: str = None,
          state_path: str = None, route_path: str = None,
          host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Build the server without starting it, so a test can drive it on a random port."""
    handler = type("BoundHandler", (Handler,), {
        "map_path": map_path or Handler.map_path,
        "state_path": state_path or Handler.state_path,
        "route_path": route_path or Handler.route_path,
    })
    return ThreadingHTTPServer((host, port),
                               lambda *a, **kw: handler(*a, directory=root, **kw))


def main() -> int:
    ap = argparse.ArgumentParser(description="Serve the page and the route endpoint")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--host", default="127.0.0.1",
                    help="interface to bind; 0.0.0.0 is needed for the camera node's bay "
                         "lights to read out/parking_state.json, and it exposes the whole "
                         "--root directory to the LAN, so it is not the default")
    ap.add_argument("--root", default=ROOT, help="static files are served from here")
    ap.add_argument("--map", dest="map_path", default=os.path.join(ROOT, "out", "parking_map.json"))
    ap.add_argument("--state", dest="state_path",
                    default=os.path.join(ROOT, "out", "parking_state.json"))
    ap.add_argument("--route-out", dest="route_path",
                    default=os.path.join(ROOT, "out", "route.json"))
    args = ap.parse_args()

    server = serve(args.port, args.root, args.map_path, args.state_path, args.route_path,
                   args.host)
    start_lifecycle(args.state_path, args.route_path)
    print(f"page     http://localhost:{args.port}/web/index.html")
    print(f"endpoint GET http://localhost:{args.port}{API_PATH}?spot=<bay id>")
    print(f"map      {args.map_path}")
    print(f"state    {args.state_path}")
    if args.host in ("127.0.0.1", "localhost"):
        print("bound    loopback only - the camera node cannot reach this;"
              " use --host 0.0.0.0 to let the bay lights read the state")
    else:
        print(f"bound    {args.host}:{args.port} - reachable from the LAN,"
              f" which means all of {args.root} is too")
    print("ctrl-c to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
