"""The smallest local service that lets the page ask for a route.

    python src/serve.py
    # then open http://localhost:8010/web/index.html and click a bay

It serves the static files - the page and the contract documents - and exactly one endpoint:

    GET /api/route?spot=38      ->  the route document, as src/plan.py writes it

That is the whole of it. The handler reads the two contract documents, calls route.plan() and
writes out/route.json, the same call src/plan.py makes. No vision, no bay selection, no state
logic: the refusals come from the planner with its own codes, unchanged.

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
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import contract                                                 # noqa: E402
import route                                                    # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
API_PATH = "/api/route"


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
        if parsed.path != API_PATH:
            return super().do_GET()
        self._api_route(urllib.parse.parse_qs(parsed.query))

    def _api_route(self, query: dict) -> None:
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


def serve(port: int = 8010, root: str = ROOT, map_path: str = None,
          state_path: str = None, route_path: str = None) -> ThreadingHTTPServer:
    """Build the server without starting it, so a test can drive it on a random port."""
    handler = type("BoundHandler", (Handler,), {
        "map_path": map_path or Handler.map_path,
        "state_path": state_path or Handler.state_path,
        "route_path": route_path or Handler.route_path,
    })
    return ThreadingHTTPServer(("127.0.0.1", port),
                               lambda *a, **kw: handler(*a, directory=root, **kw))


def main() -> int:
    ap = argparse.ArgumentParser(description="Serve the page and the route endpoint")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--root", default=ROOT, help="static files are served from here")
    ap.add_argument("--map", dest="map_path", default=os.path.join(ROOT, "out", "parking_map.json"))
    ap.add_argument("--state", dest="state_path",
                    default=os.path.join(ROOT, "out", "parking_state.json"))
    ap.add_argument("--route-out", dest="route_path",
                    default=os.path.join(ROOT, "out", "route.json"))
    args = ap.parse_args()

    server = serve(args.port, args.root, args.map_path, args.state_path, args.route_path)
    print(f"page     http://localhost:{args.port}/web/index.html")
    print(f"endpoint GET http://localhost:{args.port}{API_PATH}?spot=<bay id>")
    print(f"map      {args.map_path}")
    print(f"state    {args.state_path}")
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
