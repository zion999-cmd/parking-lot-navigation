"""The route endpoint: a thin wrapper, tested over real HTTP.

Everything here goes through a real socket. The handler is small enough that calling its internals
would test nothing worth testing, and the point of this layer is exactly the HTTP part - the
status, the body, and the promise that it changes nothing except the route document.

Two of these tests exist because of the requirements rather than the behaviour: the map and the
state files must be byte-identical after a call, and the document that comes back must be the same
shape src/plan.py writes, since the route contract is not this layer's to change.

Run:  python tests/test_api.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import route                                                    # noqa: E402
import serve                                                    # noqa: E402

W, H = 800, 600


def a_map() -> dict:
    """A road with two bays on it, one bay with no access, and a one-way edge."""
    return {
        "schema": "parking_map/1",
        "coordinate_system": {"type": "image_2d", "width": W, "height": H,
                             "origin": "top_left"},
        "spots": [
            {"id": "38", "polygon": [[80, 100], [130, 100], [130, 200], [80, 200]],
             "access": {"point": [105, 200], "road_node": "r2"}},
            {"id": "39", "polygon": [[180, 100], [230, 100], [230, 200], [180, 200]],
             "access": {"point": [205, 200], "road_node": "r3"}},
            {"id": "40", "polygon": [[280, 100], [330, 100], [330, 200], [280, 200]],
             "access": None},
        ],
        "road_network": {
            "nodes": [
                {"id": "r1", "x": 0, "y": 300, "type": "entrance"},
                {"id": "r2", "x": 100, "y": 300, "type": "junction"},
                {"id": "r3", "x": 200, "y": 300, "type": "junction"},
                {"id": "r4", "x": 300, "y": 300, "type": "junction"},
            ],
            "edges": [
                {"from": "r1", "to": "r2", "distance": 100.0, "bidirectional": True},
                {"from": "r2", "to": "r3", "distance": 100.0, "bidirectional": True},
                {"from": "r3", "to": "r4", "distance": 100.0, "bidirectional": False},
            ],
        },
        "entrances": [{"id": "main_gate", "node": "r1", "type": "entry_exit"}],
    }


def a_state(states: dict, visible: bool = True) -> dict:
    return {
        "schema": "parking_state/1",
        "timestamp": 1791283027,
        "spots": {sid: {"state": st} for sid, st in states.items()},
        "vehicle": ({"id": "hero", "x": 10.0, "y": 300.0, "heading": 0.0, "visible": True,
                     "heading_is_axis_only": True, "front_is_known": True,
                     "front_heading": 0.0}
                    if visible else {"id": "hero", "visible": False}),
    }


@contextmanager
def a_server(map_doc: dict, state_doc: dict | None, extra_static: bool = True):
    """A real server on a random port, with the documents in a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp:
        map_path = os.path.join(tmp, "parking_map.json")
        state_path = os.path.join(tmp, "parking_state.json")
        route_path = os.path.join(tmp, "route.json")
        with open(map_path, "w", encoding="utf-8") as fh:
            json.dump(map_doc, fh)
        if state_doc is not None:
            with open(state_path, "w", encoding="utf-8") as fh:
                json.dump(state_doc, fh)
        if extra_static:
            with open(os.path.join(tmp, "hello.txt"), "w", encoding="utf-8") as fh:
                fh.write("static files are served from here\n")

        server = serve.serve(0, root=tmp, map_path=map_path, state_path=state_path,
                             route_path=route_path)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield (f"http://127.0.0.1:{server.server_address[1]}", route_path,
                   map_path, state_path)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


def get(url: str):
    """Returns (status, parsed body). A 4xx/5xx is a result, not an exception."""
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def digest(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------

def test_clicking_an_empty_bay_returns_a_route() -> None:
    with a_server(a_map(), a_state({"38": "EMPTY", "39": "EMPTY", "40": "EMPTY"})) as (base, _, _, _):
        status, body = get(f"{base}/api/route?spot=38")
        assert status == 200, status
        assert body["ok"] is True, body
        assert body["target_spot"] == "38"
        assert body["start_node"] == "r1" and body["target_node"] == "r2"
        assert body["nodes"] == ["r1", "r2"], body["nodes"]
        assert body["distance"] == 100.0


def test_the_returned_document_is_the_route_contract_unchanged() -> None:
    with a_server(a_map(), a_state({"38": "EMPTY"})) as (base, _, _, _):
        _, body = get(f"{base}/api/route?spot=38")
        assert body["schema"] == route.ROUTE_SCHEMA, body["schema"]
        for key in ("target_spot", "start_node", "target_node", "nodes", "distance",
                    "ok", "code", "reason", "generated_at"):
            assert key in body, key
        assert body["nodes"][0] == body["start_node"]
        assert body["nodes"][-1] == body["target_node"]


def test_the_api_writes_the_same_document_it_returns() -> None:
    with a_server(a_map(), a_state({"38": "EMPTY"})) as (base, route_path, _, _):
        _, body = get(f"{base}/api/route?spot=38")
        with open(route_path, encoding="utf-8") as fh:
            on_disk = json.load(fh)
        assert on_disk == body, "the page and src/plan.py would disagree"


def test_the_inputs_are_not_touched() -> None:
    """The map and the state are P0002's; this layer must not modify them."""
    with a_server(a_map(), a_state({"38": "EMPTY"})) as (base, _, map_path, state_path):
        before = (digest(map_path), digest(state_path))
        get(f"{base}/api/route?spot=38")
        get(f"{base}/api/route?spot=39")
        assert (digest(map_path), digest(state_path)) == before


# ---------------------------------------------------------------------------
# Refusals come back as refusals, with their own codes
# ---------------------------------------------------------------------------

def test_an_occupied_bay_returns_the_planners_refusal() -> None:
    with a_server(a_map(), a_state({"38": "OCCUPIED"})) as (base, _, _, _):
        status, body = get(f"{base}/api/route?spot=38")
        assert status == 200, "a refusal is an answer, not an HTTP error"
        assert body["ok"] is False
        assert body["code"] == route.NOT_FREE, body["code"]
        assert "OCCUPIED" in body["reason"]
        assert body["nodes"] == [] and body["distance"] is None


def test_an_unknown_bay_returns_the_planners_refusal() -> None:
    with a_server(a_map(), a_state({"38": "UNKNOWN"})) as (base, _, _, _):
        _, body = get(f"{base}/api/route?spot=38")
        assert body["ok"] is False and body["code"] == route.NOT_FREE
        assert "UNKNOWN" in body["reason"]


def test_a_bay_that_does_not_exist_returns_its_own_code() -> None:
    with a_server(a_map(), a_state({"38": "EMPTY"})) as (base, _, _, _):
        status, body = get(f"{base}/api/route?spot=999")
        assert status == 200
        assert body["code"] == route.NO_SUCH_SPOT, body["code"]


def test_a_bay_with_no_road_access_returns_its_own_code() -> None:
    with a_server(a_map(), a_state({"40": "EMPTY"})) as (base, _, _, _):
        _, body = get(f"{base}/api/route?spot=40")
        assert body["code"] == route.NO_ACCESS, body["code"]


def test_an_invisible_hero_returns_its_own_code() -> None:
    with a_server(a_map(), a_state({"38": "EMPTY"}, visible=False)) as (base, _, _, _):
        _, body = get(f"{base}/api/route?spot=38")
        assert body["code"] == route.NO_HERO, body["code"]


def test_an_unreachable_bay_returns_its_own_code() -> None:
    map_doc = a_map()
    map_doc["road_network"]["edges"] = [e for e in map_doc["road_network"]["edges"]
                                        if "r2" not in (e["from"], e["to"])]
    with a_server(map_doc, a_state({"38": "EMPTY"}, )) as (base, _, _, _):
        _, body = get(f"{base}/api/route?spot=38")
        assert body["code"] == route.NO_ROUTE, body["code"]


def test_the_codes_it_returns_are_the_planners_codes() -> None:
    """Nothing is translated or reworded on the way out."""
    with a_server(a_map(), a_state({"38": "OCCUPIED"})) as (base, _, map_path, state_path):
        _, body = get(f"{base}/api/route?spot=38")
        direct = route.plan(route.load_document(map_path), route.load_document(state_path), "38")
        assert body["code"] == direct.code
        assert body["reason"] == direct.reason


# ---------------------------------------------------------------------------
# Bad requests, and the static files
# ---------------------------------------------------------------------------

def test_a_request_with_no_bay_is_a_400_and_says_how_to_ask() -> None:
    with a_server(a_map(), a_state({"38": "EMPTY"})) as (base, _, _, _):
        status, body = get(f"{base}/api/route")
        assert status == 400, status
        assert "spot" in body["hint"], body


def test_a_missing_state_is_reported_as_such_rather_than_as_a_refusal() -> None:
    with a_server(a_map(), None) as (base, _, _, _):
        status, body = get(f"{base}/api/route?spot=38")
        assert status == 503, status
        assert body["error"] == "no state", body


def test_the_static_files_are_served_from_the_same_place() -> None:
    """One server for the page and the endpoint, so there is no CORS to arrange."""
    with a_server(a_map(), a_state({"38": "EMPTY"})) as (base, _, _, _):
        with urllib.request.urlopen(f"{base}/hello.txt", timeout=10) as response:
            assert response.status == 200
            assert b"static files" in response.read()


def test_nothing_is_cached() -> None:
    with a_server(a_map(), a_state({"38": "EMPTY"})) as (base, _, _, _):
        with urllib.request.urlopen(f"{base}/api/route?spot=38", timeout=10) as response:
            assert response.headers.get("Cache-Control") == "no-store"


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
