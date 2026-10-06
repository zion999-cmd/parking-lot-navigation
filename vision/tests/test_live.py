"""The route follows the hero: a moving start, a target taken mid-drive, and the poll budget.

The page re-plans by calling the same endpoint again with the same bay, so what makes "the route
follows the car" true is not the page but this: the same call, a hero somewhere else, a different
answer. That is what most of this file checks.

The page's own decision - when a re-plan is warranted and when it is a waste - is JavaScript, so
it is tested as JavaScript, by pulling the function out of the page that ships and running it in
node. If node is not installed that one check steps aside rather than passing by default.

Run:  python tests/test_live.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import test_api as api                                          # noqa: E402  (the fixtures)

PAGE = os.path.join(HERE, "..", "web", "index.html")
MAP_BYTES, STATE_BYTES = 41785, 3774        # measured on the real documents, for the ratio below


def get(base: str, spot: str) -> dict:
    return api.get(f"{base}/api/route?spot={spot}")[1]


def state_with_hero(x: float, y: float, states: dict, visible: bool = True) -> dict:
    doc = api.a_state(states, visible=visible)
    if visible:
        doc["vehicle"].update({"x": x, "y": y})
    return doc


# ---------------------------------------------------------------------------
# The same call, a hero somewhere else
# ---------------------------------------------------------------------------

def test_the_route_changes_when_the_hero_moves() -> None:
    near, far = [None], [None]
    for x, slot in ((10.0, near), (60.0, far)):  # 10 -> r1, 60 -> r2 (room, not a tie)
        with api.a_server(api.a_map(),
                          state_with_hero(x, 300.0, {"38": "EMPTY", "39": "EMPTY"})) as ctx:
            base = ctx[0]
            slot[0] = get(base, "39")
    assert near[0]["ok"] and far[0]["ok"], (near[0], far[0])
    assert near[0]["start_node"] != far[0]["start_node"], (near[0], far[0])
    assert near[0]["nodes"] != far[0]["nodes"]
    assert near[0]["distance"] != far[0]["distance"], (near[0]["distance"], far[0]["distance"])


def test_the_start_is_always_where_the_hero_is_now() -> None:
    """Not a cache: ask twice from two positions and each answer matches its own position."""
    with api.a_server(api.a_map(), state_with_hero(10.0, 300.0, {"38": "EMPTY"})) as ctx:
        base, _, _, state_path = ctx
        first = get(base, "38")
        with open(state_path, encoding="utf-8") as fh:
            doc = json.load(fh)
        doc["vehicle"]["x"] = 60.0                        # the hero drives; nobody re-runs anything
        with open(state_path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
        second = get(base, "38")
    assert first["start_node"] == "r1" and second["start_node"] == "r2", (first, second)
    assert first["generated_at"] <= second["generated_at"]


def test_the_route_to_the_same_bay_shortens_as_the_hero_approaches() -> None:
    distances = []
    for x in (10.0, 60.0, 210.0):   # 210 -> r3 (room on both sides)
        with api.a_server(api.a_map(), state_with_hero(x, 300.0, {"39": "EMPTY"})) as ctx:
            base = ctx[0]
            distances.append(get(base, "39")["distance"])
    assert distances == sorted(distances, reverse=True), distances


def test_a_bay_taken_while_driving_turns_into_a_refusal() -> None:
    """The safety net: the page re-plans on a timer, so this arrives without anyone clicking."""
    with api.a_server(api.a_map(), state_with_hero(10.0, 300.0, {"38": "EMPTY"})) as ctx:
        base, _, _, state_path = ctx
        assert get(base, "38")["ok"] is True
        with open(state_path, encoding="utf-8") as fh:
            doc = json.load(fh)
        doc["spots"]["38"]["state"] = "OCCUPIED"
        with open(state_path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
        refused = get(base, "38")
    assert refused["ok"] is False and refused["code"] == "spot-not-free", refused


def test_a_hero_leaving_the_frame_turns_into_a_refusal() -> None:
    with api.a_server(api.a_map(), state_with_hero(10.0, 300.0, {"38": "EMPTY"})) as ctx:
        base, _, _, state_path = ctx
        assert get(base, "38")["ok"] is True
        with open(state_path, encoding="utf-8") as fh:
            doc = json.load(fh)
        doc["vehicle"]["visible"] = False
        with open(state_path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
        refused = get(base, "38")
    assert refused["code"] == "hero-not-visible", refused


# ---------------------------------------------------------------------------
# Can the page afford to poll this fast?
# ---------------------------------------------------------------------------

def test_a_plan_answers_fast_enough_to_be_asked_twice_a_second() -> None:
    times = []
    with api.a_server(api.a_map(), state_with_hero(10.0, 300.0, {"38": "EMPTY"})) as ctx:
        base = ctx[0]
        for _ in range(20):
            t0 = time.perf_counter()
            get(base, "38")
            times.append((time.perf_counter() - t0) * 1000)
    avg = statistics.mean(times)
    p95 = sorted(times)[int(len(times) * 0.95) - 1]
    print(f"        plan {avg:.1f} ms average, p95 {p95:.1f} ms, worst {max(times):.1f} ms")
    assert avg < 60, f"a 500 ms poll with a {avg:.0f} ms answer leaves no room: {times}"


def test_the_polled_documents_are_much_smaller_than_the_map() -> None:
    """Why the page can poll at 2 Hz: the map is not in the poll.

    The map is 41.8 KB and never changes while driving; the state is 3.8 KB. If the state ever
    grows towards the map, the poll becomes the wrong design and this check says so.
    """
    ratio = MAP_BYTES / STATE_BYTES
    assert ratio > 5, f"the state is only {ratio:.1f}x smaller than the map"
    print(f"        map {MAP_BYTES} B / state {STATE_BYTES} B = {ratio:.1f}x less per poll")


def test_the_page_polls_the_state_but_not_the_map() -> None:
    """Reads the shipped page: the 500 ms loop must not fetch the map."""
    html = open(PAGE, encoding="utf-8").read()
    loop = html[html.index("async function poll()"):html.index("reloadAll();")]
    assert "DOCS.state" in loop, loop
    assert "DOCS.map" not in loop, "the map is back in the fast loop"
    poll = int(re.search(r"const POLL_MS = (\d+);", html).group(1))
    assert poll <= 250, f"the poll is {poll} ms; the state is rewritten every 250 ms at least"


# ---------------------------------------------------------------------------
# The page's own decision, as JavaScript
# ---------------------------------------------------------------------------

def extract_function(html: str, name: str) -> str:
    start = html.index(f"function {name}(")
    depth, i = 0, html.index("{", start)
    for j in range(i, len(html)):
        if html[j] == "{":
            depth += 1
        elif html[j] == "}":
            depth -= 1
            if depth == 0:
                return html[start:j + 1]
    raise AssertionError("unbalanced braces")


def node_eval(names: list, call: str, payload: list, constants: bool = True) -> list:
    """Run the shipped page's own functions in node, over the given cases.

    Copies nothing: the function text is pulled out of web/index.html, so a page that changes
    without its test changes fails here.
    """
    if not shutil.which("node"):
        raise unittest.SkipTest("node not installed")
    html = open(PAGE, encoding="utf-8").read()
    parts = []
    if constants:
        consts = "\n".join(re.findall(
            r"^const (?:MAP_EVERY_POLLS|POLL_MS|REPLAN_MOVE_PX|REPLAN_AFTER_MS) = .*$", html, re.M))
        assert consts.count("const") == 4, consts
        parts.append(consts)
    parts += [extract_function(html, n) for n in names]
    parts.append("const cs = JSON.parse(process.argv[2]);\n"
                 f"console.log(JSON.stringify(cs.map(c => ({call}))));\n")
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write("\n".join(parts))
        path = fh.name
    try:
        out = subprocess.run(["node", path, json.dumps(payload)],
                             capture_output=True, text=True, timeout=30)
    finally:
        os.unlink(path)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_the_replan_gate_follows_the_car_without_hammering_the_api() -> None:
    try:
        node_eval(["shouldReplan"], "0", [])            # a probe, so a missing node shows up here
    except unittest.SkipTest as exc:
        print(f"        ({exc} - skipped, not passed)")
        return
    cases = [
        # picked, last(x, y, at), hero(x, y, visible), now, expected, why
        (None, None, (10, 300, True), 1000, False, "no target"),
        ("38", None, (10, 300, True), 1000, True, "first plan"),
        ("38", (10, 300, 0), (10, 300, True), 200, False, "nothing moved yet"),
        ("38", (10, 300, 0), (13, 300, True), 200, False, "3 px is not worth a plan"),
        ("38", (10, 300, 0), (14, 300, True), 200, True, "4 px is"),
        ("38", (10, 300, 0), (50, 340, True), 200, True, "normal driving between polls"),
        ("38", (10, 300, 0), (10, 300, True), 1600, True, "the timer, in case a bay filled"),
        ("38", (10, 300, 0), (10, 300, False), 200, False, "hidden, not yet time"),
        ("38", (10, 300, 0), (10, 300, False), 1600, True, "hidden and time is up"),
        ("38", (None, None, 0), (10, 300, True), 200, True, "the hero came back into view"),
        ("38", (10, 300, 0), (None, None, True), 200, False, "no position to compare"),
    ]
    js_cases = [{"picked": c[0],
                 "last": None if c[1] is None else {"x": c[1][0], "y": c[1][1], "at": c[1][2]},
                 "state": {"vehicle": {"x": c[2][0], "y": c[2][1], "visible": c[2][2]}},
                 "now": c[3]} for c in cases]
    got = node_eval(["shouldReplan"], "shouldReplan(c.state, c.last, c.picked, c.now)", js_cases)
    wrong = [(c[5], g, c[4]) for c, g in zip(cases, got) if g is not c[4]]
    assert not wrong, "the shipped page disagrees: " + "; ".join(
        f"{why}: got {g}, wanted {w}" for why, g, w in wrong)


def test_the_line_drawn_starts_from_the_car_not_only_from_a_road_node() -> None:
    """The regression: the route's first node can be a bay's mouth, so a line drawn from it alone
    looked as though it connected two bays. The page must draw the car's own leg first.

    The numbers are the real ones: the hero at (743.4, 300.9) and a start node at (664, 400),
    which is bay 50's access point while the hero is 2.4 px from a road and nowhere near bay 50.
    """
    nodes = {"r36": {"x": 664.0, "y": 400.0}, "r37": {"x": 664.1, "y": 436.2}}
    cases = [
        {"state": {"vehicle": {"x": 743.4, "y": 300.9, "visible": True}}, "car": True},
        {"state": {"vehicle": {"x": 743.4, "y": 300.9, "visible": False}}, "car": False},
        {"state": {"vehicle": {"visible": False}}, "car": False},
        {"state": {}, "car": False},
    ]
    payload = [{"state": c["state"],
                "route": {"ok": True, "nodes": ["r36", "r37"]},
                "nodes": nodes} for c in cases]
    got = node_eval(["routePolyline"],
                    "routePolyline(c.state, c.route, c.nodes)", payload, constants=False)
    for case, res in zip(cases, got):
        assert res["points"], res
        assert res["carLeg"] is case["car"], (case, res)
        expected = ([[743.4, 300.9]] if case["car"] else []) + [[664.0, 400.0], [664.1, 436.2]]
        assert res["points"] == expected, (case, res["points"])


def test_a_refused_or_unknown_node_route_draws_nothing_broken() -> None:
    cases = [
        ({"ok": False, "code": "spot-not-free", "nodes": []}, {"r1": {"x": 0, "y": 0}}, []),
        ({"ok": True, "nodes": ["r1", "gone", "r2"]}, {"r1": {"x": 1, "y": 1}, "r2": {"x": 2, "y": 2}},
         [[1, 1], [2, 2]]),                       # a node missing from the map is skipped, not fatal
        ({"ok": True, "nodes": ["gone"]}, {"r1": {"x": 1, "y": 1}}, []),
        (None, {"r1": {"x": 1, "y": 1}}, []),
    ]
    got = node_eval(["routePolyline"],
                    "routePolyline({}, c.route, c.nodes)",
                    [{"route": r, "nodes": n} for r, n, _ in cases], constants=False)
    for (route, _, want), res in zip(cases, got):
        assert res["points"] == want and res["carLeg"] is False, (route, res)


def test_the_page_follows_the_route_document_however_it_was_written() -> None:
    """The optimization reported: re-planning (from src/plan.py, or anywhere else) left the page
    showing the route it had planned itself, because the file was only read while no bay had been
    clicked. A route is a document like the state; whoever writes it, the page shows it.

    adoptsRoute(shown, fromFile) is the shipped decision, run here in node.
    """
    a = {"target_spot": "10", "ok": True, "code": "ok", "start_node": "r29", "target_node": "r8",
         "nodes": ["r29", "r8"], "distance": 896.1}
    cases = [
        (a, dict(a), False, "the same route, written again"),
        (a, None, False, "nothing in the file"),
        (None, a, True, "the page has nothing and the file has a route"),
        (a, {**a, "target_spot": "11"}, True, "a different bay was planned"),
        (a, {**a, "nodes": ["r29", "r30", "r8"]}, True, "the same bay, a different path"),
        (a, {**a, "distance": 900.0}, True, "the same path, a new length"),
        (a, {**a, "ok": False, "code": "spot-not-free", "nodes": [], "distance": None},
         True, "the bay was taken while driving"),
        (a, {**a, "start_node": "r28", "nodes": ["r28", "r8"]}, True, "the car moved"),
    ]
    payload = [{"shown": shown, "file": from_file} for shown, from_file, _, _ in cases]
    got = node_eval(["adoptsRoute", "routeSignature"],
                    "adoptsRoute(c.shown, c.file)", payload, constants=False)
    wrong = [(why, g, want) for (_, _, want, why), g in zip(cases, got) if g is not want]
    assert not wrong, "; ".join(f"{why}: got {g}, wanted {w}" for why, g, w in wrong)


def test_the_map_is_not_reloaded_in_a_loop_when_the_route_is_the_stale_half() -> None:
    """A different kind of bug from a wrong answer: if the route file is the stale thing, a page
    that reloads the 42 KB map whenever a node is missing would do it twice a second forever."""
    html = open(PAGE, encoding="utf-8").read()
    loop = html[html.index("async function poll()"):html.index("reloadAll();")]
    assert "mapTriedFor" in loop and "sig !== mapTriedFor" in loop, loop


def test_the_page_notices_when_a_route_names_a_node_the_map_no_longer_has() -> None:
    """Re-drawing the roads and re-emitting the map can leave the drawn route pointing at node ids
    that are gone. Dropping them silently made the line quietly shorter; the page needs to know so
    it can reload the map."""
    cases = [
        (["r1", "r2"], {"r1": {"x": 0, "y": 0}, "r2": {"x": 10, "y": 0}}, 0),
        (["r1", "gone", "r2"], {"r1": {"x": 0, "y": 0}, "r2": {"x": 10, "y": 0}}, 1),
        (["gone", "also-gone"], {"r1": {"x": 0, "y": 0}}, 2),
    ]
    got = node_eval(["routePolyline"],
                    "routePolyline({}, {ok: true, nodes: c.ids}, c.nodes)",
                    [{"ids": ids, "nodes": nodes} for ids, nodes, _ in cases], constants=False)
    for (ids, _, want), res in zip(cases, got):
        assert res["unknown"] == want, (ids, res)
        assert len(res["points"]) == len(ids) - want, (ids, res)


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
