"""The navigation layer: routes to a named bay, from the contract documents alone.

The point of these tests is not only that the planner works, but that it works *from the
contract*: every map and state below is built by hand right here, with no camera, no OpenCV and
no road network internals. If route.py ever reached back into P0002 to find its input, these
tests would stop compiling.

Run:  python tests/test_route.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import route                                                    # noqa: E402

W, H = 800, 600


def a_map() -> dict:
    """Four road nodes in a row, a one-way detour, and two bays hanging off it.

    r1 --- r2 --- r3 --- r4            bays: 38 off r2, 39 off r3
           \\             /
            r5 --- r6 ---           one-way from r5 to r6 only
    """
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
                {"id": "r5", "x": 100, "y": 400, "type": "junction"},
                {"id": "r6", "x": 200, "y": 400, "type": "junction"},
            ],
            "edges": [
                {"from": "r1", "to": "r2", "distance": 100.0, "bidirectional": True},
                {"from": "r2", "to": "r3", "distance": 100.0, "bidirectional": True},
                {"from": "r3", "to": "r4", "distance": 100.0, "bidirectional": True},
                {"from": "r2", "to": "r5", "distance": 100.0, "bidirectional": True},
                {"from": "r3", "to": "r6", "distance": 100.0, "bidirectional": True},
                # one-way: r5 -> r6 only, so it is a long way round from r6 to r5
                {"from": "r5", "to": "r6", "distance": 100.0, "bidirectional": False},
            ],
        },
        "entrances": [{"id": "main_gate", "node": "r1", "type": "entry_exit"}],
    }


def a_state(spot_states: dict, hero=(10.0, 300.0), visible=True, location=None) -> dict:
    spots = {sid: {"state": st} for sid, st in spot_states.items()}
    vehicle = ({"id": "hero", "x": hero[0], "y": hero[1], "heading": 0.0,
                "visible": visible, "heading_is_axis_only": True,
                "front_is_known": True, "front_heading": 0.0}
               if visible else {"id": "hero", "visible": False})
    if location is not None:
        vehicle["location"] = location
    return {
        "schema": "parking_state/1",
        "timestamp": 1791283027,
        "spots": spots,
        "vehicle": vehicle,
    }


def a_map_like_the_real_one() -> dict:
    """The shape that produced the bug: a sparse vertical lane, a denser aisle, and a bay whose
    access point is a short branch off the aisle.

        v1 (400,100)
         |                       car at (400,240): v1 is 140px, v2 is 160px,
         |                       and b1 - bay 38's mouth - is 125px, so the nearest
         |                       *node* is a bay the car is nowhere near.
         |                       (On the real map this was 127px against 135px.)
        v2 (400,400) ===== a2 (384,400) ===== a1 (200,400)
                            |
                            b1 (384,364)   bay 38's access point, 36px off the aisle
    """
    return {
        "schema": "parking_map/1",
        "coordinate_system": {"type": "image_2d", "width": W, "height": H,
                             "origin": "top_left"},
        "spots": [
            {"id": "38", "polygon": [[360, 300], [408, 300], [408, 364], [360, 364]],
             "access": {"point": [384, 364], "road_node": "b1"}},
        ],
        "road_network": {
            "nodes": [
                {"id": "v1", "x": 400, "y": 100, "type": "entrance"},
                {"id": "v2", "x": 400, "y": 400, "type": "junction"},
                {"id": "a2", "x": 384, "y": 400, "type": "junction"},
                {"id": "a1", "x": 200, "y": 400, "type": "junction"},
                {"id": "b1", "x": 384, "y": 364, "type": "spot_access"},
            ],
            "edges": [
                {"from": "v1", "to": "v2", "distance": 300.0, "bidirectional": True},
                {"from": "v2", "to": "a2", "distance": 16.0, "bidirectional": True},
                {"from": "a2", "to": "a1", "distance": 184.0, "bidirectional": True},
                {"from": "a2", "to": "b1", "distance": 36.0, "bidirectional": True},
            ],
        },
        "entrances": [{"id": "main_gate", "node": "v1", "type": "entry_exit"}],
    }


# ---------------------------------------------------------------------------
# Where the car is taken to be standing
# ---------------------------------------------------------------------------

def test_a_car_on_the_lane_does_not_start_at_a_bays_branch() -> None:
    """The reported bug: the line looked as though it left a bay the car was never in."""
    state = a_state({"38": "EMPTY"}, hero=(400.0, 240.0),
                    location={"kind": "road", "id": None, "distance_to_road": 2.4})
    planned = route.plan(a_map_like_the_real_one(), state, "38")
    assert planned.ok, planned.reason
    nearest = route.nearest_node(a_map_like_the_real_one(), 400.0, 240.0)
    assert nearest[0] == "b1", "the fixture no longer reproduces the bug"
    assert planned.start_node == "v1", planned.start_node      # 140px, on the lane the car is on
    assert planned.nodes[0] == "v1" and planned.nodes[-1] == "b1"


def test_a_car_reported_in_a_bay_starts_at_that_bays_own_mouth() -> None:
    """The exception: standing in a bay, its access point is exactly where the route starts."""
    state = a_state({"38": "EMPTY"}, hero=(384.0, 340.0),
                    location={"kind": "spot", "id": "38", "distance_to_road": None})
    planned = route.plan(a_map_like_the_real_one(), state, "38")
    assert planned.ok, planned.reason
    assert planned.start_node == "b1", planned.start_node


def test_a_lane_node_that_a_bay_also_uses_is_still_a_place_to_start() -> None:
    """Only dead ends are skipped. Where the lane passes through, the mouth and the lane are the
    same place and there is nothing wrong with starting there."""
    planned = route.plan(a_map(), a_state({"38": "EMPTY"}, hero=(100.0, 300.0),
                                          location={"kind": "road", "id": None,
                                                    "distance_to_road": 0.5}), "38")
    assert planned.ok, planned.reason
    assert planned.start_node == "r2", planned.start_node      # bay 38's access, and the lane


def test_skipping_never_turns_a_route_into_a_refusal() -> None:
    """If every node were a branch there would be nothing left to skip to; the nearest node is
    then used rather than refusing a route that plainly exists."""
    only_branches = a_map_like_the_real_one()
    for edge in only_branches["road_network"]["edges"]:          # make b1 the whole network
        edge["from"] = edge["to"] = "b1"
    only_branches["road_network"]["nodes"] = [
        {"id": "b1", "x": 384, "y": 364, "type": "spot_access"}]
    planned = route.plan(only_branches, a_state({"38": "EMPTY"}, hero=(384.0, 300.0)), "38")
    assert planned.ok, planned.reason
    assert planned.start_node == "b1"


def test_the_branch_rule_reads_the_map_not_a_hard_coded_name() -> None:
    """b1 is only a branch because the map says a bay uses it and nothing passes through it."""
    doc = a_map_like_the_real_one()
    assert route.branch_access_nodes(doc) == frozenset({"b1"})
    assert route.branch_access_nodes(a_map()) == frozenset()     # r2 and r3 carry the lane through


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------

def test_a_route_is_planned_to_an_empty_bay() -> None:
    planned = route.plan(a_map(), a_state({"38": "EMPTY"}), "38")
    assert planned.ok, planned.reason
    assert planned.target_spot == "38"
    assert planned.start_node == "r1"        # the hero at (10,300) is nearest r1
    assert planned.target_node == "r2"       # bay 38 hangs off r2
    assert planned.nodes[0] == "r1" and planned.nodes[-1] == "r2"
    assert planned.distance == 100.0


def test_the_nodes_are_a_connected_chain_along_the_graph() -> None:
    planned = route.plan(a_map(), a_state({"39": "EMPTY"}), "39")
    assert planned.ok
    graph = route.adjacency_from_map(a_map())
    for a, b in zip(planned.nodes, planned.nodes[1:]):
        assert any(n == b for n, _ in graph[a]), f"{a} -> {b} is not an edge"


def test_the_distance_is_the_sum_of_the_edge_weights() -> None:
    planned = route.plan(a_map(), a_state({"39": "EMPTY"}), "39")
    assert planned.ok
    graph = route.adjacency_from_map(a_map())
    total = 0.0
    for a, b in zip(planned.nodes, planned.nodes[1:]):
        total += dict(graph[a])[b]
    assert abs(planned.distance - total) < 1e-6, (planned.distance, total)


def test_a_longer_route_is_chosen_when_it_is_the_shorter_one() -> None:
    """The hero starts on the far side of the one-way, so the detour is the only way."""
    planned = route.plan(a_map(), a_state({"38": "EMPTY"}, hero=(200.0, 400.0)), "38")
    assert planned.ok, planned.reason
    assert planned.start_node == "r6"
    # r6 -> r3 (100) -> r2 (100) = 200 going round; the direct r6 -> r5 edge does not exist.
    assert planned.distance == 200.0, planned.distance


# ---------------------------------------------------------------------------
# Refusals, each with its own code
# ---------------------------------------------------------------------------

def test_an_occupied_bay_is_refused() -> None:
    planned = route.plan(a_map(), a_state({"38": "OCCUPIED"}), "38")
    assert planned.ok is False
    assert planned.code == route.NOT_FREE, planned.code
    assert "OCCUPIED" in planned.reason, planned.reason
    assert planned.nodes == [] and planned.distance is None


def test_an_unknown_bay_is_refused_and_says_so_differently() -> None:
    planned = route.plan(a_map(), a_state({"38": "UNKNOWN"}), "38")
    assert planned.ok is False
    assert planned.code == route.NOT_FREE
    assert "UNKNOWN" in planned.reason, planned.reason
    assert "not" not in planned.reason.replace("not visible", ""), planned.reason


def test_a_bay_missing_from_the_state_is_treated_as_unknown() -> None:
    planned = route.plan(a_map(), a_state({}), "38")
    assert planned.ok is False and planned.code == route.NOT_FREE
    assert "UNKNOWN" in planned.reason


def test_a_bay_that_does_not_exist_is_refused() -> None:
    planned = route.plan(a_map(), a_state({"38": "EMPTY"}), "999")
    assert planned.ok is False and planned.code == route.NO_SUCH_SPOT


def test_a_bay_with_no_road_is_refused() -> None:
    planned = route.plan(a_map(), a_state({"40": "EMPTY"}), "40")
    assert planned.ok is False and planned.code == route.NO_ACCESS


def test_no_hero_means_no_route_and_not_a_guess() -> None:
    planned = route.plan(a_map(), a_state({"38": "EMPTY"}, visible=False), "38")
    assert planned.ok is False and planned.code == route.NO_HERO


def test_an_unreachable_bay_fails_explicitly() -> None:
    map_doc = a_map()
    # isolate r2: with any of its edges left it is still reachable round the other way
    map_doc["road_network"]["edges"] = [e for e in map_doc["road_network"]["edges"]
                                        if "r2" not in (e["from"], e["to"])]
    planned = route.plan(map_doc, a_state({"38": "EMPTY"}, hero=(300.0, 300.0)), "38")
    assert planned.ok is False
    assert planned.code == route.NO_ROUTE, planned.code
    assert "no path" in planned.reason


def test_a_one_way_edge_is_not_walked_backwards() -> None:
    """r5 -> r6 exists; r6 -> r5 must not, which is the whole reason for the flag."""
    graph = route.adjacency_from_map(a_map())
    assert ("r6", 100.0) in graph["r5"]
    assert all(n != "r5" for n, _ in graph["r6"]), graph["r6"]


def test_a_map_with_no_road_network_is_refused_rather_than_crashing() -> None:
    planned = route.plan({"spots": [{"id": "38", "access": {"road_node": "r1"}}]},
                         a_state({"38": "EMPTY"}), "38")
    assert planned.ok is False
    assert planned.code == route.NO_ROUTE


# ---------------------------------------------------------------------------
# The hero moves, the route follows
# ---------------------------------------------------------------------------

def test_moving_the_hero_changes_the_route() -> None:
    map_doc = a_map()
    near = route.plan(map_doc, a_state({"39": "EMPTY"}, hero=(10.0, 300.0)), "39")
    far = route.plan(map_doc, a_state({"39": "EMPTY"}, hero=(300.0, 300.0)), "39")
    assert near.ok and far.ok
    assert near.start_node == "r1" and far.start_node == "r4"
    assert near.distance != far.distance
    assert near.nodes != far.nodes


def test_the_nearest_node_is_used_for_a_car_between_nodes() -> None:
    planned = route.plan(a_map(), a_state({"38": "EMPTY"}, hero=(40.0, 300.0)), "38")
    assert planned.start_node == "r1"          # 40 is nearer 0 than 100
    planned = route.plan(a_map(), a_state({"38": "EMPTY"}, hero=(90.0, 300.0)), "38")
    assert planned.start_node == "r2"


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------

def test_the_route_document_has_the_agreed_shape() -> None:
    doc = route.plan(a_map(), a_state({"38": "EMPTY"}), "38").as_document()
    assert doc["schema"] == route.ROUTE_SCHEMA
    for key in ("target_spot", "start_node", "target_node", "nodes", "distance", "ok", "code"):
        assert key in doc, key
    assert isinstance(doc["nodes"], list) and doc["nodes"][0] == doc["start_node"]
    assert isinstance(doc["distance"], float)


def test_a_refused_route_is_still_a_valid_document() -> None:
    doc = route.plan(a_map(), a_state({"38": "OCCUPIED"}), "38").as_document()
    assert doc["ok"] is False
    assert doc["code"] == route.NOT_FREE
    assert doc["reason"]
    assert doc["nodes"] == [] and doc["distance"] is None


def test_a_missing_document_reads_as_empty_rather_than_raising() -> None:
    assert route.load_document("/nonexistent/parking_map.json") == {}


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
