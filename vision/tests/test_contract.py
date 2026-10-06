"""The P0002 output contract: parking_map.json and parking_state.json.

This is the boundary the rest of the system is built on, so the tests here are about the
shape of the two documents and about the promises they make:

    the map says where the bays are and how a car reaches them
    the state says what is true now, or admits that it does not know

The second half matters as much as the first. A bay whose state is unknown - no reference
frame, no frames arriving, a hidden bay - must never be reported as EMPTY.

Run:  python tests/test_contract.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import contract                                                 # noqa: E402
from road_network import (KIND_ENTRANCE, KIND_SPOT,              # noqa: E402
                          KIND_WAYPOINT, RoadNetwork)
from spot_config import Spot                                    # noqa: E402

W, H = 900, 600


def a_bay(bay_id: str, x0: float, y0: float, x1: float, y1: float) -> Spot:
    return Spot(id=bay_id, polygon=np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                                            dtype=float))


def two_bays() -> list:
    return [a_bay("38", 100, 100, 150, 260), a_bay("39", 160, 100, 210, 260)]


def roads_with_aisle() -> RoadNetwork:
    net = RoadNetwork(shape=(H, W))
    a = net.add_node(0, 400, kind=KIND_ENTRANCE)
    b = net.add_node(880, 400)
    net.connect(a.id, b.id)
    return net


class FakeResult:
    def __init__(self, state: str, ratio: float = 0.0):
        self.state = state
        self.changed_ratio = ratio


# ---------------------------------------------------------------------------
# The static map
# ---------------------------------------------------------------------------

def test_the_map_declares_one_coordinate_system_for_everything() -> None:
    doc = contract.build_map(two_bays(), roads_with_aisle().derive(two_bays()), (H, W))
    cs = doc["coordinate_system"]
    assert cs == {"type": "image_2d", "width": W, "height": H, "origin": "top_left"}, cs


def test_the_map_gives_every_bay_an_access_point_on_a_road_node() -> None:
    bays = two_bays()
    graph = roads_with_aisle().derive(bays)
    doc = contract.build_map(bays, graph, (H, W))

    assert [s["id"] for s in doc["spots"]] == ["38", "39"]
    node_ids = {n["id"] for n in doc["road_network"]["nodes"]}
    for spot in doc["spots"]:
        access = spot["access"]
        assert access, "a bay with a road in front of it must have an access point"
        assert len(access["point"]) == 2
        assert access["road_node"] in node_ids, access
        assert len(spot["polygon"]) == 4


def test_the_access_point_is_on_the_side_facing_the_road() -> None:
    bays = two_bays()
    doc = contract.build_map(bays, roads_with_aisle().derive(bays), (H, W))
    for spot in doc["spots"]:
        assert spot["access"]["point"][1] == 260, spot["access"]


def test_a_bay_with_no_road_gets_no_access_point_rather_than_a_wrong_one() -> None:
    bays = two_bays()
    doc = contract.build_map(bays, RoadNetwork(shape=(H, W)).derive(bays), (H, W))
    assert all(s["access"] is None for s in doc["spots"]), doc["spots"]


def test_road_nodes_are_named_r_and_carry_a_type() -> None:
    bays = two_bays()
    doc = contract.build_map(bays, roads_with_aisle().derive(bays), (H, W))
    types = set()
    for node in doc["road_network"]["nodes"]:
        assert node["id"].startswith("r"), node
        assert isinstance(node["x"], (int, float)) and isinstance(node["y"], (int, float))
        types.add(node["type"])
    assert "entrance" in types and "spot_access" in types, types


def test_edges_carry_distance_and_direction_for_the_navigator() -> None:
    bays = two_bays()
    net = roads_with_aisle()
    net.edges[0].oneway = True
    doc = contract.build_map(bays, net.derive(bays), (H, W))
    for edge in doc["road_network"]["edges"]:
        assert edge["distance"] > 0, edge
        assert isinstance(edge["bidirectional"], bool), edge
    assert any(e["bidirectional"] is False for e in doc["road_network"]["edges"]), \
        "a one-way road must be reported as one-way"


def test_the_car_park_entrance_is_named_separately_from_bay_access() -> None:
    """The two would be confused constantly, which is why a bay has 'access' and the car
    park has 'entrances'."""
    bays = two_bays()
    doc = contract.build_map(bays, roads_with_aisle().derive(bays), (H, W))
    assert len(doc["entrances"]) == 1
    gate = doc["entrances"][0]
    assert gate["type"] == "entry_exit"
    assert gate["node"] in {n["id"] for n in doc["road_network"]["nodes"]}
    assert "entrance" not in doc["spots"][0], "a bay must not have an 'entrance' field"


def test_old_style_node_names_are_renamed_on_load() -> None:
    doc = {"image_size": [H, W],
           "nodes": [{"id": "n1", "kind": "entrance", "x": 0, "y": 400},
                     {"id": "n2", "kind": "waypoint", "x": 800, "y": 400}],
           "edges": [{"a": "n1", "b": "n2"}]}
    net = RoadNetwork.from_document(doc, (H, W))
    assert [n.id for n in net.nodes] == ["r1", "r2"], net.nodes
    assert (net.edges[0].a, net.edges[0].b) == ("r1", "r2"), net.edges[0]


# ---------------------------------------------------------------------------
# The live state
# ---------------------------------------------------------------------------

def test_the_state_reports_one_of_three_states_per_bay() -> None:
    doc = contract.build_state({"38": FakeResult(contract.EMPTY, 0.0),
                                "39": FakeResult(contract.OCCUPIED, 0.42)})
    assert doc["schema"] == contract.STATE_SCHEMA
    assert doc["spots"]["38"]["state"] == "EMPTY"
    assert doc["spots"]["39"]["state"] == "OCCUPIED"
    assert doc["spots"]["39"]["changed_ratio"] == 0.42
    assert doc["vehicle"] == {"id": "hero", "visible": False}


def test_unknown_is_a_real_state_with_a_reason() -> None:
    doc = contract.unknown_state(["38", "39"], "no frames for 4.0s")
    for spot in doc["spots"].values():
        assert spot["state"] == "UNKNOWN"
        assert spot["reason"] == "no frames for 4.0s"
    # Nothing was measured, so there must be no measurement in the document.
    assert "changed_ratio" not in doc["spots"]["38"]


def test_every_bay_is_present_in_the_state() -> None:
    ids = ["10", "38", "39", "B1"]
    doc = contract.unknown_state(ids, "camera stopped")
    assert set(doc["spots"]) == set(ids)


def test_the_state_carries_the_vehicle_when_it_is_known() -> None:
    vehicle = {"id": "hero", "x": 423.0, "y": 286.0, "heading": 91.4, "visible": True}
    doc = contract.build_state({"38": FakeResult(contract.EMPTY)}, vehicle=vehicle)
    assert doc["vehicle"] == vehicle
    assert "423" in contract.summarize_state(doc)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def test_writing_is_atomic_and_leaves_no_litter() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out", "parking_map.json")
        contract.write_json(path, {"schema": "x"})
        assert json.load(open(path))["schema"] == "x"
        assert os.listdir(os.path.dirname(path)) == ["parking_map.json"], \
            "a temporary file was left behind"


def test_a_written_document_is_readable_json_with_a_trailing_newline() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "parking_state.json")
        contract.write_json(path, contract.unknown_state(["38"], "test"))
        text = open(path, encoding="utf-8").read()
        assert text.endswith("\n")
        assert json.loads(text)["spots"]["38"]["state"] == "UNKNOWN"


def test_the_summaries_say_something_useful() -> None:
    bays = two_bays()
    doc = contract.build_map(bays, roads_with_aisle().derive(bays), (H, W))
    line = contract.summarize_map(doc)
    assert "2 spot(s)" in line and "2 with an access point" in line, line
    state = contract.build_state({"38": FakeResult(contract.OCCUPIED)})
    assert "1 OCCUPIED" in contract.summarize_state(state)


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
