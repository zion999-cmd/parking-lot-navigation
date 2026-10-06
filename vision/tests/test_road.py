"""Road network layer: the map document, the graph, and the two-layer editor.

The road layer is the navigator's half of the map - "how does a car get there" - as
opposed to the camera's half, "is this bay free". It is drawn by hand on purpose, so what
is worth testing is the data model and the invariants around it, not any detection.

Run:  python tests/test_road.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from calibration import CalibrationApp                          # noqa: E402
from detector import DetectorParams                             # noqa: E402
from map_document import read                                   # noqa: E402
from road_editor import RoadEditor                              # noqa: E402
from road_network import (KIND_ENTRANCE, KIND_JUNCTION,         # noqa: E402
                          KIND_SPOT, KIND_WAYPOINT, RoadNetwork)
from spot_config import load_spots, save_spots                  # noqa: E402
from spot_editor import SpotEditor                              # noqa: E402

W, H = 900, 600


def frame() -> np.ndarray:
    img = np.full((H, W, 3), 120, np.uint8)
    cv2.line(img, (0, H // 2), (W, H // 2), (240, 240, 240), 2)
    return img


def two_bays() -> list:
    """Two bays on a 900x600 frame, without going through a file."""
    poly = [[100, 100], [150, 100], [150, 260], [100, 260]]
    other = [[160, 100], [210, 100], [210, 260], [160, 260]]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "spots.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"image_size": [H, W],
                       "spots": [{"id": "38", "polygon": poly},
                                 {"id": "39", "polygon": other}]}, fh)
        return load_spots(path, (H, W))


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------

def test_ids_are_never_reused_after_a_deletion() -> None:
    net = RoadNetwork(shape=(H, W))
    a = net.add_node(10, 10)
    b = net.add_node(20, 20)
    net.remove_node(a.id)
    c = net.add_node(30, 30)
    assert c.id not in (a.id, b.id), "a freed id was handed out again"
    assert len({n.id for n in net.nodes}) == len(net.nodes)


def test_a_node_cannot_be_placed_outside_the_frame() -> None:
    net = RoadNetwork(shape=(H, W))
    n = net.add_node(5000, -400)
    assert 0 <= n.x <= W - 1 and 0 <= n.y <= H - 1, (n.x, n.y)


def test_connect_refuses_self_loops_and_duplicates() -> None:
    net = RoadNetwork(shape=(H, W))
    a, b = net.add_node(10, 10), net.add_node(100, 10)
    assert net.connect(a.id, a.id) is None
    assert net.connect(a.id, b.id) is not None
    assert net.connect(a.id, b.id) is None, "a duplicate edge would double a path weight"
    assert net.connect(b.id, a.id) is None, "and so would the same edge reversed"


def test_removing_a_node_removes_its_edges() -> None:
    net = RoadNetwork(shape=(H, W))
    a, b, c = (net.add_node(10, 10), net.add_node(100, 10), net.add_node(200, 10))
    net.connect(a.id, b.id)
    net.connect(b.id, c.id)
    net.remove_node(b.id)
    assert net.edges == [], net.edges
    assert {n.id for n in net.nodes} == {a.id, c.id}


def test_oneway_edges_are_asymmetric_for_the_navigator() -> None:
    net = RoadNetwork(shape=(H, W))
    a, b = net.add_node(0, 0), net.add_node(100, 0)
    net.connect(a.id, b.id, oneway=True)
    graph = net.adjacency()
    assert [n for n, _ in graph[a.id]] == [b.id]
    assert graph[b.id] == [], "a one-way road must not be walkable backwards"
    assert net.neighbors(b.id) == []


def test_edge_weight_is_the_pixel_distance() -> None:
    net = RoadNetwork(shape=(H, W))
    a, b = net.add_node(0, 0), net.add_node(30, 40)
    net.connect(a.id, b.id)
    assert abs(net.distance(a.id, b.id) - 50.0) < 1e-6
    assert abs(net.adjacency()[a.id][0][1] - 50.0) < 1e-6


def test_spot_nodes_map_bays_to_their_connection_point() -> None:
    net = RoadNetwork(shape=(H, W))
    j = net.add_node(100, 300, kind=KIND_JUNCTION)
    s1 = net.add_node(100, 200, kind=KIND_SPOT, spot="38")
    s2 = net.add_node(100, 400, kind=KIND_SPOT, spot="39")
    net.connect(j.id, s1.id)
    net.connect(j.id, s2.id)
    assert net.spot_nodes() == {"38": s1.id, "39": s2.id}


def test_validate_reports_the_things_that_break_navigation() -> None:
    net = RoadNetwork(shape=(H, W))
    j = net.add_node(100, 300)
    s = net.add_node(100, 200, kind=KIND_SPOT, spot="99")    # bay 99 does not exist
    net.connect(j.id, s.id)
    net.edges.append(type(net.edges[0])(a=j.id, b="ghost"))  # dangling endpoint
    problems = net.validate(["38", "39"])
    joined = " | ".join(problems)
    assert "ghost" in joined, problems
    assert "99" in joined, problems
    assert "no road connection" in joined, problems


def test_validate_is_quiet_about_a_fully_connected_map() -> None:
    net = RoadNetwork(shape=(H, W))
    e = net.add_node(10, 300, kind=KIND_ENTRANCE)
    j = net.add_node(100, 300, kind=KIND_JUNCTION)
    s = net.add_node(100, 200, kind=KIND_SPOT, spot="38")
    net.connect(e.id, j.id)
    net.connect(j.id, s.id)
    assert net.validate(["38"]) == []


# ---------------------------------------------------------------------------
# One document, two layers
#
# The invariani that matters: the two layers are written by different editors into one
# file, so each writer must re-emit whatever it found. These are the tests that would have
# caught the bays vanishing because someone saved a road.
# ---------------------------------------------------------------------------

def test_saving_the_roads_keeps_the_bays() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "map.json")
        bays = two_bays()
        save_spots(path, bays, (H, W))
        assert len(read(path)["spots"]) == 2

        net = RoadNetwork.load(path, (H, W))
        e = net.add_node(10, 300, kind=KIND_ENTRANCE)
        j = net.add_node(200, 300)
        net.connect(e.id, j.id)
        net.save(path)

        doc = read(path)
        assert len(doc["spots"]) == 2, "saving the road layer destroyed the bays"
        assert len(doc["nodes"]) == 2 and len(doc["edges"]) == 1
        # The bay layer must still load: same document, same reader.
        assert len(load_spots(path, (H, W))) == 2


def test_saving_the_bays_keeps_the_roads() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "map.json")
        net = RoadNetwork(shape=(H, W))
        a, b = net.add_node(10, 10, kind=KIND_ENTRANCE), net.add_node(90, 10)
        net.connect(a.id, b.id)
        net.save(path)

        save_spots(path, two_bays(), (H, W))

        doc = read(path)
        assert len(doc["nodes"]) == 2 and len(doc["edges"]) == 1, \
            "saving the bay layer destroyed the roads"
        assert len(doc["spots"]) == 2
        reloaded = RoadNetwork.load(path, (H, W))
        assert [e.a for e in reloaded.edges] == [a.id]


def test_coordinates_stay_whole_numbers() -> None:
    """Whole pixels must not be written as 64.0 - it turns a small diff into 48 lines."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "map.json")
        net = RoadNetwork(shape=(H, W))
        net.add_node(64, 194, kind=KIND_ENTRANCE)
        net.save(path)
        text = open(path, encoding="utf-8").read()
        assert '"x": 64' in text and '"x": 64.0' not in text, text
        assert '"y": 194' in text, text


def test_the_map_file_is_readable_one_object_per_line() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "map.json")
        save_spots(path, two_bays(), (H, W))
        lines = [ln for ln in open(path, encoding="utf-8").read().splitlines()
                 if ln.strip().startswith("{") and not ln.startswith("{")]
        assert len(lines) == 2, "each bay should be one line"


# ---------------------------------------------------------------------------
# The editor
# ---------------------------------------------------------------------------

def test_clicking_along_a_path_draws_a_chained_road() -> None:
    ed = RoadEditor((H, W))
    ed.add_at(100, 300)
    ed.add_at(200, 300)
    ed.add_at(300, 300)
    assert len(ed.net.nodes) == 3
    assert [(e.a, e.b) for e in ed.net.edges] == [("n1", "n2"), ("n2", "n3")]
    assert ed.net.nodes[0].kind == KIND_ENTRANCE, "a road starts at an entrance"
    assert ed.net.nodes[1].kind == KIND_WAYPOINT, "and continues as plain path"


def test_breaking_the_chain_starts_a_new_road() -> None:
    ed = RoadEditor((H, W))
    ed.add_at(100, 300)
    ed.add_at(200, 300)
    ed.break_chain()
    ed.add_at(100, 500)
    assert len(ed.net.nodes) == 3
    assert len(ed.net.edges) == 1, "the new road must not join the old one"


def test_selecting_a_node_makes_it_the_branch_point() -> None:
    ed = RoadEditor((H, W))
    ed.add_at(100, 300)
    ed.add_at(200, 300)
    ed.select(0)                      # go back to the first node
    ed.add_at(100, 400)               # and extend from it
    pairs = {(e.a, e.b) for e in ed.net.edges}
    assert ("n1", "n3") in pairs, pairs


def test_joining_two_roads_with_a_right_click() -> None:
    ed = RoadEditor((H, W))
    ed.add_at(100, 300)
    ed.add_at(200, 300)
    ed.break_chain()
    ed.add_at(100, 500)
    assert ed.connect_to(0) is True          # join the chain head to node 0
    assert len(ed.net.edges) == 2, ed.net.edges


def test_deleting_a_node_takes_its_edges_and_the_chain_with_it() -> None:
    ed = RoadEditor((H, W))
    ed.add_at(100, 300)
    ed.add_at(200, 300)
    ed.select(1)
    assert ed.delete_selected() is True
    assert len(ed.net.nodes) == 1 and ed.net.edges == []
    assert ed.chain is None, "the chain pointed at the node that no longer exists"


def test_undo_restores_nodes_edges_and_the_chain() -> None:
    ed = RoadEditor((H, W))
    ed.add_at(100, 300)
    ed.add_at(200, 300)
    before_chain = ed.chain
    ed.delete_selected()
    ed.undo()
    assert len(ed.net.nodes) == 2 and len(ed.net.edges) == 1
    assert ed.chain == before_chain


def test_binding_bays_never_hands_out_a_bay_twice() -> None:
    """The invariant: a bay has at most one connection point, and walking the list is how
    the 48 of them get done without typing ids."""
    ed = RoadEditor((H, W))
    ed.add_at(100, 200)
    ed.add_at(200, 200)
    ed.add_at(300, 200)
    bays = ["38", "39", "40"]
    for i, expect in enumerate(bays):
        ed.select(i)
        assert ed.bind_next_bay(bays) == expect
    assert ed.net.spot_nodes() == {"38": "n1", "39": "n2", "40": "n3"}
    ed.select(0)
    assert ed.bind_next_bay(bays) is None, "everything is taken; nothing may be stolen"
    assert ed.net.spot_nodes()["40"] == "n3", "an existing binding was moved"


def test_binding_again_on_the_same_node_moves_it_rather_than_duplicating() -> None:
    """Deliberate, and the only way to correct a wrong binding from the UI.

    add_at() selects the node it just placed, so pressing 'b' twice without moving on
    rebinds that node to the next free bay - and frees the bay it had. The message says
    which bay it came from so this cannot look like a bay silently vanishing, and 'z'
    undoes it like any other edit.
    """
    ed = RoadEditor((H, W))
    ed.add_at(100, 200)
    assert ed.bind_next_bay(["38", "39"]) == "38"
    assert ed.bind_next_bay(["38", "39"]) == "39"
    assert ed.net.spot_nodes() == {"39": "n1"}, ed.net.spot_nodes()
    assert "rebound" in ed.message.text, ed.message.text
    ed.undo()
    assert ed.net.spot_nodes() == {"38": "n1"}, ed.net.spot_nodes()


def test_nudging_a_node_keeps_it_inside_the_frame() -> None:
    ed = RoadEditor((H, W))
    ed.add_at(20, 20)
    ed.select(0)
    for _ in range(30):
        ed.nudge(-5, -5)
    node = ed.selected_node()
    assert node.x >= 0 and node.y >= 0, (node.x, node.y)


def test_a_drag_cannot_push_a_node_out_of_the_frame() -> None:
    ed = RoadEditor((H, W))
    ed.add_at(20, 20)
    assert ed.begin_drag(20, 20) is True
    ed.update_drag(9000, 9000)
    ed.end_drag()
    node = ed.selected_node()
    assert node.x <= W - 1 and node.y <= H - 1, (node.x, node.y)


# ---------------------------------------------------------------------------
# The two layers in one window
# ---------------------------------------------------------------------------

def test_the_layer_switches_with_t_and_with_the_tabs() -> None:
    ed = SpotEditor((H, W))
    app = CalibrationApp(ed, "unused.json")
    assert app.layer == "spots"
    app.act("t")
    assert app.layer == "roads"
    app.act("@spots")                 # the tab button
    assert app.layer == "spots"
    app.act("@roads")
    assert app.layer == "roads"


def test_render_works_in_both_layers_and_shows_both_tabs() -> None:
    ed = SpotEditor((H, W))
    app = CalibrationApp(ed, "unused.json")
    img = frame()
    for layer in ("spots", "roads", "spots"):
        app.layer = layer
        out = app.render(img)
        assert out.shape == img.shape
        labels = [b[4] for b in app._buttons]
        assert "Parking Spots" in labels and "Road Network" in labels, labels


def test_clicks_only_place_road_nodes_in_the_road_layer() -> None:
    ed = SpotEditor((H, W))
    app = CalibrationApp(ed, "unused.json")
    app.render(frame())

    app.layer = "spots"
    app.on_mouse(cv2.EVENT_LBUTTONDOWN, 300, 200, 0, None)
    app.on_mouse(cv2.EVENT_LBUTTONUP, 300, 200, 0, None)
    assert app.road.net.nodes == [], "a click in the bay layer drew a road node"

    app.layer = "roads"
    for x in (200, 300, 400):
        app.on_mouse(cv2.EVENT_LBUTTONDOWN, x, 200, 0, None)
        app.on_mouse(cv2.EVENT_LBUTTONUP, x, 200, 0, None)
    assert len(app.road.net.nodes) == 3, app.road.net.nodes
    assert len(app.road.net.edges) == 2


def test_saving_from_either_layer_writes_one_document_with_both() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "map.json")
        save_spots(path, two_bays(), (H, W))

        ed = SpotEditor((H, W))
        ed.load(path)
        app = CalibrationApp(ed, path)
        app.layer = "roads"
        app.on_mouse(cv2.EVENT_LBUTTONDOWN, 300, 400, 0, None)
        app.act("s")                 # save the road layer only

        doc = read(path)
        assert len(doc["spots"]) == 2, "the bays did not survive the road save"
        assert len(doc["nodes"]) == 1


def test_quitting_saves_both_layers() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "map.json")
        save_spots(path, two_bays(), (H, W))

        ed = SpotEditor((H, W))
        ed.load(path)
        app = CalibrationApp(ed, path)
        app.layer = "roads"
        app.on_mouse(cv2.EVENT_LBUTTONDOWN, 300, 400, 0, None)
        assert app.act("q") is False

        doc = read(path)
        assert len(doc["nodes"]) == 1 and len(doc["spots"]) == 2, doc.keys()


def test_a_bay_layer_save_does_not_invent_road_sections() -> None:
    """A map with no roads must stay a two-key file, not gain empty sections."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "map.json")
        save_spots(path, two_bays(), (H, W))
        doc = read(path)
        assert "nodes" not in doc and "edges" not in doc, doc.keys()


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
