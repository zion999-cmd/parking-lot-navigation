"""The parking lot map: bays, roads, and the graph derived from them.

The user-facing model is deliberately small. A person marks bays and draws where cars
drive; the bay-to-road connections are derived from those two things, so the tests that
matter most are the ones about derive() - that it is pure, deterministic, puts each
entrance on the side of the bay facing the road, and never lets one bay's entrance become
another bay's road.

No test here needs to know about nodes, edges or chains except the ones that check the data
model itself, which is the point.

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
    """Two bays side by side, y 100..260, x 100..150 and 160..210, on a 900x600 frame."""
    poly = [[100, 100], [150, 100], [150, 260], [100, 260]]
    other = [[160, 100], [210, 100], [210, 260], [160, 260]]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "spots.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"image_size": [H, W],
                       "spots": [{"id": "38", "polygon": poly},
                                 {"id": "39", "polygon": other}]}, fh)
        return load_spots(path, (H, W))


def bay(polygon, bay_id: str = "38"):
    class _Bay:
        pass
    b = _Bay()
    b.id = bay_id
    b.polygon = np.asarray(polygon, dtype=float)
    return b


def roads_with_aisle() -> RoadNetwork:
    """A road running under both bays: from the left edge of the picture to the right."""
    net = RoadNetwork(shape=(H, W))
    a = net.add_node(0, 400)
    b = net.add_node(880, 400)
    net.connect(a.id, b.id)
    return net


# ---------------------------------------------------------------------------
# The data model
# ---------------------------------------------------------------------------

def test_ids_are_never_reused_after_a_deletion() -> None:
    net = RoadNetwork(shape=(H, W))
    a = net.add_node(10, 10)
    b = net.add_node(20, 20)
    net.remove_node(a.id)
    c = net.add_node(30, 30)
    assert c.id not in (a.id, b.id), "a freed id was handed out again"


def test_a_road_point_cannot_be_placed_outside_the_frame() -> None:
    net = RoadNetwork(shape=(H, W))
    n = net.add_node(5000, -400)
    assert 0 <= n.x <= W - 1 and 0 <= n.y <= H - 1, (n.x, n.y)


def test_connect_refuses_self_loops_and_duplicates() -> None:
    net = RoadNetwork(shape=(H, W))
    a, b = net.add_node(10, 10), net.add_node(100, 10)
    assert net.connect(a.id, a.id) is None
    assert net.connect(a.id, b.id) is not None
    assert net.connect(a.id, b.id) is None, "a duplicate segment would double a path weight"
    assert net.connect(b.id, a.id) is None


def test_removing_a_point_removes_its_segments() -> None:
    net = RoadNetwork(shape=(H, W))
    a, b, c = (net.add_node(10, 10), net.add_node(100, 10), net.add_node(200, 10))
    net.connect(a.id, b.id)
    net.connect(b.id, c.id)
    net.remove_node(b.id)
    assert net.edges == []
    assert {n.id for n in net.nodes} == {a.id, c.id}


def test_oneway_roads_are_asymmetric_for_the_navigator() -> None:
    net = RoadNetwork(shape=(H, W))
    a, b = net.add_node(0, 0), net.add_node(100, 0)
    net.connect(a.id, b.id, oneway=True)
    graph = net.adjacency()
    assert [n for n, _ in graph[a.id]] == [b.id]
    assert graph[b.id] == [], "a one-way road must not be walkable backwards"


def test_segment_weight_is_the_pixel_distance() -> None:
    net = RoadNetwork(shape=(H, W))
    a, b = net.add_node(0, 0), net.add_node(30, 40)
    net.connect(a.id, b.id)
    assert abs(net.adjacency()[a.id][0][1] - 50.0) < 1e-6


def test_validate_reports_dangling_and_unknown_things() -> None:
    net = RoadNetwork(shape=(H, W))
    j = net.add_node(100, 300)
    s = net.add_node(100, 200, kind=KIND_SPOT, spot="99")
    net.connect(j.id, s.id)
    net.edges.append(type(net.edges[0])(a=j.id, b="ghost"))
    problems = " | ".join(net.validate(["38", "39"]))
    assert "ghost" in problems, problems
    assert "99" in problems, problems


# ---------------------------------------------------------------------------
# derive(): bays + roads -> navigation graph
# ---------------------------------------------------------------------------

def test_an_entrance_is_placed_on_the_side_of_the_bay_facing_the_road() -> None:
    graph = roads_with_aisle().derive(two_bays())
    for spot_id in ("38", "39"):
        node = graph.node(graph.spot_nodes()[spot_id])
        assert abs(node.y - 260) < 1, \
            f"bay {spot_id}: entrance at y={node.y}; the road is below, so it belongs at 260"
        assert node.x == {38: 125, 39: 185}[int(spot_id)] or abs(
            node.x - {38: 125, 39: 185}[int(spot_id)]) < 1, (spot_id, node.x)


def test_every_bay_gets_an_entrance_with_no_extra_step() -> None:
    """The product requirement: the user draws roads and stops there."""
    bays = two_bays()
    graph = roads_with_aisle().derive(bays)
    assert set(graph.spot_nodes()) == {"38", "39"}
    assert graph.validate(["38", "39"]) == []


def test_deriving_leaves_the_drawn_roads_untouched() -> None:
    """Purity is what stops the saved map becoming a mix of decisions and machine output."""
    roads = roads_with_aisle()
    nodes, edges = len(roads.nodes), len(roads.edges)
    roads.derive(two_bays())
    assert (len(roads.nodes), len(roads.edges)) == (nodes, edges)
    assert all(n.kind != KIND_SPOT for n in roads.nodes)


def test_deriving_twice_gives_the_same_graph() -> None:
    roads = roads_with_aisle()
    first = roads.derive(two_bays())
    second = roads.derive(two_bays())
    assert len(first.nodes) == len(second.nodes)
    assert [(n.id, n.kind, round(n.x), round(n.y)) for n in first.nodes] == \
           [(n.id, n.kind, round(n.x), round(n.y)) for n in second.nodes]


def test_an_entrance_joins_a_road_in_the_middle_of_a_segment() -> None:
    """A bay on a long road adds a point where it is, rather than being dragged to an end."""
    roads = roads_with_aisle()
    assert len(roads.edges) == 1
    graph = roads.derive(two_bays())
    assert len(graph.edges) >= 4, [(e.a, e.b) for e in graph.edges]
    for spot_id in ("38", "39"):
        entrance = graph.node(graph.spot_nodes()[spot_id])
        partner = graph.node(graph.adjacency()[entrance.id][0][0])
        assert abs(partner.y - 400) < 1, (spot_id, partner.y)
        assert 0 < partner.x < 880, "the join must be on the segment, not at one end"


def test_an_entrance_never_hangs_off_another_bays_entrance() -> None:
    graph = roads_with_aisle().derive(two_bays())
    for node_id, neighbours in graph.adjacency().items():
        node = graph.node(node_id)
        if node.kind != KIND_SPOT:
            continue
        assert len(neighbours) == 1, (node_id, neighbours)
        partner = graph.node(neighbours[0][0])
        assert partner.kind != KIND_SPOT, f"{node_id} hangs off {partner.id}"


def test_the_entrance_is_the_road_point_closest_to_the_edge_of_the_picture() -> None:
    roads = roads_with_aisle()
    graph = roads.derive(two_bays())
    marked = [n for n in graph.nodes if n.kind == KIND_ENTRANCE]
    assert len(marked) == 1, marked
    assert marked[0].x in (0, 880), marked[0]


def test_a_marked_entrance_is_not_overwritten() -> None:
    roads = roads_with_aisle()
    middle = roads.nodes[1]
    middle.kind = KIND_ENTRANCE
    graph = roads.derive(two_bays())
    marked = [n for n in graph.nodes if n.kind == KIND_ENTRANCE]
    assert len(marked) == 1 and marked[0].id == middle.id, marked


def test_every_bay_is_reachable_from_the_entrance() -> None:
    graph = roads_with_aisle().derive(two_bays())
    adjacency = graph.adjacency()
    start = next(n.id for n in graph.nodes if n.kind == KIND_ENTRANCE)
    seen, stack = {start}, [start]
    while stack:
        for nxt, _ in adjacency[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    for bay_id, node_id in graph.spot_nodes().items():
        assert node_id in seen, f"bay {bay_id} is not reachable"


def test_bays_far_from_any_road_are_reported() -> None:
    """The one thing a person can act on, said in map terms."""
    roads = RoadNetwork(shape=(H, W))
    a = roads.add_node(0, 590)
    b = roads.add_node(880, 590)
    roads.connect(a.id, b.id)
    far = roads.far_bays(two_bays(), limit=100.0)
    assert {sid for sid, _ in far} == {"38", "39"}, far
    assert all(d > 100 for _, d in far)


def test_no_bays_far_from_the_road_when_it_is_close() -> None:
    assert roads_with_aisle().far_bays(two_bays(), limit=220.0) == []


# ---------------------------------------------------------------------------
# The saved map: bays and roads in one file, each writer preserving the other
# ---------------------------------------------------------------------------

def test_saving_the_roads_keeps_the_bays() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "map.json")
        save_spots(path, two_bays(), (H, W))

        roads = RoadNetwork.load(path, (H, W))
        a = roads.add_node(0, 400)
        b = roads.add_node(800, 400)
        roads.connect(a.id, b.id)
        roads.save(path)

        doc = read(path)
        assert len(doc["spots"]) == 2, "saving the roads destroyed the bays"
        assert len(doc["nodes"]) == 2 and len(doc["edges"]) == 1
        assert len(load_spots(path, (H, W))) == 2


def test_saving_the_bays_keeps_the_roads() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "map.json")
        roads = RoadNetwork(shape=(H, W))
        a, b = roads.add_node(0, 400), roads.add_node(800, 400)
        roads.connect(a.id, b.id)
        roads.save(path)

        save_spots(path, two_bays(), (H, W))

        doc = read(path)
        assert len(doc["nodes"]) == 2 and len(doc["edges"]) == 1, \
            "saving the bays destroyed the roads"
        assert len(doc["spots"]) == 2


def test_the_file_holds_bays_and_roads_and_nothing_derived() -> None:
    """The map is what a human decided; the graph is worked out from it."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "map.json")
        save_spots(path, two_bays(), (H, W))
        roads = RoadNetwork.load(path, (H, W))
        a, b = roads.add_node(0, 400), roads.add_node(800, 400)
        roads.connect(a.id, b.id)
        roads.save(path)

        doc = read(path)
        assert "nodes" in doc and "edges" in doc
        kinds = {n.get("kind") for n in doc["nodes"]}
        assert KIND_SPOT not in kinds, "a derived bay entrance was written to the map"


def test_coordinates_stay_whole_numbers() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "map.json")
        roads = RoadNetwork(shape=(H, W))
        roads.add_node(64, 194)
        roads.save(path)
        text = open(path, encoding="utf-8").read()
        assert '"x": 64' in text and '"x": 64.0' not in text, text


def test_a_bay_only_file_does_not_invent_road_sections() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "map.json")
        save_spots(path, two_bays(), (H, W))
        doc = read(path)
        assert "nodes" not in doc and "edges" not in doc, doc.keys()


def test_the_map_file_is_readable_one_object_per_line() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "map.json")
        save_spots(path, two_bays(), (H, W))
        lines = [ln for ln in open(path, encoding="utf-8").read().splitlines()
                 if ln.strip().startswith("{") and not ln.startswith("{")]
        assert len(lines) == 2, "each bay should be one line"


# ---------------------------------------------------------------------------
# Drawing roads
# ---------------------------------------------------------------------------

def test_clicking_along_a_road_chaindraws_it() -> None:
    ed = RoadEditor((H, W))
    ed.add_at(100, 300)
    ed.add_at(200, 300)
    ed.add_at(300, 300)
    assert [(e.a, e.b) for e in ed.net.edges] == [("n1", "n2"), ("n2", "n3")]


def test_starting_another_road_does_not_join_the_previous_one() -> None:
    ed = RoadEditor((H, W))
    ed.add_at(100, 300)
    ed.add_at(200, 300)
    ed.break_chain()
    ed.add_at(100, 500)
    assert len(ed.net.nodes) == 3 and len(ed.net.edges) == 1


def test_selecting_a_point_makes_it_where_the_next_click_continues() -> None:
    ed = RoadEditor((H, W))
    ed.add_at(100, 300)
    ed.add_at(200, 300)
    ed.select(0)
    ed.add_at(100, 400)
    assert ("n1", "n3") in {(e.a, e.b) for e in ed.net.edges}


def test_joining_two_roads() -> None:
    ed = RoadEditor((H, W))
    ed.add_at(100, 300)
    ed.add_at(200, 300)
    ed.break_chain()
    ed.add_at(100, 500)
    assert ed.connect_to(0) is True
    assert len(ed.net.edges) == 2


def test_deleting_a_point_takes_its_segments() -> None:
    ed = RoadEditor((H, W))
    ed.add_at(100, 300)
    ed.add_at(200, 300)
    ed.select(1)
    assert ed.delete_selected() is True
    assert len(ed.net.nodes) == 1 and ed.net.edges == []


def test_undo_restores_what_was_deleted() -> None:
    ed = RoadEditor((H, W))
    ed.add_at(100, 300)
    ed.add_at(200, 300)
    ed.delete_selected()
    ed.undo()
    assert len(ed.net.nodes) == 2 and len(ed.net.edges) == 1


def test_a_drag_cannot_push_a_point_out_of_the_frame() -> None:
    ed = RoadEditor((H, W))
    ed.add_at(20, 20)
    assert ed.begin_drag(20, 20) is True
    ed.update_drag(9000, 9000)
    ed.end_drag()
    node = ed.selected_node()
    assert node.x <= W - 1 and node.y <= H - 1


def test_marking_the_way_in_and_unmarking_it() -> None:
    ed = RoadEditor((H, W))
    ed.add_at(40, 400)
    ed.add_at(500, 400)
    ed.select(1)
    ed.toggle_entrance()
    assert ed.selected_node().kind == KIND_ENTRANCE
    ed.select(0)
    ed.toggle_entrance()
    assert ed.selected_node().kind == KIND_ENTRANCE, "only one way in at a time"
    assert sum(1 for n in ed.net.nodes if n.kind == KIND_ENTRANCE) == 1


def test_the_editor_only_ever_holds_roads() -> None:
    """Bay entrances are derived, so they must never end up in the editable set."""
    ed = RoadEditor((H, W))
    ed.add_at(0, 400)
    ed.add_at(880, 400)
    assert all(n.kind != KIND_SPOT for n in ed.net.nodes)
    assert len(ed.derived(two_bays()).spot_nodes()) == 2


def test_a_check_reports_a_missing_road_and_a_complete_map() -> None:
    ed = RoadEditor((H, W))
    assert "no roads drawn yet" in ed.problems(two_bays())[0]
    ed.add_at(0, 400)
    ed.add_at(880, 400)
    assert ed.problems(two_bays()) == []


# ---------------------------------------------------------------------------
# The two layers in one window
# ---------------------------------------------------------------------------

def test_the_layer_switches_with_t_and_with_the_tabs() -> None:
    ed = SpotEditor((H, W))
    app = CalibrationApp(ed, "unused.json")
    assert app.layer == "spots"
    app.act("t")
    assert app.layer == "roads"
    app.act("@spots")
    assert app.layer == "spots"
    app.act("@roads")
    assert app.layer == "roads"


def test_render_works_in_both_layers_and_shows_both_tabs() -> None:
    ed = SpotEditor((H, W))
    app = CalibrationApp(ed, "unused.json")
    img = frame()
    for layer in ("spots", "roads", "spots"):
        app.layer = layer
        assert app.render(img).shape == img.shape
        labels = [b[4] for b in app._buttons]
        assert "Parking Spots" in labels and "Roads" in labels, labels


def test_clicks_only_draw_roads_in_the_road_layer() -> None:
    ed = SpotEditor((H, W))
    app = CalibrationApp(ed, "unused.json")
    app.render(frame())

    app.layer = "spots"
    app.on_mouse(cv2.EVENT_LBUTTONDOWN, 300, 200, 0, None)
    app.on_mouse(cv2.EVENT_LBUTTONUP, 300, 200, 0, None)
    assert app.road.net.nodes == [], "a click in the bay layer drew a road"

    app.layer = "roads"
    for x in (200, 300, 400):
        app.on_mouse(cv2.EVENT_LBUTTONDOWN, x, 200, 0, None)
        app.on_mouse(cv2.EVENT_LBUTTONUP, x, 200, 0, None)
    assert len(app.road.net.nodes) == 3
    assert len(app.road.net.edges) == 2


def test_drawing_a_road_is_all_it_takes_to_connect_the_bays() -> None:
    """The whole product requirement in one test: mark bays, draw roads, done."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "map.json")
        save_spots(path, two_bays(), (H, W))

        ed = SpotEditor((H, W))
        ed.load(path)
        app = CalibrationApp(ed, path)
        app.layer = "roads"
        for x in (0, 880):                      # one road under the two bays
            app.on_mouse(cv2.EVENT_LBUTTONDOWN, x, 400, 0, None)
            app.on_mouse(cv2.EVENT_LBUTTONUP, x, 400, 0, None)

        # No binding step: the entrances and the graph are already there.
        graph = app.road.derived(ed.spots)
        assert set(graph.spot_nodes()) == {"38", "39"}
        assert app.road.problems(ed.spots) == []
        assert "2/2" in app.road.stats(ed.spots), app.road.stats(ed.spots)


def test_saving_from_either_layer_writes_one_document_with_both() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "map.json")
        save_spots(path, two_bays(), (H, W))

        ed = SpotEditor((H, W))
        ed.load(path)
        app = CalibrationApp(ed, path)
        app.layer = "roads"
        app.on_mouse(cv2.EVENT_LBUTTONDOWN, 300, 400, 0, None)
        app.act("s")

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


def test_stored_bay_connection_points_are_restored_to_plain_road_points() -> None:
    """An earlier version could turn one of the user's own road points into a bay
    connection point, which is how two corners of a drawn ring ended up labelled as bays.
    Turning them back keeps the geometry; deleting them would cut the ring in half."""
    doc = {
        "image_size": [H, W],
        "nodes": [{"id": "n1", "kind": "entrance", "x": 0, "y": 400},
                  {"id": "n2", "kind": "spot", "x": 880, "y": 400, "spot": "10"}],
        "edges": [{"a": "n1", "b": "n2"}],
    }
    net = RoadNetwork.from_document(doc, (H, W))
    assert net.stale_connections == 1
    assert net.node("n2").kind == KIND_WAYPOINT
    assert net.node("n2").spot == ""
    assert len(net.edges) == 1, "the road itself must survive"
    assert net.spot_nodes() == {}


def test_a_saved_map_never_contains_a_derived_bay_entrance() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "map.json")
        save_spots(path, two_bays(), (H, W))
        roads = RoadNetwork.load(path, (H, W))
        a, b = roads.add_node(0, 400), roads.add_node(880, 400)
        roads.connect(a.id, b.id)
        roads.save(path)
        kinds = {n.get("kind") for n in read(path)["nodes"]}
        assert KIND_SPOT not in kinds, kinds


def test_the_editor_opens_with_the_roads_already_in_the_file() -> None:
    """Regression: the road layer used to start empty and would then overwrite them."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "map.json")
        save_spots(path, two_bays(), (H, W))
        roads = RoadNetwork.load(path, (H, W))
        a, b = roads.add_node(0, 400), roads.add_node(880, 400)
        roads.connect(a.id, b.id)
        roads.save(path)

        ed = SpotEditor((H, W))
        ed.load(path)
        app = CalibrationApp(ed, path)
        assert len(app.road.net.nodes) == 2, app.road.net.nodes
        assert len(app.road.net.edges) == 1
        assert "2/2" in app.road.stats(ed.spots), app.road.stats(ed.spots)


def test_opening_the_editor_on_a_map_with_stale_points_marks_it_for_saving() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "map.json")
        save_spots(path, two_bays(), (H, W))
        roads = RoadNetwork.load(path, (H, W))
        a = roads.add_node(0, 400, kind=KIND_ENTRANCE)
        b = roads.add_node(880, 400, kind=KIND_SPOT, spot="38")
        roads.connect(a.id, b.id)
        roads.save(path)

        ed = SpotEditor((H, W))
        ed.load(path)
        app = CalibrationApp(ed, path)
        assert app.road.net.stale_connections == 1
        assert app.road.dirty, "a repaired map should be written back out"
        app.act("q")
        assert KIND_SPOT not in {n.get("kind") for n in read(path)["nodes"]}


def test_switching_layers_with_both_a_map_and_roads_loaded() -> None:
    """Regression: switch_layer() used to hand the bay *ids* where the map code wants the
    bays themselves. The existing tab test did not catch it because it switched layers with
    no bays loaded, where the wrong argument happens to be harmless."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "map.json")
        save_spots(path, two_bays(), (H, W))
        roads = RoadNetwork.load(path, (H, W))
        a = roads.add_node(0, 400)
        b = roads.add_node(880, 400)
        roads.connect(a.id, b.id)
        roads.save(path)

        ed = SpotEditor((H, W))
        ed.load(path)
        app = CalibrationApp(ed, path)
        app.act("t")
        assert app.layer == "roads"
        app.render(frame())
        app.act("@spots")
        assert app.layer == "spots"


def test_a_bays_entrance_can_be_pinned_to_a_side() -> None:
    """At the end of a row two roads are equally close, so the side a car comes in from is
    a judgement. Pinning it is how that judgement is recorded - and the connection point and
    the graph then follow from it with nothing else to do."""
    bays = two_bays()
    roads = roads_with_aisle()

    auto = roads.derive(bays)
    auto_mouth = auto.node(auto.spot_nodes()["38"])
    assert (round(auto_mouth.x), round(auto_mouth.y)) == (125, 260), \
        "with the road below, the automatic entrance is the bottom side"

    # side 1 of this polygon is its right-hand edge (midpoint 150,180)
    bays[0].entrance_edge = 1
    pinned = roads.derive(bays)
    pinned_mouth = pinned.node(pinned.spot_nodes()["38"])
    assert (round(pinned_mouth.x), round(pinned_mouth.y)) == (150, 180), \
        (pinned_mouth.x, pinned_mouth.y)


def test_pinning_one_bay_leaves_the_others_alone() -> None:
    bays = two_bays()
    roads = roads_with_aisle()
    bays[0].entrance_edge = 1
    graph = roads.derive(bays)
    other = graph.node(graph.spot_nodes()["39"])
    assert (round(other.x), round(other.y)) == (185, 260), (other.x, other.y)


def test_dropping_a_bays_entrance_records_the_side_it_was_dropped_on() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "map.json")
        save_spots(path, two_bays(), (H, W))
        roads = RoadNetwork.load(path, (H, W))
        a, b = roads.add_node(0, 400), roads.add_node(880, 400)
        roads.connect(a.id, b.id)
        roads.save(path)

        ed = SpotEditor((H, W))
        ed.load(path)
        app = CalibrationApp(ed, path)
        app.layer = "roads"
        app.render(frame())

        assert app._entrance_under(125, 260) == "38", "the derived entrance should be grabbable"
        app._entrance_drag = "38"
        app._drop_entrance("38", 170, 180)          # dropped beside the right-hand edge
        pinned = [s for s in ed.spots if s.id == "38"][0]
        assert pinned.entrance_edge == 1, pinned.entrance_edge
        app._entrance_drag = None

        app.act("s")
        saved = {s["id"]: s.get("entrance") for s in read(path)["spots"]}
        assert saved["38"] == 1, saved

        again = load_spots(path, (H, W))
        assert [s.entrance_edge for s in again if s.id == "38"] == [1]


def test_an_entrance_can_be_sent_back_to_automatic() -> None:
    bays = two_bays()
    roads = roads_with_aisle()
    bays[0].entrance_edge = 1
    assert roads.derive(bays).node(
        roads.derive(bays).spot_nodes()["38"]).x == 150

    ed = SpotEditor((H, W))
    ed.spots = bays
    ed.select(0)
    assert ed.set_entrance(0, None) is True
    assert bays[0].entrance_edge is None
    back = roads.derive(bays).node(roads.derive(bays).spot_nodes()["38"])
    assert (round(back.x), round(back.y)) == (125, 260), (back.x, back.y)
    assert ed.undo() is True
    # undo() swaps in deep copies, so the live list is ed.spots, not the one passed in.
    assert ed.spots[0].entrance_edge == 1, "undo should bring the pinned side back"


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
