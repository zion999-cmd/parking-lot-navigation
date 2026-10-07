"""The editor's frame rate, and what made dragging a bay corner feel like treacle.

Measured cause, on the real 48-bay map: deriving the navigation graph costs 20.3 ms, and the
window did it on *every* frame - and, in the roads layer, twice per frame, because the mouse
handler derived it again to find out which entrance was under the pointer. Dragging a corner
therefore ran at 15 fps, and 20 ms of every frame was spent working out bay entrances that
nothing on screen needed at that moment.

Two things changed. The derived graph is now cached against a fingerprint of everything
`derive()` reads, so a frame that changes nothing costs 0.03 ms instead of 20.3 ms; and while a
corner is being dragged, the layer underneath is drawn from the roads as drawn, because the
derived entrances are the only expensive part and they are decoration while the pointer moves.

The guards below are all fast: no window is opened, and the fixtures are a handful of bays.

Run:  python tests/test_editor_speed.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
REAL_CONFIG = os.path.join(HERE, "..", "config", "parking_spots.json")
sys.path.insert(0, SRC)

import road_editor                                            # noqa: E402
import road_network                                            # noqa: E402
from calibration import CalibrationApp                         # noqa: E402
from road_network import RoadNetwork                           # noqa: E402
from spot_editor import SpotEditor                             # noqa: E402

W, H = 900, 600


def app() -> CalibrationApp:
    """An editor with a few bays in a row and one straight road beside them."""
    ed = SpotEditor((H, W), start_id=38)
    for x in (120, 220, 320):
        ed.add_at(x, 300)
    a = CalibrationApp(ed, "/tmp/does-not-matter.json")
    first = a.road.net.add_node(60, 420)
    second = a.road.net.add_node(800, 420)
    a.road.net.connect(first.id, second.id)
    a._frame = np.full((H, W, 3), 120, np.uint8)
    return a


class counted:
    """Count RoadNetwork.derive calls, so "did it recompute?" is answerable directly."""

    def __enter__(self):
        self.original = road_network.RoadNetwork.derive
        self.n = 0

        def wrapper(inner_self, *args, **kwargs):
            self.n += 1
            return self.original(inner_self, *args, **kwargs)

        road_network.RoadNetwork.derive = wrapper
        return self

    def __exit__(self, *exc):
        road_network.RoadNetwork.derive = self.original
        return False


def signature(net: RoadNetwork) -> tuple:
    """What a derived graph *is*, as a comparable value: ids, positions, kinds and edges."""
    nodes = sorted((n.id, round(float(n.x), 6), round(float(n.y), 6), n.kind, n.spot)
                   for n in net.nodes)
    edges = sorted((e.a, e.b, bool(e.oneway)) for e in net.edges)
    return (nodes, edges)


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------

def test_a_repeated_lookup_derives_once() -> None:
    a = app()
    with counted() as c:
        first = a.road.derived(a.ed.spots)
        second = a.road.derived(a.ed.spots)
        third = a.road.derived(a.ed.spots)
    assert c.n == 1, f"derived {c.n} times for an unchanged drawing"
    assert signature(first) == signature(second) == signature(third)


def test_moving_a_bay_recomputes_it() -> None:
    a = app()
    with counted() as c:
        before = signature(a.road.derived(a.ed.spots))
        # The whole bay, not one corner: which side faces the road decides the entrance, so a
        # single corner can move without moving it - which is what the first version of this
        # check asserted, wrongly, and the code was right and the check was wrong.
        a.ed.spots[0].polygon = a.ed.spots[0].polygon + np.array([40.0, -30.0])
        after = signature(a.road.derived(a.ed.spots))
    assert c.n == 2, f"derived {c.n} times across a changed bay"
    assert before != after, "the cache answered with the old graph"


def test_moving_a_road_node_recomputes_it() -> None:
    a = app()
    with counted() as c:
        first = signature(a.road.derived(a.ed.spots))
        a.road.net.nodes[0].y -= 40.0
        second = signature(a.road.derived(a.ed.spots))
    assert c.n == 2, c.n
    assert first != second


def test_changing_the_frame_size_recomputes_it() -> None:
    """The frame size picks the border entrance, so it has to be part of the key.

    Not hypothetical: the first version of the fingerprint left it out, and the graph would have
    been reused for a map of a different size. What is asserted is that the key notices, not that
    this particular fixture's entrance moves - it happens not to.
    """
    a = app()
    with counted() as c:
        a.road.derived(a.ed.spots)
        before_key = road_editor.fingerprint(a.road.net, a.ed.spots)
        a.road.net.shape = (H + 100, W + 100)
        after_key = road_editor.fingerprint(a.road.net, a.ed.spots)
        a.road.derived(a.ed.spots)
    assert c.n == 2, c.n
    assert before_key != after_key, "a different frame size must change the key"


def test_a_cached_graph_is_the_graph_derive_would_have_returned() -> None:
    """The guard that matters: fast is worthless if it is also wrong.

    After a sequence of edits, the cached answer is compared against derive() itself, called
    directly so the cache cannot be involved.
    """
    a = app()
    a.road.derived(a.ed.spots)                                  # warm it
    a.ed.spots[1].polygon[2] = a.ed.spots[1].polygon[2] + np.array([0.0, 25.0])
    a.road.net.nodes[1].x += 30.0
    a.road.net.add_node(500, 100)
    assert signature(a.road.derived(a.ed.spots)) == signature(a.road.net.derive(a.ed.spots))


def test_a_caller_cannot_poison_the_cache() -> None:
    """derive() promises a new object the caller may do as it likes with; the cache keeps that."""
    a = app()
    first = a.road.derived(a.ed.spots)
    clean = signature(first)
    first.nodes.clear()
    first.edges.clear()
    for n in first.nodes:
        n.x = -9999.0
    again = a.road.derived(a.ed.spots)
    assert signature(again) == clean, "a caller's edits leaked into the cached graph"


# ---------------------------------------------------------------------------
# Not deriving while a corner is being dragged
# ---------------------------------------------------------------------------

def test_dragging_is_reported() -> None:
    a = app()
    assert a.ed.dragging() is False
    assert a.ed.begin_drag(a.ed.spots[0].polygon[0][0] + 1, a.ed.spots[0].polygon[0][1] + 1)
    assert a.ed.dragging() is True
    a.ed.update_drag(a.ed.spots[0].polygon[0][0] + 6, a.ed.spots[0].polygon[0][1] + 4)
    assert a.ed.dragging() is True
    a.ed.end_drag()
    assert a.ed.dragging() is False


def test_the_faint_layer_does_not_derive_while_a_corner_is_dragged() -> None:
    """The fix, at the call site the window actually uses."""
    a = app()
    a.layer = "spots"
    canvas = np.zeros((H, W, 3), np.uint8)
    a.ed.begin_drag(a.ed.spots[0].polygon[0][0] + 1, a.ed.spots[0].polygon[0][1] + 1)

    with counted() as c:
        for step in range(8):                       # a drag is many frames
            a.ed.update_drag(a.ed.spots[0].polygon[0][0] + step, a.ed.spots[0].polygon[0][1])
            a.render(a._frame)
    assert c.n == 0, f"{c.n} derives during a drag; a bay's corners are not its entrances"

    a.ed.end_drag()
    with counted() as c2:
        a.render(a._frame)
    assert c2.n == 1, "the entrances should come back the moment the drag ends"


def test_the_faint_layer_still_derives_when_nothing_is_being_dragged() -> None:
    """So the derived entrances do not simply disappear from the spots layer."""
    a = app()
    a.layer = "spots"
    with counted() as c:
        a.render(a._frame)
    assert c.n == 1, c.n


def test_the_roads_layer_keeps_deriving_while_dragging() -> None:
    """In the roads layer the derived entrances are what is being edited, so they stay live."""
    a = app()
    a.layer = "roads"
    a.ed.begin_drag(a.ed.spots[0].polygon[0][0] + 1, a.ed.spots[0].polygon[0][1] + 1)
    with counted() as c:
        a.road._derived_key = None                  # as if the roads had just changed
        a.render(a._frame)
    assert c.n == 1, c.n


def test_the_real_map_is_cheap_on_a_cache_hit() -> None:
    """The number that started this: 48 bays against a handful of roads on every frame."""
    if not os.path.exists(REAL_CONFIG):
        return
    import time
    ed = SpotEditor((600, 800))
    if not ed.load(REAL_CONFIG):
        return
    a = CalibrationApp(ed, "/tmp/does-not-matter.json", template=None)
    a._frame = np.full((600, 800, 3), 120, np.uint8)
    a.road.derived(ed.spots)                                        # warm
    t0 = time.perf_counter()
    for _ in range(30):
        a.road.derived(ed.spots)
    hit_ms = (time.perf_counter() - t0) / 30 * 1000
    print(f"        48-bay map: {hit_ms:.2f} ms per cached derived() (was 20.3 ms)")
    assert hit_ms < 3.0, f"a cache hit costs {hit_ms:.1f} ms; the cache is not working"


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
