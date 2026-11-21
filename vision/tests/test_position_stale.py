"""P0013.1: position_stale is "the last trusted position", not a position.

The navigation lifecycle refused it from the start; the route planner, the page's replanning and the
page's arrival check all did not. A car visible with a stale fix could therefore be routed from, moved on
the map and even complete a navigation - from where it used to be.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import route                                                     # noqa: E402


def state(visible=True, stale=False, bay_state="EMPTY"):
    return {
        "schema": "parking_state/1",
        "timestamp": 1,
        "spots": {"B3": {"state": bay_state}},
        "vehicle": {"id": "hero", "x": 300.0, "y": 500.0, "visible": visible,
                    "position_stale": stale, "location": {"kind": "road", "id": None}},
    }


def map_doc():
    """A minimal map: one bay with an access node, reachable from the hero."""
    return {
        "schema": "parking_map/1",
        "image_size": [600, 800],
        "spots": [{"id": "B3", "polygon": [[280, 480], [320, 480], [320, 520], [280, 520]],
                   "access": {"road_node": "b"}}],
        "road_network": {"nodes": [{"id": "a", "x": 300, "y": 500},
                                   {"id": "b", "x": 300, "y": 540}],
                         "edges": [{"from": "a", "to": "b"}]},
    }


def test_a_stale_position_is_refused():
    planned = route.plan(map_doc(), state(stale=True), "B3")
    assert planned.ok is False
    assert planned.code == route.STALE_HERO, planned.code
    assert "stale" in planned.reason


def test_a_fresh_position_is_not_refused_for_staleness():
    planned = route.plan(map_doc(), state(stale=False), "B3")
    # It may still be refused for another reason (no access drawn in this fixture), but never for
    # staleness - that is the only thing under test here.
    assert planned.code != route.STALE_HERO


def test_invisible_still_wins_the_message():
    """visible=false is refused as hero-not-visible, so the two reasons stay distinguishable."""
    planned = route.plan(map_doc(), state(visible=False, stale=True), "B3")
    assert planned.code == route.NO_HERO
