"""Plan a route to a named bay. The navigation layer, and nothing else.

Input is strictly the two documents P0002 emits:

    parking_map.json     the static world - the road graph and each bay's access point
    parking_state.json   the live world  - which bays are free, where the hero car is

That is the whole point of the contract, and it is enforced by construction: nothing here
imports the camera, OpenCV, the bay polygons as drawn, or the road network's internals. The
tests build both documents by hand and plan from them, which would be impossible if this module
reached back into P0002.

The phase 1 job, in the user's words: name an empty bay and get a route from wherever the hero
car is on the road network to that bay's access point. Deciding *which* bay to aim for - the
nearest, the best, holding a reservation - is deliberately not here. Nor is driving.

Refusals are first class. A bay that is occupied, a bay nobody can see, a car that has not been
located, a bay with no road to it, and a target with no path are five different answers, and a
navigator that cannot tell them apart is worse than one that says nothing.
"""

from __future__ import annotations

import heapq
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROUTE_SCHEMA = "route/1"

# Why a route was refused. Stable strings, so a caller can branch on them.
NO_SUCH_SPOT = "no-such-spot"
NO_ACCESS = "spot-has-no-access"
NOT_FREE = "spot-not-free"
NO_HERO = "hero-not-visible"
NO_ROUTE = "no-route"

EMPTY = "EMPTY"
OCCUPIED = "OCCUPIED"
UNKNOWN = "UNKNOWN"


@dataclass
class Route:
    """A plan, or the reason there is none. `ok` says which."""

    target_spot: str
    ok: bool
    code: str
    reason: str
    start_node: Optional[str] = None
    target_node: Optional[str] = None
    nodes: List[str] = field(default_factory=list)
    distance: Optional[float] = None

    def as_document(self) -> Dict[str, Any]:
        """The route contract. Node ids only: the geometry is already in the map, and copying
        it here would be two sources of truth for the same line."""
        return {
            "schema": ROUTE_SCHEMA,
            "generated_at": int(time.time()),
            "target_spot": self.target_spot,
            "ok": self.ok,
            "code": self.code,
            "reason": self.reason,
            "start_node": self.start_node,
            "target_node": self.target_node,
            "nodes": list(self.nodes),
            "distance": None if self.distance is None else round(self.distance, 1),
        }


def describe(route: Route) -> str:
    if not route.ok:
        return f"refused ({route.code}): {route.reason}"
    return (f"{route.target_spot}: {route.start_node} -> {route.target_node}, "
            f"{len(route.nodes)} node(s), {route.distance:.0f}px")


# ---------------------------------------------------------------------------
# Reading the two documents
# ---------------------------------------------------------------------------

def load_document(path: str) -> Dict[str, Any]:
    """Read a contract document. A missing file is an empty document, not an exception."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            document = json.load(fh)
    except (OSError, ValueError):
        return {}
    return document if isinstance(document, dict) else {}


def nodes_of(map_doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    network = map_doc.get("road_network") or {}
    return {n["id"]: n for n in (network.get("nodes") or []) if isinstance(n, dict) and "id" in n}


def adjacency_from_map(map_doc: Dict[str, Any]) -> Dict[str, List[Tuple[str, float]]]:
    """The weighted graph, as the map describes it.

    A `bidirectional: false` edge appears in one direction only. That is the whole reason the
    field exists - the graph the navigator walks must not be symmetric for a one-way road.
    """
    graph: Dict[str, List[Tuple[str, float]]] = {nid: [] for nid in nodes_of(map_doc)}
    network = map_doc.get("road_network") or {}
    for edge in network.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        a, b = edge.get("from"), edge.get("to")
        if a not in graph or b not in graph:
            continue
        weight = float(edge.get("distance") or 0.0)
        graph[a].append((b, weight))
        if edge.get("bidirectional", True):
            graph[b].append((a, weight))
    return graph


def spot_of(map_doc: Dict[str, Any], spot_id: str) -> Optional[Dict[str, Any]]:
    for spot in map_doc.get("spots") or []:
        if isinstance(spot, dict) and str(spot.get("id")) == str(spot_id):
            return spot
    return None


def nearest_node(map_doc: Dict[str, Any], x: float, y: float
                 ) -> Optional[Tuple[str, float]]:
    """The graph node closest to a point, and how far away it is.

    The hero car is a position, not a node. The map's nodes are dense - the aisles and every bay
    access point are all nodes - so the nearest one is a few tens of pixels away at most, and the
    gap is reported so a caller can see when that assumption is strained.
    """
    best: Optional[Tuple[str, float]] = None
    for node in nodes_of(map_doc).values():
        d = math.hypot(float(node.get("x", 0)) - x, float(node.get("y", 0)) - y)
        if best is None or d < best[1]:
            best = (node["id"], d)
    return best


def spot_state(state_doc: Dict[str, Any], spot_id: str) -> str:
    entry = (state_doc.get("spots") or {}).get(str(spot_id))
    if not isinstance(entry, dict):
        return UNKNOWN
    return str(entry.get("state") or UNKNOWN)


# ---------------------------------------------------------------------------
# Shortest path
# ---------------------------------------------------------------------------

def shortest_path(graph: Dict[str, List[Tuple[str, float]]], start: str, goal: str
                  ) -> Optional[Tuple[List[str], float]]:
    """Dijkstra. Returns the node list and the total weight, or None if there is no path.

    Dijkstra rather than A* because the graph is a few hundred nodes and the result is the same:
    the cost of the search is not what this system is short of, and a heuristic would only add a
    way to get the answer slightly wrong.
    """
    if start not in graph or goal not in graph:
        return None
    if start == goal:
        return [start], 0.0

    best = {start: 0.0}
    came_from: Dict[str, str] = {}
    queue: List[Tuple[float, str]] = [(0.0, start)]
    while queue:
        cost, node = heapq.heappop(queue)
        if node == goal:
            path = [node]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            return list(reversed(path)), cost
        if cost > best.get(node, math.inf):
            continue
        for neighbour, weight in graph.get(node, []):
            through = cost + weight
            if through < best.get(neighbour, math.inf):
                best[neighbour] = through
                came_from[neighbour] = node
                heapq.heappush(queue, (through, neighbour))
    return None


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def plan(map_doc: Dict[str, Any], state_doc: Dict[str, Any],
         target_spot: str) -> Route:
    """Route from the hero car's current road position to a named bay's access point."""
    spot_id = str(target_spot)

    spot = spot_of(map_doc, spot_id)
    if spot is None:
        return Route(spot_id, False, NO_SUCH_SPOT,
                     f"the map has no bay called {spot_id}")

    access = spot.get("access")
    if not access or not access.get("road_node"):
        return Route(spot_id, False, NO_ACCESS,
                     f"bay {spot_id} has no road access point in the map")

    state = spot_state(state_doc, spot_id)
    if state != EMPTY:
        # OCCUPIED and UNKNOWN are both refusals, and they are different refusals: one means
        # somebody is parked there, the other means nobody knows. Neither is a reason to guess.
        return Route(spot_id, False, NOT_FREE,
                     f"bay {spot_id} is {state}" + (" - take another bay"
                                                    if state == OCCUPIED else
                                                    " - nobody can see it yet"))

    vehicle = state_doc.get("vehicle") or {}
    if not vehicle.get("visible"):
        return Route(spot_id, False, NO_HERO,
                     "the hero car is not visible, so there is nothing to route from")

    found = nearest_node(map_doc, float(vehicle.get("x", 0.0)), float(vehicle.get("y", 0.0)))
    if found is None:
        return Route(spot_id, False, NO_ROUTE, "the map has no road network")
    start_node, gap = found

    target_node = str(access["road_node"])
    path = shortest_path(adjacency_from_map(map_doc), start_node, target_node)
    if path is None:
        return Route(spot_id, False, NO_ROUTE,
                     f"no path from {start_node} to {target_node}",
                     start_node=start_node, target_node=target_node)
    nodes, distance = path
    return Route(spot_id, True, "ok",
                 f"{len(nodes)} node(s), {distance:.0f}px, and the car is {gap:.0f}px "
                 f"from the first one",
                 start_node=start_node, target_node=target_node,
                 nodes=nodes, distance=distance)
