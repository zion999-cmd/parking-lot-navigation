"""The P0002 output contract: everything past this layer is allowed to know.

P0002 answers two questions and nothing else:

    parking_map.json      what the car park IS      static, changes only when re-mapped
    parking_state.json    what is happening NOW     live

Downstream must never have to look at a camera, an ROI, a changed_ratio or a road centre
line again. That is the whole point of the boundary: the navigation layer is handed the
world, not a video stream.

P0002 states facts. It does not decide:

    38 is EMPTY                                   a fact      -> P0002 says it
    the access point of 38 is at (x, y), on r17   a fact      -> P0002 says it
    38 is the best bay to take                    a decision  -> P0003 makes it
    drive r12 -> r17 -> 38                        a plan      -> P0003 makes it
    turn right                                    a decision  -> P0003 makes it

Every spot state is one of EMPTY, OCCUPIED or UNKNOWN. UNKNOWN exists because the
alternative is lying: a camera that has stopped, or a bay that is hidden, or a reference
frame that is missing, must not be reported as EMPTY. A navigator can be told "come and
look again"; it cannot recover from being told a bay is free when nobody knows.

Coordinate system: one 2D image frame, pixels, origin top left, y increasing downward.
Every coordinate in both documents - bay polygons, access points, road nodes, the vehicle -
is in that one frame, so nothing downstream has to convert anything.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence

from road_network import KIND_ENTRANCE, KIND_JUNCTION, KIND_SPOT, KIND_WAYPOINT

# Bumped when a shape changes in a way a consumer would notice.
MAP_SCHEMA = "parking_map/1"
STATE_SCHEMA = "parking_state/1"

# What a spot state may be. UNKNOWN is not a failure state, it is an admission.
EMPTY = "EMPTY"
OCCUPIED = "OCCUPIED"
UNKNOWN = "UNKNOWN"

# Node kinds as they appear in the map, named for what they are rather than for how the
# editor happens to store them.
_NODE_TYPE = {
    KIND_ENTRANCE: "entrance",
    KIND_JUNCTION: "junction",
    KIND_WAYPOINT: "waypoint",
    KIND_SPOT: "spot_access",
}


def coordinate_system(shape: Sequence[int]) -> Dict[str, Any]:
    """The one frame every coordinate in every document refers to."""
    height, width = int(shape[0]), int(shape[1])
    return {"type": "image_2d", "width": width, "height": height, "origin": "top_left"}


def build_map(spots: Sequence, graph, shape: Sequence[int]) -> Dict[str, Any]:
    """The static world: where the bays are, and the graph a car can drive on.

    `graph` is the derived graph (roads + one access node per bay), not the roads as drawn:
    the navigation layer should not have to re-derive anything, which is why this document
    is emitted rather than left to be worked out on the other side.
    """
    state_nodes = {n.id: n for n in graph.nodes}
    access_of: Dict[str, str] = {}          # bay id -> access node id
    access_point: Dict[str, List[float]] = {}
    for node in graph.nodes:
        if node.kind == KIND_SPOT and node.spot:
            access_of[node.spot] = node.id
            access_point[node.spot] = [round(float(node.x), 1), round(float(node.y), 1)]

    out_spots: List[Dict[str, Any]] = []
    for spot in spots:
        entry: Dict[str, Any] = {
            "id": spot.id,
            "polygon": [[round(float(x), 1), round(float(y), 1)] for x, y in spot.polygon],
        }
        node_id = access_of.get(spot.id)
        if node_id is not None:
            # access, not entrance: "entrance" in this project means the way into the car
            # park, and the two would be confused constantly.
            entry["access"] = {"point": access_point[spot.id], "road_node": node_id}
        else:
            entry["access"] = None          # no road reaches this bay yet
        out_spots.append(entry)

    entrances: List[Dict[str, Any]] = []
    for i, node in enumerate(n for n in graph.nodes if n.kind == KIND_ENTRANCE):
        entrances.append({"id": "main_gate" if i == 0 else f"gate_{i + 1}",
                      "node": node.id, "type": "entry_exit"})

    return {
        "schema": MAP_SCHEMA,
        "generated_at": int(time.time()),
        "coordinate_system": coordinate_system(shape),
        "spots": out_spots,
        "road_network": {
            "nodes": [{"id": n.id, "x": round(float(n.x), 1), "y": round(float(n.y), 1),
                       "type": _NODE_TYPE.get(n.kind, "waypoint")}
                      for n in graph.nodes if n.id in state_nodes],
            "edges": [{"from": e.a, "to": e.b,
                       "distance": round(graph.distance(e.a, e.b), 1),
                       "bidirectional": not e.oneway}
                      for e in graph.edges],
        },
        "entrances": entrances,
    }


def build_state(results: Dict[str, Any], vehicle: Optional[Dict[str, Any]] = None,
                unknown_reason: Optional[str] = None,
                timestamp: Optional[int] = None) -> Dict[str, Any]:
    """The live world: which bays are free, and where the car is.

    `results` maps bay id -> something with `state` and `changed_ratio` (a SpotResult).
    `unknown_reason` says why nothing is known at all right now - no reference frame, no
    frames arriving - and then every bay is UNKNOWN rather than EMPTY.
    """
    spots: Dict[str, Any] = {}
    for spot_id, r in results.items():
        if unknown_reason is not None:
            spots[spot_id] = {"state": UNKNOWN, "reason": unknown_reason}
            continue
        entry: Dict[str, Any] = {"state": getattr(r, "state", UNKNOWN)}
        ratio = getattr(r, "changed_ratio", None)
        if ratio is not None:
            # Diagnostic only. A consumer may show it; it must not make decisions from it,
            # because the threshold in here is P0002's business.
            entry["changed_ratio"] = round(float(ratio), 4)
        spots[spot_id] = entry

    if vehicle is None:
        vehicle = {"id": "hero", "visible": False}
    return {
        "schema": STATE_SCHEMA,
        "timestamp": int(timestamp if timestamp is not None else time.time()),
        "spots": spots,
        "vehicle": vehicle,
    }


def unknown_state(spot_ids: Sequence[str], reason: str,
                  vehicle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A state document for when nothing can be said. Every bay UNKNOWN, with the reason."""
    return {
        "schema": STATE_SCHEMA,
        "timestamp": int(time.time()),
        "spots": {sid: {"state": UNKNOWN, "reason": reason} for sid in spot_ids},
        "vehicle": vehicle if vehicle is not None else {"id": "hero", "visible": False},
    }


def write_json(path: str, document: Dict[str, Any]) -> None:
    """Write a contract document atomically.

    Same reasoning as everywhere else in this project that writes a file something else
    reads: truncating in place means a reader that looks at the wrong moment gets half a
    document, and half a map is worse than a stale one.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".contract-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(document, fh, ensure_ascii=False, indent=2, sort_keys=False)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def summarize_map(document: Dict[str, Any]) -> str:
    spots = document.get("spots", [])
    linked = [s for s in spots if s.get("access")]
    nodes = document.get("road_network", {}).get("nodes", [])
    edges = document.get("road_network", {}).get("edges", [])
    entrances = document.get("entrances", [])
    return (f"{len(spots)} spot(s), {len(linked)} with an access point, "
            f"{len(nodes)} road nodes, {len(edges)} edges, {len(entrances)} entrance(s)")


def summarize_state(document: Dict[str, Any]) -> str:
    counts: Dict[str, int] = {}
    for entry in document.get("spots", {}).values():
        counts[entry.get("state", UNKNOWN)] = counts.get(entry.get("state", UNKNOWN), 0) + 1
    parts = ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
    vehicle = document.get("vehicle", {})
    where = ("not localised" if not vehicle.get("visible")
             else f"at ({vehicle.get('x')}, {vehicle.get('y')}) heading {vehicle.get('heading')}")
    return f"{parts or 'no spots'}; vehicle {where}"
