"""The road network: the navigation layer of the parking lot map.

Two layers, deliberately separate:

    Parking Spots    the camera's question  - is this bay free?
    Road Network     the navigator's question - how does a car get there?

Nothing here takes part in occupancy detection, and nothing here is auto-detected. That is
the whole point of the layer: the *visual* edges of the road do not matter to navigation.
What matters is the centre line a car should follow, so a human places the nodes where a
car would drive and that becomes the ground truth. Drawing the road as one big polygon
would answer the wrong question.

Nodes are points on that centre line and edges connect them, so the result is a graph:

                 Spot 38
                    |
        入口 ────●────●────● 路口
                    |
                 Spot 52

A node of kind "spot" is a bay's connection point and carries the bay id, which is what
lets the navigator answer "drive to the node for Spot 38" without knowing anything about
pixels.

    entrance   the way in from outside
    junction   where roads meet
    waypoint   a bend in a road; exists to shape the path
    spot       a bay's connection point (carries the bay id in `spot`)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import map_document

KIND_ENTRANCE = "entrance"
KIND_JUNCTION = "junction"
KIND_WAYPOINT = "waypoint"
KIND_SPOT = "spot"
NODE_KINDS: Tuple[str, ...] = (KIND_ENTRANCE, KIND_JUNCTION, KIND_WAYPOINT, KIND_SPOT)

# Short glyph per kind, for the on-screen label and the header summary.
KIND_TAG = {KIND_ENTRANCE: "IN", KIND_JUNCTION: "X", KIND_WAYPOINT: ".", KIND_SPOT: "P"}
KIND_NAMES = {KIND_ENTRANCE: "entrance", KIND_JUNCTION: "junction",
              KIND_WAYPOINT: "waypoint", KIND_SPOT: "spot"}


@dataclass
class ClosestPoint:
    """Where the network passes nearest to some point.

    Either `node_id` is set (the nearest thing is a node) or `edge_index` is (the point
    lies along an edge), never both.
    """
    distance: float
    x: float
    y: float
    node_id: Optional[str] = None
    edge_index: Optional[int] = None


def _project_on_segment(px: float, py: float, ax: float, ay: float,
                        bx: float, by: float) -> Tuple[float, float]:
    """The point on segment AB closest to P, clamped to the segment."""
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-9:
        return ax, ay
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    t = min(max(t, 0.0), 1.0)
    return ax + t * dx, ay + t * dy


def bay_mouth(polygon: Sequence[Sequence[float]],
              target: Tuple[float, float],
              access_side: Optional[int] = None) -> Tuple[float, float]:
    """The midpoint of the side of the bay that faces `target`.

    A bay's polygon is the visual truth for where a car parks; which of its sides is the
    entrance is decided by which one faces the road the human drew. That is a derivation
    from two decisions a person already made, not a detection - nothing here looks at the
    image.

    `access_side` overrides that with a decided side. It is needed because the rule is
    genuinely ambiguous at the end of a row or on a corner, where two roads are about
    equally close: there, which side a car drives in from is a judgement, not a
    measurement, and the person drawing can see it.
    """
    pts = np.asarray(polygon, dtype=np.float64)
    n = len(pts)
    if access_side is not None:
        i = int(access_side) % n
        mid = (pts[i] + pts[(i + 1) % n]) / 2.0
        return (float(mid[0]), float(mid[1]))

    target_arr = np.asarray(target, dtype=np.float64)
    best: Optional[Tuple[float, np.ndarray]] = None
    for i in range(n):
        mid = (pts[i] + pts[(i + 1) % n]) / 2.0
        d = float(np.linalg.norm(mid - target_arr))
        if best is None or d < best[0]:
            best = (d, mid)
    return (float(best[1][0]), float(best[1][1]))


def nearest_edge(polygon: Sequence[Sequence[float]], x: float, y: float) -> int:
    """Index of the side the point is nearest to.

    This is what turns dropping a bay's entrance somewhere into a side of that bay: the
    person points at the side a car comes in from, and the side is what gets remembered.
    Distance to the side itself, not to its midpoint, because that is what pointing at a
    side means - and a long side should be easier to hit than a short one.
    """
    pts = np.asarray(polygon, dtype=np.float64)
    best_i, best_d = 0, None
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        px, py = _project_on_segment(float(x), float(y), float(a[0]), float(a[1]),
                                     float(b[0]), float(b[1]))
        d = float(np.hypot(px - x, py - y))
        if best_d is None or d < best_d:
            best_i, best_d = i, d
    return best_i


@dataclass
class RoadNode:
    id: str
    kind: str = KIND_WAYPOINT
    x: float = 0.0
    y: float = 0.0
    label: str = ""
    spot: str = ""            # only meaningful for KIND_SPOT: which bay it serves

    def as_dict(self) -> Dict[str, Any]:
        def num(v: float):
            r = round(float(v), 1)
            return int(r) if float(r).is_integer() else r
        out: Dict[str, Any] = {"id": self.id, "kind": self.kind,
                               "x": num(self.x), "y": num(self.y)}
        if self.label:
            out["label"] = self.label
        if self.kind == KIND_SPOT and self.spot:
            out["spot"] = self.spot
        return out

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "RoadNode":
        kind = str(raw.get("kind", KIND_WAYPOINT))
        if kind not in NODE_KINDS:
            kind = KIND_WAYPOINT
        return RoadNode(
            id=str(raw.get("id", "?")),
            kind=kind,
            x=float(raw.get("x", 0.0)),
            y=float(raw.get("y", 0.0)),
            label=str(raw.get("label", "")),
            spot=str(raw.get("spot", "")),
        )


@dataclass
class RoadEdge:
    a: str
    b: str
    oneway: bool = False       # true means a -> b only

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"a": self.a, "b": self.b}
        if self.oneway:
            out["oneway"] = True
        return out

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "RoadEdge":
        return RoadEdge(a=str(raw.get("a", "")), b=str(raw.get("b", "")),
                        oneway=bool(raw.get("oneway", False)))


@dataclass
class RoadNetwork:
    """Nodes and edges, plus the frame size they were drawn on."""

    shape: Tuple[int, int] = (0, 0)          # (height, width); 0 means unset
    nodes: List[RoadNode] = field(default_factory=list)
    edges: List[RoadEdge] = field(default_factory=list)
    # Not part of the map: how many stored bay connection points a load had to undo.
    stale_connections: int = 0

    # -- lookup -------------------------------------------------------------

    def node(self, node_id: str) -> Optional[RoadNode]:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def index_of(self, node_id: str) -> Optional[int]:
        for i, n in enumerate(self.nodes):
            if n.id == node_id:
                return i
        return None

    def has_edge(self, a: str, b: str) -> bool:
        return any((e.a == a and e.b == b) or (e.a == b and e.b == a) for e in self.edges)

    def neighbors(self, node_id: str) -> List[str]:
        out = []
        for e in self.edges:
            if e.a == node_id:
                out.append(e.b)
            elif e.b == node_id and not e.oneway:
                out.append(e.a)
        return out

    def edges_of(self, node_id: str) -> List[RoadEdge]:
        return [e for e in self.edges if e.a == node_id or e.b == node_id]

    def distance(self, a: str, b: str) -> float:
        na, nb = self.node(a), self.node(b)
        if na is None or nb is None:
            return float("inf")
        return math.hypot(na.x - nb.x, na.y - nb.y)

    def closest_point(self, x: float, y: float,
                      skip_stubs: bool = False) -> Optional[ClosestPoint]:
        """Nearest point on the network: a node when one is close, otherwise a point along
        an edge. None for an empty network.

        `skip_stubs` ignores bay connection points and the edges hanging off them. That is
        not a tweak: a bay's stub is not a road, and letting one be the "nearest road" for
        the next bay makes the result tangle. Each new stub becomes a candidate for the one
        after it, and the mouths end up pointing at the wrong side of their own bay.
        """
        stubs = {n.id for n in self.nodes if n.kind == KIND_SPOT} if skip_stubs else set()

        best: Optional[ClosestPoint] = None
        for n in self.nodes:
            if n.id in stubs:
                continue
            d = math.hypot(n.x - x, n.y - y)
            if best is None or d < best.distance:
                best = ClosestPoint(distance=d, x=n.x, y=n.y, node_id=n.id)
        for i, e in enumerate(self.edges):
            if e.a in stubs or e.b in stubs:
                continue
            a, b = self.node(e.a), self.node(e.b)
            if a is None or b is None:
                continue
            px, py = _project_on_segment(x, y, a.x, a.y, b.x, b.y)
            d = math.hypot(px - x, py - y)
            if best is None or d < best.distance:
                best = ClosestPoint(distance=d, x=px, y=py, edge_index=i)
        return best

    def split_edge(self, index: int, x: float, y: float,
                   kind: str = KIND_JUNCTION) -> Optional[RoadNode]:
        """Insert a node into an edge, replacing it with two edges.

        Needed so a bay's connection stub joins the aisle where the bay actually is, rather
        than being dragged to whichever end of the aisle happens to be nearest: on a
        sixteen-bay row that difference is hundreds of pixels, and the stub would run
        across other bays.
        """
        if not (0 <= index < len(self.edges)):
            return None
        edge = self.edges[index]
        node = self.add_node(x, y, kind=kind)
        self.edges.pop(index)
        self.connect(edge.a, node.id, oneway=edge.oneway)
        self.connect(edge.b, node.id, oneway=edge.oneway)
        return node

    def attach(self, x: float, y: float, kind: str = KIND_SPOT,
               spot: str = "", skip_stubs: bool = True) -> Optional[RoadNode]:
        """Add a node beside the network and connect it to the closest point on it.

        This is how a bay gets its connection point: the caller works out *where* (the bay's
        mouth, derived from its polygon and the road), and the network works out *what to
        join it to*, splitting an edge when the join is mid-segment.
        """
        near = self.closest_point(x, y, skip_stubs=skip_stubs)
        if near is None:
            return None
        node = self.add_node(x, y, kind=kind, spot=spot)
        if near.node_id is not None:
            self.connect(node.id, near.node_id)
            return node
        junction = self.split_edge(int(near.edge_index), near.x, near.y)
        if junction is not None:
            self.connect(node.id, junction.id)
        return node

    def next_id(self) -> str:
        """r1, r2, ... never reusing a freed number.

        The contract names road nodes r*, because "n" was ambiguous with nodes of any kind
        and the navigation layer only ever sees the road graph. SpotEditor had a bug where
        next_id() handed out a number that was still in use; the same mistake here would
        silently merge two nodes' edges, so the id is always max(existing) + 1.
        """
        highest = 0
        for n in self.nodes:
            for prefix in ("r", "n"):
                if n.id.startswith(prefix) and n.id[1:].isdigit():
                    highest = max(highest, int(n.id[1:]))
        return f"r{highest + 1}"

    # -- editing ------------------------------------------------------------

    def add_node(self, x: float, y: float, kind: str = KIND_WAYPOINT,
                 label: str = "", spot: str = "") -> RoadNode:
        """Add a node, clamped into the frame.

        Same reasoning as the bay clamp: a click or drag whose coordinates land outside the
        image must not be able to produce a map that cannot be used.
        """
        if self.shape[0] > 0 and self.shape[1] > 0:
            x = min(max(float(x), 0.0), self.shape[1] - 1.0)
            y = min(max(float(y), 0.0), self.shape[0] - 1.0)
        node = RoadNode(id=self.next_id(), kind=kind if kind in NODE_KINDS else KIND_WAYPOINT,
                        x=float(x), y=float(y), label=label, spot=spot)
        self.nodes.append(node)
        return node

    def remove_node(self, node_id: str) -> bool:
        idx = self.index_of(node_id)
        if idx is None:
            return False
        self.nodes.pop(idx)
        self.edges = [e for e in self.edges if e.a != node_id and e.b != node_id]
        return True

    def connect(self, a: str, b: str, oneway: bool = False) -> Optional[RoadEdge]:
        """Connect two nodes. Returns the new edge, or None if it would be invalid.

        Refuses self-loops and duplicates: both would be silent corruption of the graph
        (a duplicate edge would double a shortest path's weight).
        """
        if a == b or self.node(a) is None or self.node(b) is None:
            return None
        if self.has_edge(a, b):
            return None
        edge = RoadEdge(a=a, b=b, oneway=oneway)
        self.edges.append(edge)
        return edge

    def remove_edge(self, a: str, b: str) -> bool:
        before = len(self.edges)
        self.edges = [e for e in self.edges
                      if not ((e.a == a and e.b == b) or (e.a == b and e.b == a))]
        return len(self.edges) != before

    def move_node(self, node_id: str, x: float, y: float) -> None:
        node = self.node(node_id)
        if node is None:
            return
        if self.shape[0] > 0 and self.shape[1] > 0:
            x = min(max(float(x), 0.0), self.shape[1] - 1.0)
            y = min(max(float(y), 0.0), self.shape[0] - 1.0)
        node.x, node.y = float(x), float(y)

    def clear(self) -> None:
        self.nodes, self.edges = [], []

    # -- what the navigator needs -------------------------------------------

    def adjacency(self) -> Dict[str, List[Tuple[str, float]]]:
        """Weighted adjacency for A*/Dijkstra, in the sense the road network actually has.

        A oneway edge appears in one direction only, which is the whole reason the flag
        exists - the graph is what the navigator walks, so it must not be symmetric for a
        one-way road.
        """
        graph: Dict[str, List[Tuple[str, float]]] = {n.id: [] for n in self.nodes}
        for e in self.edges:
            w = self.distance(e.a, e.b)
            if e.a in graph and e.b in graph:
                graph[e.a].append((e.b, w))
                if not e.oneway:
                    graph[e.b].append((e.a, w))
        return graph

    def spot_nodes(self) -> Dict[str, str]:
        """bay id -> node id, for the bays that have a connection point."""
        return {n.spot: n.id for n in self.nodes if n.kind == KIND_SPOT and n.spot}

    def summary(self) -> str:
        if not self.nodes:
            return "road: empty"
        counts: Dict[str, int] = {}
        for n in self.nodes:
            counts[n.kind] = counts.get(n.kind, 0) + 1
        parts = [f"{counts[k]} {KIND_NAMES[k]}" for k in NODE_KINDS if counts.get(k)]
        return f"road: {len(self.nodes)} nodes ({', '.join(parts)}), {len(self.edges)} edges"

    # -- the derived navigation graph ---------------------------------------

    def _border_node(self) -> Optional[RoadNode]:
        """The road point closest to the edge of the frame.

        A road that reaches the edge of the picture is where cars come in from outside, so
        this is how the entrance is worked out rather than asked about. A person can still
        mark a different point as the entrance; this is only the default.
        """
        if not self.nodes or self.shape[0] <= 0:
            return None
        h, w = self.shape
        best: Optional[RoadNode] = None
        best_d = None
        for n in self.nodes:
            d = min(n.x, n.y, (w - 1) - n.x, (h - 1) - n.y)
            if best_d is None or d < best_d:
                best, best_d = n, d
        return best

    def _attach_bay(self, spot) -> Optional[RoadNode]:
        """Give one bay a connection point on the nearest road."""
        pts = np.asarray(spot.polygon, dtype=np.float64)
        centre = pts.mean(axis=0)
        near = self.closest_point(float(centre[0]), float(centre[1]), skip_stubs=True)
        if near is None:
            return None
        mx, my = bay_mouth(pts, (near.x, near.y), getattr(spot, "access_side", None))
        return self.attach(mx, my, kind=KIND_SPOT, spot=spot.id, skip_stubs=True)

    def derive(self, spots: Sequence) -> "RoadNetwork":
        """The navigation graph: the roads as drawn, plus one connection per bay.

        Derived, and deliberately not stored. The map file holds the two things a human
        decided - the bay polygons and the roads a car can drive - and the graph is worked
        out from those. Nothing here looks at the image: a bay's entrance is the midpoint of
        whichever of its sides faces the nearest road, which follows from the polygon and
        the road alone.

        Pure. It returns a new network and leaves the roads as drawn untouched, so calling
        it again cannot accumulate stubs, and the saved map never becomes a mix of human
        decisions and machine output. It is also deterministic: same roads, same bays, same
        result.
        """
        out = RoadNetwork(shape=self.shape,
                          nodes=[replace(n) for n in self.nodes],
                          edges=[replace(e) for e in self.edges])
        if not any(n.kind == KIND_ENTRANCE for n in out.nodes):
            entrance = out._border_node()
            if entrance is not None:
                entrance.kind = KIND_ENTRANCE
        for spot in spots:
            out._attach_bay(spot)
        return out

    def far_bays(self, spots: Sequence, limit: float) -> List[Tuple[str, float]]:
        """Bays whose entrance is further than `limit` from the nearest road.

        Reported in map terms rather than graph terms because that is the only thing the
        person drawing can act on: a bay that far away usually means a road is missing.
        """
        out: List[Tuple[str, float]] = []
        for spot in spots:
            pts = np.asarray(spot.polygon, dtype=np.float64)
            centre = pts.mean(axis=0)
            near = self.closest_point(float(centre[0]), float(centre[1]))
            if near is None:
                continue
            mouth = bay_mouth(pts, (near.x, near.y))
            d = math.hypot(mouth[0] - near.x, mouth[1] - near.y)
            if d > limit:
                out.append((spot.id, d))
        out.sort(key=lambda item: -item[1])
        return out

    # -- validation ---------------------------------------------------------

    def validate(self, spot_ids: Optional[Sequence[str]] = None) -> List[str]:
        """Everything wrong with the network, as human-readable lines.

        Returned rather than raised: a half-drawn road is the normal state while working,
        and the editor shows these in its message bar instead of refusing to open.
        """
        problems: List[str] = []

        seen: Dict[str, int] = {}
        for n in self.nodes:
            seen[n.id] = seen.get(n.id, 0) + 1
        for node_id, count in seen.items():
            if count > 1:
                problems.append(f"node {node_id} is defined {count} times")

        ids = {n.id for n in self.nodes}
        for e in self.edges:
            if e.a not in ids or e.b not in ids:
                problems.append(f"edge {e.a}-{e.b} references a node that does not exist")
            elif e.a == e.b:
                problems.append(f"edge {e.a}-{e.b} is a self-loop")

        if self.shape[0] > 0 and self.shape[1] > 0:
            h, w = self.shape
            for n in self.nodes:
                if not (0 <= n.x <= w - 1 and 0 <= n.y <= h - 1):
                    problems.append(f"node {n.id} is outside the {w}x{h} frame")

        if spot_ids is not None:
            known = set(spot_ids)
            for n in self.nodes:
                if n.kind == KIND_SPOT and n.spot and n.spot not in known:
                    problems.append(f"node {n.id} points at bay {n.spot}, which does not exist")

        for n in self.nodes:
            if n.kind == KIND_SPOT and not n.spot:
                problems.append(f"node {n.id} is a bay connection but names no bay")

        return problems

    # -- serialisation ------------------------------------------------------

    def as_dict(self) -> Dict[str, Any]:
        return {"nodes": [n.as_dict() for n in self.nodes],
                "edges": [e.as_dict() for e in self.edges]}

    @staticmethod
    def from_document(doc: Dict[str, Any], shape: Tuple[int, int] = (0, 0)) -> "RoadNetwork":
        """Read the roads from a map document.

        Any stored bay connection point is turned back into a plain road point. Those are
        derived data and no longer written, but an earlier version of the tool did write
        them - and worse, its "bind this point to a bay" action could convert one of the
        user's own road points into a connection point, which is how two corners of a drawn
        ring ended up labelled as bays. Turning them back keeps the geometry: deleting them
        would have cut the ring in half.
        """
        nodes: List[RoadNode] = []
        stale = 0
        renames: Dict[str, str] = {}
        for raw in doc.get("nodes", []) or []:
            if not isinstance(raw, dict):
                continue
            node = RoadNode.from_dict(raw)
            # Older maps named road nodes n1, n2, ...; the contract uses r*. Renaming here
            # keeps a drawn map usable instead of demanding it be redrawn.
            if node.id.startswith("n") and node.id[1:].isdigit():
                renames[node.id] = "r" + node.id[1:]
                node.id = renames[node.id]
            if node.kind == KIND_SPOT:
                node.kind = KIND_WAYPOINT
                node.spot = ""
                stale += 1
            nodes.append(node)
        edges = []
        for r in doc.get("edges", []) or []:
            if not isinstance(r, dict):
                continue
            edge = RoadEdge.from_dict(r)
            edge.a = renames.get(edge.a, edge.a)
            edge.b = renames.get(edge.b, edge.b)
            edges.append(edge)
        size = doc.get("image_size")
        if isinstance(size, (list, tuple)) and len(size) == 2:
            try:
                shape = (int(size[0]), int(size[1]))
            except (TypeError, ValueError):
                pass
        return RoadNetwork(shape=shape, nodes=nodes, edges=edges, stale_connections=stale)

    @staticmethod
    def load(path: str, shape: Tuple[int, int] = (0, 0)) -> "RoadNetwork":
        return RoadNetwork.from_document(map_document.read(path), shape)

    def save(self, path: str) -> None:
        """Write the road sections, leaving the bays in the same file alone."""
        sections = map_document.build(
            path,
            image_size=list(self.shape) if self.shape[0] > 0 else None,
            nodes=[n.as_dict() for n in self.nodes],
            edges=[e.as_dict() for e in self.edges],
        )
        sections.setdefault("comment",
                            '"Parking lot map: bays (camera layer) and road network '
                            '(navigation layer). Edit with src/calibration.py."')
        map_document.write(path, sections)
