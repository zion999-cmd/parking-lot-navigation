"""Drawing roads on the parking lot map.

The person using this draws one thing: where a car can drive. That is the whole interface.

Everything else that the navigator needs is derived, not asked for:

    bay polygons (already marked)  +  roads (drawn)  ->  bay entrances  ->  graph

A bay's entrance is the midpoint of whichever of its sides faces the nearest road, and when
that lands in the middle of a road the road gains a point there. Nobody has to place 48
points, and nobody has to know that any of it involves nodes, edges or chains - those are
the shapes of the data, not part of the job.

So this file holds only the roads as drawn. The graph is built by RoadNetwork.derive(),
which is pure, so it is recomputed whenever it is needed rather than cached: 48 bays against
a handful of road segments costs microseconds, and a cache is how derived data goes stale.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

from road_network import (KIND_ENTRANCE, KIND_NAMES, KIND_TAG, KIND_WAYPOINT,
                          RoadNetwork, RoadNode)
from spot_editor import EditMessage

MAX_UNDO = 50
# How close the pointer must be to a road point to grab it instead of starting a new one.
HIT_TOLERANCE = 14.0
# Beyond this a bay's entrance is far enough from every road that something is probably
# missing from the drawing. Reported as "this bay is far from any road", which is the only
# form of the problem a person can act on.
FAR_BAY_LIMIT = 220.0


class RoadEditor:
    def __init__(self, shape: Tuple[int, int], network: Optional[RoadNetwork] = None):
        self.shape = shape
        # The roads as drawn. This is what gets saved; nothing derived is stored here.
        self.net = network if network is not None else RoadNetwork(shape=shape)
        self.net.shape = shape
        self.selected: Optional[int] = None
        # The point the next click continues from. None means the next click starts a road.
        self.chain: Optional[str] = None
        self.dirty = False
        self.message = EditMessage()
        self._undo: List[Tuple[List[RoadNode], list, Optional[int], Optional[str]]] = []
        self._drag: Optional[Dict] = None

    # -- housekeeping -------------------------------------------------------

    def _say(self, text: str, level: str = "info") -> None:
        self.message = EditMessage(text, level)

    def _push_undo(self) -> None:
        self._undo.append((copy.deepcopy(self.net.nodes), copy.deepcopy(self.net.edges),
                           self.selected, self.chain))
        if len(self._undo) > MAX_UNDO:
            self._undo.pop(0)

    def derived(self, spots: Sequence) -> RoadNetwork:
        """The graph a navigator would use: roads plus one entrance per bay."""
        return self.net.derive(spots)

    # -- selection ----------------------------------------------------------

    def select(self, index: Optional[int]) -> None:
        if index is None or not (0 <= index < len(self.net.nodes)):
            self.selected = None
            return
        self.selected = index
        # Touching a point makes it the one the next click continues from: that is one rule
        # instead of two ("select" here, "anchor" there), and it is how a road is joined to
        # another one.
        self.chain = self.net.nodes[index].id

    def selected_node(self) -> Optional[RoadNode]:
        if self.selected is None or not (0 <= self.selected < len(self.net.nodes)):
            return None
        return self.net.nodes[self.selected]

    def select_next(self, step: int = 1) -> None:
        if not self.net.nodes:
            self._say("no roads yet: click on the image to draw where a car can drive")
            return
        if self.selected is None:
            self.select(0)
        else:
            self.select((self.selected + step) % len(self.net.nodes))
        node = self.selected_node()
        if node is not None:
            where = "" if node.kind != KIND_ENTRANCE else " - the way in from outside"
            self._say(f"{node.id} selected{where}")

    # -- pointer ------------------------------------------------------------

    def hit_test(self, x: float, y: float, tol: float = HIT_TOLERANCE) -> Optional[int]:
        best, best_d = None, tol
        for i, n in enumerate(self.net.nodes):
            d = ((n.x - x) ** 2 + (n.y - y) ** 2) ** 0.5
            if d <= best_d:
                best, best_d = i, d
        return best

    def begin_drag(self, x: float, y: float) -> bool:
        """Grab a point to move it. False means the click was on empty space."""
        idx = self.hit_test(x, y)
        if idx is None:
            return False
        self._push_undo()
        self.select(idx)
        self._drag = {"index": idx, "last": (x, y)}
        return True

    def update_drag(self, x: float, y: float) -> None:
        if self._drag is None:
            return
        idx = self._drag["index"]
        if not (0 <= idx < len(self.net.nodes)):
            return
        node = self.net.nodes[idx]
        self.net.move_node(node.id, node.x + (x - self._drag["last"][0]),
                           node.y + (y - self._drag["last"][1]))
        self._drag["last"] = (x, y)
        self.dirty = True

    def end_drag(self) -> None:
        if self._drag is None:
            return
        self._drag = None
        self.dirty = True

    def nudge(self, dx: float, dy: float) -> None:
        node = self.selected_node()
        if node is None:
            self._say("select a road point first (Tab, or click one)", "warn")
            return
        self._push_undo()
        self.net.move_node(node.id, node.x + dx, node.y + dy)
        self.dirty = True

    # -- drawing ------------------------------------------------------------

    def add_at(self, x: float, y: float) -> RoadNode:
        """Place a point, continuing the road being drawn when there is one."""
        self._push_undo()
        node = self.net.add_node(x, y, kind=KIND_WAYPOINT)
        if self.chain is not None and self.chain != node.id:
            if self.net.connect(self.chain, node.id) is not None:
                self._say(f"road continues to {node.id}")
            else:
                self._say(f"could not extend the road to {node.id}", "warn")
        else:
            self._say("road started - click along the way cars drive")
        self.chain = node.id
        self.select(self.net.index_of(node.id))
        self.dirty = True
        return node

    def connect_to(self, index: int) -> bool:
        """Join the road being drawn to an existing road: how two roads meet."""
        if index is None or not (0 <= index < len(self.net.nodes)):
            return False
        target = self.net.nodes[index]
        if self.chain is None or self.chain == target.id:
            self.select(index)
            self._say(f"{target.id} selected - click empty space to continue from here")
            return False
        if self.net.node(self.chain) is None:
            self.chain = target.id
            return False
        self._push_undo()
        if self.net.connect(self.chain, target.id) is None:
            self._say("these two points are already joined", "warn")
        else:
            self._say("roads joined")
            self.dirty = True
        self.chain = target.id
        self.select(index)
        return True

    def break_chain(self) -> None:
        self._say("next click starts another road")
        self.chain = None

    def delete_selected(self) -> bool:
        node = self.selected_node()
        if node is None:
            self._say("nothing selected to delete", "warn")
            return False
        self._push_undo()
        edges = len(self.net.edges_of(node.id))
        self.net.remove_node(node.id)
        if self.chain == node.id:
            self.chain = None
        self.selected = None
        self.dirty = True
        self._say(f"deleted {node.id}" + (f" and its {edges} road segment(s)" if edges else ""))
        return True

    def toggle_entrance(self) -> bool:
        """Mark the selected point as the way in from outside, or unmark it.

        Only an override: the entrance is worked out as the road point closest to the edge
        of the picture, so a person normally does not have to say anything about it.
        """
        node = self.selected_node()
        if node is None:
            self._say("select the road point that is the way in", "warn")
            return False
        self._push_undo()
        if node.kind == KIND_ENTRANCE:
            node.kind = KIND_WAYPOINT
            self._say(f"{node.id} is no longer marked as the way in "
                      f"(the automatic choice will be used)")
        else:
            for other in self.net.nodes:
                if other.kind == KIND_ENTRANCE:
                    other.kind = KIND_WAYPOINT
            node.kind = KIND_ENTRANCE
            self.dirty = True
            self._say(f"{node.id} is where cars come in")
        return True

    def clear(self) -> None:
        self._push_undo()
        self.net.clear()
        self.selected = None
        self.chain = None
        self.dirty = True
        self._say("roads cleared")

    def undo(self) -> bool:
        if not self._undo:
            self._say("nothing to undo", "warn")
            return False
        nodes, edges, selected, chain = self._undo.pop()
        self.net.nodes, self.net.edges = nodes, edges
        self.selected, self.chain = selected, chain
        self.dirty = True
        self._say("undo")
        return True

    # -- persistence --------------------------------------------------------

    def load(self, path: str) -> bool:
        self.net = RoadNetwork.load(path, self.shape)
        self.net.shape = self.shape
        self.selected = None
        self.chain = None
        self.dirty = False
        self._undo = []
        self._say(f"loaded {len(self.net.nodes)} road point(s)")
        return True

    def save(self, path: str) -> bool:
        self.net.shape = self.shape
        self.net.save(path)
        self.dirty = False
        self._say(f"saved {len(self.net.nodes)} road point(s)")
        return True

    # -- reporting ----------------------------------------------------------

    def problems(self, spots: Sequence) -> List[str]:
        """What is wrong with the map, in map terms.

        Deliberately not a graph report. Whoever is drawing roads can only act on "no roads
        yet", "this bay is far from any road" and "this bay cannot be reached", so that is
        what comes out.
        """
        if not self.net.nodes:
            return ["no roads drawn yet"]

        problems: List[str] = []
        graph_net = self.derived(spots)
        problems.extend(graph_net.validate())

        far = self.net.far_bays(spots, FAR_BAY_LIMIT)
        if far:
            shown = ", ".join(f"{sid} ({d:.0f}px)" for sid, d in far[:4])
            more = f" (+{len(far) - 4} more)" if len(far) > 4 else ""
            problems.append(f"{len(far)} bay(s) are a long way from any road: {shown}{more}")

        adjacency = graph_net.adjacency()
        start = next((n.id for n in graph_net.nodes if n.kind == KIND_ENTRANCE), None)
        reachable = set()
        if start:
            stack = [start]
            reachable.add(start)
            while stack:
                for nxt, _ in adjacency.get(stack.pop(), []):
                    if nxt not in reachable:
                        reachable.add(nxt)
                        stack.append(nxt)
        stranded = [sid for sid, nid in graph_net.spot_nodes().items()
                    if nid not in reachable]
        if stranded:
            problems.append(f"{len(stranded)} bay(s) cannot be reached from the entrance: "
                            + ", ".join(stranded[:5]))
        return problems

    def stats(self, spots: Sequence) -> str:
        """One line for the header: what is drawn, and what was derived from it."""
        if not self.net.nodes:
            return "ROADS   nothing drawn yet - click along where cars drive"
        graph_net = self.derived(spots)
        bays = len(graph_net.spot_nodes())
        manual = sum(1 for s in spots if getattr(s, "entrance_edge", None) is not None)
        extra = f", {manual} put there by hand" if manual else ""
        segments = len(self.net.edges)
        return (f"ROADS   {len(self.net.nodes)} points, {segments} segments   |   "
                f"{bays}/{len(spots)} bay entrances derived{extra}")

    def node_tag(self, node: RoadNode) -> str:
        return KIND_TAG.get(node.kind, ".")
