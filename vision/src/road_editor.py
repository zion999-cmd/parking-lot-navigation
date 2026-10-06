"""Editing model for the road network, shaped exactly like SpotEditor.

The window layer (``CalibrationApp``) drives both layers through the same handful of calls
- select, begin_drag, update_drag, end_drag, nudge, delete_selected, undo, clear, load,
save, ``_say`` - so hosting two layers in one window needs no special cases scattered
through the input code. Keeping the two editors interchangeable is the point of this file.

The drawing interaction is a polyline, because that is how a road is actually drawn: click
along the path and each new node connects to the one before it.

    入口 ──click──click──click── Spot 38

Branching is explicit: select the node the branch leaves from (which becomes the chain
head, drawn on screen) and keep clicking. 'n' breaks the chain so the next click starts a
new road somewhere else instead of joining the one you were drawing.

Nothing here is auto-detected. The road's visual edges do not matter to navigation - what
matters is the centre line a car should follow, and that is a human decision.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

from road_network import (KIND_NAMES, KIND_SPOT, KIND_TAG, NODE_KINDS,
                          RoadNetwork, RoadNode)
from spot_editor import EditMessage

MAX_UNDO = 50
# How close the pointer must be to a node to grab it instead of placing a new one.
HIT_TOLERANCE = 14.0


class RoadEditor:
    def __init__(self, shape: Tuple[int, int], network: Optional[RoadNetwork] = None):
        self.shape = shape
        self.net = network if network is not None else RoadNetwork(shape=shape)
        self.net.shape = shape
        self.selected: Optional[int] = None
        # The node the next click chains from. None means the next click starts a new road.
        self.chain: Optional[str] = None
        self.message = EditMessage()
        self.dirty = False
        self._undo: List[Tuple[List[RoadNode], list, Optional[int], Optional[str]]] = []
        self._drag: Optional[Dict] = None
        self._counter = 0

    # -- housekeeping -------------------------------------------------------

    def _say(self, text: str, level: str = "info") -> None:
        self.message = EditMessage(text, level)

    def _push_undo(self) -> None:
        self._undo.append((copy.deepcopy(self.net.nodes), copy.deepcopy(self.net.edges),
                           self.selected, self.chain))
        if len(self._undo) > MAX_UNDO:
            self._undo.pop(0)

    # -- selection ----------------------------------------------------------

    def select(self, index: Optional[int]) -> None:
        if index is None or not (0 <= index < len(self.net.nodes)):
            self.selected = None
            return
        self.selected = index
        # Touching a node makes it the chain head: that is how a branch is started, and it
        # keeps one rule instead of two ("select" here, "anchor" there).
        self.chain = self.net.nodes[index].id

    def selected_node(self) -> Optional[RoadNode]:
        if self.selected is None or not (0 <= self.selected < len(self.net.nodes)):
            return None
        return self.net.nodes[self.selected]

    def select_next(self, step: int = 1) -> None:
        if not self.net.nodes:
            self._say("no nodes yet: click on the image to draw a road")
            return
        if self.selected is None:
            self.select(0)
        else:
            self.select((self.selected + step) % len(self.net.nodes))
        node = self.selected_node()
        if node is not None:
            self._say(f"selected {node.id} ({KIND_NAMES[node.kind]})")

    # -- pointer ------------------------------------------------------------

    def hit_test(self, x: float, y: float, tol: float = HIT_TOLERANCE) -> Optional[int]:
        best, best_d = None, tol
        for i, n in enumerate(self.net.nodes):
            d = ((n.x - x) ** 2 + (n.y - y) ** 2) ** 0.5
            if d <= best_d:
                best, best_d = i, d
        return best

    def begin_drag(self, x: float, y: float) -> bool:
        """Grab a node for dragging. False means the click was on empty space."""
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
        # Absolute from the grab point, so a drag that leaves the window cannot drift.
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
            self._say("select a node first (Tab, or click one)", "warn")
            return
        self._push_undo()
        self.net.move_node(node.id, node.x + dx, node.y + dy)
        self.dirty = True

    # -- structural edits ---------------------------------------------------

    def add_at(self, x: float, y: float, kind: Optional[str] = None) -> RoadNode:
        """Place a node, chained from the chain head when there is one."""
        self._push_undo()
        node_kind = kind or self._kind_for_new()
        node = self.net.add_node(x, y, kind=node_kind)
        if self.chain is not None and self.chain != node.id:
            if self.net.connect(self.chain, node.id) is not None:
                self._say(f"{self.chain} -> {node.id} ({KIND_NAMES[node.kind]})")
            else:
                self._say(f"could not connect {self.chain} to {node.id}", "warn")
        else:
            self._say(f"{node.id} ({KIND_NAMES[node.kind]}) - chain started")
        self.chain = node.id
        self.select(self.net.index_of(node.id))
        self.dirty = True
        return node

    def _kind_for_new(self) -> str:
        """New nodes inherit the kind of the chain head, so a run of junctions is one key.

        The first node of a road is where a car comes in, so it defaults to an entrance -
        that is the common case when starting to draw.
        """
        head = self.net.node(self.chain) if self.chain else None
        if head is None:
            return "entrance"
        return head.kind if head.kind != "entrance" else "waypoint"

    def connect_to(self, index: int) -> bool:
        """Connect the chain head to an existing node: how a road joins another one."""
        if index is None or not (0 <= index < len(self.net.nodes)):
            return False
        target = self.net.nodes[index]
        if self.chain is None or self.chain == target.id:
            self.select(index)
            self._say(f"{target.id} selected - click empty space to extend from it")
            return False
        if self.net.node(self.chain) is None:
            self.chain = target.id
            return False
        self._push_undo()
        if self.net.connect(self.chain, target.id) is None:
            self._say(f"{self.chain} and {target.id} are already connected", "warn")
        else:
            self._say(f"{self.chain} -> {target.id} joined")
            self.dirty = True
        self.chain = target.id
        self.select(index)
        return True

    def break_chain(self) -> None:
        if self.chain is None:
            self._say("no chain to break: the next click starts a new road")
        else:
            self._say("chain broken: the next click starts a new road")
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
        self._say(f"deleted {node.id} and its {edges} edge(s)")
        return True

    def cycle_kind(self) -> str:
        node = self.selected_node()
        if node is None:
            self._say("select a node first (Tab, or click one)", "warn")
            return ""
        self._push_undo()
        node.kind = NODE_KINDS[(NODE_KINDS.index(node.kind) + 1) % len(NODE_KINDS)]
        if node.kind != KIND_SPOT:
            node.spot = ""
        self.dirty = True
        self._say(f"{node.id} is now a {KIND_NAMES[node.kind]}")
        return node.kind

    def bind_next_bay(self, spot_ids: Sequence[str]) -> Optional[str]:
        """Turn the selected node into the connection point for the next unconnected bay.

        Walking the bays in order is how this actually gets done: 48 connections is a lot
        of typing, and the order is already in the spot list.
        """
        node = self.selected_node()
        if node is None:
            self._say("select the node that should connect to a bay", "warn")
            return None
        taken = self.net.spot_nodes()
        free = [s for s in spot_ids if s not in taken]
        if not free:
            self._say("every bay already has a connection point", "warn")
            return None
        self._push_undo()
        previous = node.spot if node.kind == KIND_SPOT and node.spot else ""
        node.kind = KIND_SPOT
        node.spot = free[0]
        self.dirty = True
        left = len(free) - 1
        if previous and previous != node.spot:
            # Saying so matters: pressing 'b' again is how you walk to the bay you want,
            # and silently losing the previous binding would look like a bug.
            self._say(f"{node.id} rebound from bay {previous} to {free[0]} "
                      f"({left} bay(s) left)")
        else:
            self._say(f"{node.id} now connects to bay {free[0]} ({left} bay(s) left)")
        return free[0]

    def clear(self) -> None:
        self._push_undo()
        self.net.clear()
        self.selected = None
        self.chain = None
        self.dirty = True
        self._say("road network cleared")

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
        self._say(f"loaded {len(self.net.nodes)} node(s), {len(self.net.edges)} edge(s)")
        return True

    def save(self, path: str) -> bool:
        self.net.shape = self.shape
        self.net.save(path)
        self.dirty = False
        self._say(f"saved {len(self.net.nodes)} node(s), {len(self.net.edges)} edge(s)")
        return True

    # -- reporting ----------------------------------------------------------

    def problems(self, spot_ids: Sequence[str]) -> List[str]:
        return self.net.validate(spot_ids)

    def summary(self) -> str:
        return self.net.summary()

    def node_tag(self, node: RoadNode) -> str:
        tag = KIND_TAG.get(node.kind, "?")
        if node.kind == KIND_SPOT and node.spot:
            return f"{tag}{node.spot}"
        return tag
