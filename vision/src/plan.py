"""Plan a route to a named bay, and write it as a document.

    python src/plan.py --spot 38
    python src/plan.py --spot 38 --verbose
    python src/plan.py --spot B4 --route-out out/route.json

Reads the two contract documents P0002 emits - parking_map.json and parking_state.json - plans
from wherever the hero car is to that bay's access point, writes out/route.json, and prints what
it did. Exit code 0 when a route was planned and 1 when it was refused, so a script or a test can
tell the difference without parsing prose.

Nothing here decides anything beyond the one bay it was told to aim for.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import contract                                                 # noqa: E402
import route                                                    # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main() -> int:
    ap = argparse.ArgumentParser(description="Plan a route to a named bay")
    ap.add_argument("--spot", required=True, help="the bay to aim for, by id")
    ap.add_argument("--map", dest="map_path",
                    default=os.path.join(ROOT, "out", "parking_map.json"),
                    help="the static world (emit it with src/emit_map.py)")
    ap.add_argument("--state", dest="state_path",
                    default=os.path.join(ROOT, "out", "parking_state.json"),
                    help="the live world (written by src/main.py while it runs)")
    ap.add_argument("--route-out", default=os.path.join(ROOT, "out", "route.json"),
                    help="where to write the route document")
    ap.add_argument("--verbose", action="store_true", help="list every node in the path")
    args = ap.parse_args()

    map_doc = route.load_document(args.map_path)
    state_doc = route.load_document(args.state_path)
    if not map_doc:
        print(f"error: no map at {args.map_path} - run src/emit_map.py first", file=sys.stderr)
        return 2
    if not state_doc:
        print(f"error: no state at {args.state_path} - run src/main.py first", file=sys.stderr)
        return 2

    planned = route.plan(map_doc, state_doc, args.spot)
    contract.write_json(args.route_out, planned.as_document())

    print(f"map      {args.map_path}")
    print(f"state    {args.state_path}  ({int(state_doc.get('timestamp', 0))})")
    print(f"target   bay {args.spot}")
    print(f"route    {route.describe(planned)}")
    print(f"written  {args.route_out}")
    if args.verbose and planned.ok:
        nodes = route.nodes_of(map_doc)
        for node_id in planned.nodes:
            node = nodes.get(node_id, {})
            print(f"    {node_id:>5}  ({node.get('x', 0):6.1f},{node.get('y', 0):6.1f})  "
                  f"{node.get('type', '?')}")
    if not planned.ok:
        print("(a refusal is an answer, not a failure of the program)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
