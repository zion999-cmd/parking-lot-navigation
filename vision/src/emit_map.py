"""Emit the static half of the P0002 contract: parking_map.json.

    python src/emit_map.py
    python src/emit_map.py --config config/parking_spots.json --out out/parking_map.json
    python src/emit_map.py --strict      # fail if any bay has no road access

This is the boundary. Everything downstream reads the emitted document and never touches the
camera, the ROIs, the bay polygons as drawn or the road centre line: the access point of
each bay, the navigation graph and the car park gates are all worked out here, because
deriving them is P0002's job rather than the navigator's.

The editable map (what a person draws) and this document (what a program reads) are
deliberately different files. The first is hand-editable and holds only decisions; the
second is generated, holds the derived graph as well, and is free to change shape whenever
the contract does.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import contract                                                 # noqa: E402
from road_network import RoadNetwork                            # noqa: E402
from spot_config import SpotConfigError, load_spots              # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULT_CONFIG = os.path.join(ROOT, "config", "parking_spots.json")
DEFAULT_OUT = os.path.join(ROOT, "out", "parking_map.json")


def out_for(config: str) -> str:
    """Where the emitted map for this config belongs.

    The real config emits to the repository's out/, and anything else - a copy a test made, an
    experiment in /tmp - emits beside itself. That rule exists because the opposite happened: the
    editor re-emits on save, and a test that pointed it at a copy of the map overwrote the real
    out/parking_map.json with a two-bay fixture, leaving the page clicking bays that were not there.
    """
    if os.path.abspath(config) == DEFAULT_CONFIG:
        return DEFAULT_OUT
    return os.path.join(os.path.dirname(os.path.abspath(config)), "parking_map.json")


def emit(config: str, out: Optional[str] = None, strict: bool = False,
         quiet: bool = False) -> int:
    """Emit the contract from an editable map. Returns 0, or 1 for a strict failure.

    Separated from main() so the editor can keep the two documents in step: a rename changes a
    bay's id in the config, and the page reads the ids in the emitted map, so a config saved
    without an emit leaves the page describing a bay that no longer exists.
    """
    out = out or out_for(config)
    shape = _shape_of(config)
    if shape is None:
        print(f"error: {config} does not say what size frame it was drawn on", file=sys.stderr)
        return 2
    try:
        spots = load_spots(config, shape)
    except SpotConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    roads = RoadNetwork.load(config, shape)
    document = contract.build_map(spots, roads.derive(spots), shape)
    contract.write_json(out, document)

    if not quiet:
        print(f"map      {config}")
        print(f"         {contract.summarize_map(document)}")
        print(f"emitted  {out}")

    missing = [s["id"] for s in document["spots"] if not s.get("access")]
    if missing:
        shown = ", ".join(missing[:8])
        more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        print(f"warning: {len(missing)} bay(s) have no road access: {shown}{more}",
              file=sys.stderr)
        print("         draw a road in front of them, or check with 'p' in the editor",
              file=sys.stderr)
        if strict:
            return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Emit parking_map.json (the P0002 contract)")
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "parking_spots.json"),
                    help="the editable map: bays as marked and roads as drawn")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "parking_map.json"),
                    help="where to write the emitted document")
    ap.add_argument("--strict", action="store_true",
                    help="fail if any bay has no road access (useful in a check)")
    args = ap.parse_args()

    return emit(args.config, args.out, strict=args.strict)


def _shape_of(path: str):
    """The frame size the map was drawn on, from the document itself."""
    import map_document
    return map_document.read_shape(path)


if __name__ == "__main__":
    raise SystemExit(main())
