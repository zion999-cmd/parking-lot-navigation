"""The three documents in the working tree, checked against each other.

This file exists because of a real failure: out/parking_map.json was left holding two bays from
some earlier experiment while config/parking_spots.json held forty-eight, and the board page kept
reloading it. The picture was drawn from the config and the clicks came from the map, so the page
selected bays that were nowhere near where they were shown - and nothing anywhere said so.

Reading the repository's own files is unusual in a test, and it is deliberate: the invariant is
about the working tree, and the only moment it can be checked is after everything has been run.

Run:  python tests/test_documents.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

import spot_config                                              # noqa: E402

CONFIG = os.path.join(ROOT, "config", "parking_spots.json")
MAP = os.path.join(ROOT, "out", "parking_map.json")
STATE = os.path.join(ROOT, "out", "parking_state.json")

FIX = "run:  python src/emit_map.py     (or start src/main.py, which re-emits it)"


def load(path: str):
    if not os.path.exists(path):
        raise AssertionError(f"{path} is missing - {FIX}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def polygons_of(spots) -> dict:
    return {str(s["id"] if isinstance(s, dict) else s.id):
            [[round(float(v)) for v in point] for point in
             (s["polygon"] if isinstance(s, dict) else s.polygon)]
            for s in spots}


def test_the_emitted_map_describes_the_same_bays_as_the_config() -> None:
    """Ids and positions both, because a rename changes the first and a drag changes the second."""
    config = spot_config.load_spots(CONFIG, skip_bad=True)
    emitted = load(MAP)["spots"]
    want, got = polygons_of(config), polygons_of(emitted)
    assert set(want) == set(got), (
        f"only in the config: {sorted(set(want) - set(got))[:6]}; "
        f"only in the emitted map: {sorted(set(got) - set(want))[:6]} - {FIX}")
    moved = [sid for sid in want if want[sid] != got[sid]]
    assert not moved, f"{len(moved)} bay(s) are in different places: {moved[:6]} - {FIX}"


def test_every_bay_in_the_emitted_map_still_has_an_access_point() -> None:
    bays = load(MAP)["spots"]
    missing = [s["id"] for s in bays if not s.get("access")]
    assert not missing, f"{len(missing)} bay(s) with no road access: {missing[:8]}"


def test_the_picture_the_page_shows_and_the_map_it_clicks_are_the_same_size() -> None:
    """The board draws the analysed frame and lays the map over it; different sizes would offset
    every click silently."""
    frame = load(MAP)["coordinate_system"]
    state = load(STATE)
    assert frame["width"] == 800 and frame["height"] == 600, frame
    assert "spots" in state and state["spots"], "the state must describe bays"


def test_the_state_the_page_reads_is_describing_bays_the_map_has() -> None:
    """main.py writes the state from the config it loaded; the page looks those ids up in the map.
    An id in one and not the other is a bay that exists on screen and cannot be clicked."""
    known = {str(s["id"]) for s in load(MAP)["spots"]}
    ids = set(load(STATE)["spots"])
    unknown = sorted(ids - known)
    assert not unknown, f"the state names {len(unknown)} bay(s) the map does not have: "\
                        f"{unknown[:6]} - {FIX}"


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
