"""Renaming a bay, end to end: config, keys, and the emitted map the page reads.

Reported twice from the same place. First "renaming did not work", which was the window's hint
telling the user to press a key that captured a reference frame instead. Then "the page did not
refresh", which was this: the editor wrote the editable map, and the page reads the *emitted* one,
so a renamed bay existed in the state under a new id while the page still described the old one.

Both are checked here, against a copy of the real map, without a display.

Run:  python tests/test_rename.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
REAL_CONFIG = os.path.join(HERE, "..", "config", "parking_spots.json")
sys.path.insert(0, SRC)

import emit_map                                                 # noqa: E402
import spot_config                                              # noqa: E402
import spot_editor                                              # noqa: E402


def a_loaded_editor(tmp: str):
    cfg = os.path.join(tmp, "parking_spots.json")
    shutil.copy(REAL_CONFIG, cfg)
    editor = spot_editor.SpotEditor((600, 800))
    assert editor.load(cfg), "could not load the copy"
    return editor, cfg


# ---------------------------------------------------------------------------
# The rename itself
# ---------------------------------------------------------------------------

def test_renaming_a_bay_works_and_refuses_only_empty_or_duplicate_ids() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        editor, _ = a_loaded_editor(tmp)
        original = [s.id for s in editor.spots[:3]]
        editor.select(0)
        assert editor.rename_selected("A1") is True
        assert editor.spots[0].id == "A1" and editor.dirty is True
        assert editor.rename_selected("") is False, "an empty id is not a name"
        assert editor.rename_selected(original[1]) is False, "a duplicate id would be ambiguous"
        assert editor.spots[0].id == "A1", "a refused rename must not half-apply"


def test_a_renamed_bay_survives_saving_and_reloading_even_in_chinese() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        editor, cfg = a_loaded_editor(tmp)
        editor.select(0)
        assert editor.rename_selected("东 1") is True
        assert editor.save(cfg) is True
        again = spot_config.load_spots(cfg, skip_bad=True)
        assert again[0].id == "东 1", [s.id for s in again[:3]]


# ---------------------------------------------------------------------------
# The rename reaches the document the page reads
# ---------------------------------------------------------------------------

def test_a_rename_reaches_the_emitted_map() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        editor, cfg = a_loaded_editor(tmp)
        out = os.path.join(tmp, "parking_map.json")
        assert emit_map.emit(cfg, out, quiet=True) == 0
        assert json.load(open(out, encoding="utf-8"))["spots"][0]["id"] != "ZZ9"

        editor.select(0)
        assert editor.rename_selected("ZZ9") is True
        assert editor.save(cfg) is True
        assert emit_map.emit(cfg, out, quiet=True) == 0

        after = json.load(open(out, encoding="utf-8"))
        assert after["spots"][0]["id"] == "ZZ9", [b["id"] for b in after["spots"][:3]]
        assert after["spots"][0]["access"] is not None, "the emitted map must stay complete"


def test_saving_in_the_editor_keeps_the_emitted_map_in_step_by_itself() -> None:
    """The editor does it, not the person: _save_all() re-emits, so a rename cannot sit in the
    config while the page reads a map that does not have it."""
    src = open(os.path.join(SRC, "calibration.py"), encoding="utf-8").read()
    body = src[src.index("def _save_all"):]
    body = body[:body.index("\n    def ", 10)]
    assert "emit_map.emit(" in body, "_save_all must re-emit the contract"
    assert "if not wrote:" in body, "and must have nothing to do when nothing changed"


def test_emit_map_still_works_as_a_command() -> None:
    """emit() was split out of main(); the command people already use must be unaffected."""
    src = open(os.path.join(SRC, "emit_map.py"), encoding="utf-8").read()
    assert "def emit(" in src and "def main() -> int:" in src
    assert "return emit(args.config, args.out, strict=args.strict)" in src, \
        "main() must delegate rather than duplicate"


# ---------------------------------------------------------------------------
# The key arrangement, which is what "did not work" was really about
# ---------------------------------------------------------------------------

def test_the_window_says_which_key_renames_in_the_window_you_are_looking_at() -> None:
    editor_src = open(os.path.join(SRC, "calibration.py"), encoding="utf-8").read()
    main_src = open(os.path.join(SRC, "main.py"), encoding="utf-8").read()
    tail = main_src[main_src.index("def run_edit_mode("):main_src.index("def main() -> int:")]
    assert 'self.rename_hint = "r"' in editor_src, "the standalone default"
    assert "{self.rename_hint} rename" in editor_src, "the hint must use it, not a literal"
    assert "app.rename_hint" not in tail, "the edit mode must not override it: r renames there too"
    assert "key == ord('R')" in tail, "the reference capture must be uppercase R"
    assert "key == ord('r')" not in tail, "and r must reach the editor, where it renames"
    assert "r = rename" in tail, "the mode says out loud what each key does"


def test_leaving_the_editor_flushes_what_is_still_dirty() -> None:
    main_src = open(os.path.join(SRC, "main.py"), encoding="utf-8").read()
    tail = main_src[main_src.index("def run_edit_mode("):main_src.index("def main() -> int:")]
    assert "app._save_all()" in tail, "the edit mode must flush on the way out"
    assert tail.index("app._save_all()") > tail.index("cv2.destroyAllWindows()"), \
        "and after the loop, not before it"


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
