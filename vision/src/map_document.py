"""The parking lot map document: one file, two layers.

    {
      "comment":     "...",
      "image_size":  [600, 800],
      "spots":       [ ... ],     Parking Spots  - the camera's question
      "nodes":       [ ... ],     Road Network   - the navigator's question
      "edges":       [ ... ]
    }

The two layers answer different questions and are edited by different modes of the same
tool. The rule that matters here is one thing only:

    writing one layer must never destroy the other.

So each layer serialises *its own* section and this module assembles the document around
them. That is why the sections arrive as already-formatted JSON text rather than objects:
the bays are one line each on purpose (a file of polygons with json.dump(indent=2) is
almost impossible to read or repair by hand), and the road layer formats its own section
to suit itself. Neither layer has to know how the other one likes to look.

Why one file instead of two: a map whose bays and roads can disagree about the frame size,
or that can be updated in one place and not the other, is worse than a slightly larger
file. The vision code reads the same document and simply ignores the keys it does not know.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

# Order in which sections are written, so a diff shows the same thing every time.
SECTION_ORDER = ("comment", "image_size", "spots", "nodes", "edges")


def read(path: str) -> Dict[str, Any]:
    """The document as a dict. A missing or unreadable file reads as empty.

    Deliberately not raising: a missing map is a normal first-run state, and every caller
    has a sensible answer for it (no bays, no roads).
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def read_shape(path: str) -> Optional[Tuple[int, int]]:
    """The frame size the map was drawn on, as (height, width), or None."""
    size = read(path).get("image_size")
    if isinstance(size, (list, tuple)) and len(size) == 2:
        try:
            h, w = int(size[0]), int(size[1])
        except (TypeError, ValueError):
            return None
        return (h, w) if h > 0 and w > 0 else None
    return None


def format_objects(items: List[Dict[str, Any]]) -> str:
    """A JSON array of objects, one object per line.

    json.dumps(indent=2) puts every number on its own line, which makes a file full of
    polygons or waypoints nearly impossible to read or repair in an editor. One object per
    line keeps it scannable and it is still plain JSON.

    Generic on purpose: both layers use it, so neither has to know how the other one
    likes to look, and neither has to import the other.
    """
    if not items:
        return "[]"
    rows = []
    for item in items:
        fields = ", ".join(
            f"{json.dumps(str(k))}: {json.dumps(v, ensure_ascii=False, separators=(',', ':'))}"
            for k, v in item.items())
        rows.append("    { " + fields + " }")
    return "[\n" + ",\n".join(rows) + "\n  ]"


def build(path: str, **overrides: Any) -> Dict[str, str]:
    """Section texts for the document at *path*, with *overrides* applied.

    This is what makes "saving one layer never destroys the other" a property of the code
    rather than a promise in a comment: a layer passes only its own sections, and every
    other section it found in the file is re-emitted untouched.
    """
    doc = read(path)
    doc.update({k: v for k, v in overrides.items() if v is not None})

    sections: Dict[str, str] = {}
    for key in SECTION_ORDER:
        if key not in doc:
            continue
        value = doc[key]
        if isinstance(value, list) and all(isinstance(v, dict) for v in value):
            sections[key] = format_objects(value)
        else:
            sections[key] = json.dumps(value, ensure_ascii=False)
    return sections


def render(sections: Dict[str, str]) -> str:
    """Assemble the document text from pre-formatted section values."""
    parts: List[str] = ["{"]
    keys = [k for k in SECTION_ORDER if k in sections]
    keys += [k for k in sections if k not in SECTION_ORDER]      # anything unexpected last
    for i, key in enumerate(keys):
        comma = "," if i < len(keys) - 1 else ""
        value = sections[key].strip()
        lines = value.splitlines() or ["null"]
        parts.append(f'  "{key}": {lines[0]}')
        parts.extend(lines[1:])
        parts[-1] = parts[-1] + comma
    parts.append("}")
    return "\n".join(parts) + "\n"


def write(path: str, sections: Dict[str, str]) -> None:
    """Write the document atomically.

    Same reasoning as everywhere else in this project that writes a file the running
    system reads: cv2.imwrite and open(..., "w") both truncate in place, so a reader that
    looks at the wrong moment gets half a file. Write beside the target, then move it.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    text = render(sections)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".map-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
