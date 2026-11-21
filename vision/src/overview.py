"""P0007: put vision, magnetometer and RFID side by side, and say how fresh each is.

Not a fusion layer. Each source answers its own question and is shown with the time it was last
heard from:

    vision -> which bays have a car            (out/parking_state.json, written by main.py)
    mag    -> has this bay's field moved       (/api/mag + the baseline in vision/data/)
    rfid   -> which tags are on the sandbox    (/api/rfid + tools/rfid/rfid_presence.py)

A source that has gone quiet is reported as stale or missing; its last values are still shown but
never as if they were current. Nothing here changes what vision decides, what the magnetometer
decides, or what the strip does.
"""

import json
import os
import sys
import time

# A source older than this is not "what is true now". The board's magnetometer is polled at 4 Hz and
# main.py writes the state a few times a second, so seconds-scale thresholds are generous.
VISION_STALE_S = 5.0
MAG_STALE_S = 3.0
RFID_STALE_S = 3.0

BAYS = ("B1", "B2", "B3", "B4")


def _link(ok, age, stale_after):
    if not ok:
        return "missing"
    if age is None:
        return "unknown"
    return "ok" if age <= stale_after else "stale"


def vision_block(state_document, now, path=None):
    """Everything main.py already publishes, plus how long ago it was written."""
    if not state_document:
        return {"ok": False, "link": "missing", "age_s": None, "error": f"no state document",
                "bays": {}, "source": path}
    written = state_document.get("timestamp")
    age = None
    if isinstance(written, (int, float)):
        # main.py writes an epoch timestamp; older files may not have one.
        age = max(0.0, now - float(written))
    elif isinstance(written, str):
        try:
            age = max(0.0, now - time.mktime(time.strptime(written[:19], "%Y-%m-%dT%H:%M:%S")))
        except ValueError:
            age = None
    spots = state_document.get("spots") or {}
    bays = {}
    for bay in BAYS:
        entry = spots.get(bay) or {}
        bays[bay] = {"state": entry.get("state", "UNKNOWN"),
                     "changed_ratio": entry.get("changed_ratio")}
    vehicle = state_document.get("vehicle") or {}
    return {
        "ok": True,
        "link": _link(True, age, VISION_STALE_S),
        "age_s": None if age is None else round(age, 2),
        "timestamp": written,
        "source": path,
        "bays": bays,
        "bays_occupied": sum(1 for b in spots.values() if (b or {}).get("state") == "OCCUPIED"),
        "bays_total": len(spots),
        "vehicle": {"x": vehicle.get("x"), "y": vehicle.get("y"),
                    "heading": vehicle.get("heading")},
    }


def mag_block(states, mapping, baseline, now, error=None):
    """The judge's current verdict per bay, with the numbers behind it."""
    if error or states is None:
        return {"ok": False, "link": "missing", "age_s": None, "error": error or "no data",
                "bays": {}}
    bays = {}
    ages = []
    for bay in BAYS:
        entry = states.get(bay) or {}
        channel = mapping.get(bay)
        bays[bay] = {
            "channel": channel,
            "state": entry.get("state", "UNKNOWN"),
            "delta_counts": None if entry.get("delta_counts") is None
            else round(entry["delta_counts"], 1),
            "delta_ut": None if entry.get("delta_ut") is None else round(entry["delta_ut"], 2),
            "detail": entry.get("reason", ""),
        }
    return {
        "ok": True,
        "link": "ok",
        "threshold_counts": None if baseline is None else baseline.threshold_counts,
        "hysteresis_counts": None if baseline is None else baseline.hysteresis_counts,
        "baseline": None if baseline is None else {
            "loaded": True,
            "calibrated_at": baseline.calibrated_at,
            "valid": sum(1 for e in baseline.bays.values() if e.get("valid")),
            "total": len(baseline.bays),
            "source": baseline.source,
        },
        "mapping": dict(mapping),
        "bays": bays,
        "error": None,
    }


def rfid_block(document, presence, now):
    """Tag set with presence states; the reader's own health decides whether they are trustworthy."""
    if document is None:
        return {"ok": False, "link": "missing", "age_s": None,
                "error": "no answer from /api/rfid", "tags": [], "count": 0}
    tags = []
    if presence is not None:
        for row in presence.rows(now):
            tags.append({"epc": row["epc"], "state": row["state"],
                         "silent_s": round(row["silent"], 2), "port": row["port"],
                         "rssi": row["rssi"], "reads": row["reads"]})
    return {
        "ok": bool(document.get("ok")) and (presence is not None and presence.reader_ok),
        "link": "ok" if (presence is not None and presence.reader_ok) else "missing",
        "error": None if (presence is not None and presence.reader_ok)
                 else (presence.health_note if presence is not None else "no presence tracker"),
        "polls": document.get("polls"),
        "frames": document.get("frames"),
        "notifications": document.get("notifications"),
        "timeouts": document.get("timeouts"),
        "count": sum(1 for t in tags if t["state"] == "PRESENT"),
        "known": len(tags),
        "tags": tags,
        "ports": document.get("ports") or [],
    }


def build(state_document, mag_states, mapping, baseline, rfid_document, presence, now=None,
          state_path=None):
    now = time.time() if now is None else now
    vision = vision_block(state_document, now, state_path)
    mag = mag_block(mag_states, mapping, baseline, now)
    rfid = rfid_block(rfid_document, presence, now)
    links = {"vision": vision["link"], "mag": mag["link"], "rfid": rfid["link"]}
    return {
        "schema": "overview/1",
        "t": now,
        "links": links,
        "all_present": all(v == "ok" for v in links.values()),
        "vision": vision,
        "mag": mag,
        "rfid": rfid,
        # The three questions, stated so the page cannot imply more than the data says.
        "questions": {
            "vision": "which bays have a car",
            "mag": "has this bay's field moved from its empty baseline",
            "rfid": "which tags are on the sandbox (no bay attribution)",
        },
    }
