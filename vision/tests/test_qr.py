"""Live QR API: the payload must name the serving PC and scan as that URL."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import qr_data                                                  # noqa: E402
from test_api import a_map, a_server, a_state                   # noqa: E402


def fetch(base: str, host: str):
    request = urllib.request.Request(base + "/api/qr", headers={"Host": host})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def scan(document: dict) -> str:
    size = document["size"]
    assert len(document["rows"]) == size
    quiet, scale = 4, 8
    image = np.full(((size + 2 * quiet) * scale,) * 2, 255, dtype=np.uint8)
    for y, row in enumerate(document["rows"]):
        assert len(row) == size and set(row) <= {"0", "1"}
        for x, bit in enumerate(row):
            if bit == "1":
                image[(y + quiet) * scale:(y + quiet + 1) * scale,
                      (x + quiet) * scale:(x + quiet + 1) * scale] = 0
    value, _, _ = cv2.QRCodeDetector().detectAndDecode(image)
    return value


def test_pc_a_and_pc_b_each_get_a_scannable_url_for_their_own_host() -> None:
    with a_server(a_map(), a_state({"38": "EMPTY"})) as (base, _, _, _):
        for host in ("192.168.3.71:8010", "192.168.3.72:8010"):
            status, document = fetch(base, host)
            assert status == 200, document
            assert document["schema"] == "parking_qr/1"
            assert document["url"] == f"http://{host}/go"
            assert scan(document) == document["url"], (host, document)


def test_loopback_is_rejected_because_a_phone_cannot_use_it() -> None:
    with a_server(a_map(), a_state({"38": "EMPTY"})) as (base, _, _, _):
        for host in ("localhost:8010", "127.0.0.1:8010", "0.0.0.0:8010"):
            status, document = fetch(base, host)
            assert status == 400, (host, document)
            assert "phone" in document["error"]


def test_bad_authorities_never_reach_the_encoder() -> None:
    for host in ("", "example.com/path", "user@example.com:8010", "[::1]:8010"):
        try:
            qr_data.navigation_url(host)
            raise AssertionError(host)
        except ValueError:
            pass


def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok    {name}")
    print(f"\n{len(tests)} checks, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
