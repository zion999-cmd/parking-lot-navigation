#!/usr/bin/env python3
"""Add the demonstration publisher HMAC to an OTA manifest."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys

DEFAULT_KEY_FILE = os.path.expanduser("~/.config/parking-lot/ota.key")


def key_from_file(path: str | None = None) -> bytes:
    filename = path or os.environ.get("PARKING_OTA_KEY_FILE", DEFAULT_KEY_FILE)
    with open(filename, "rb") as fh:
        key = fh.read().strip()
    if len(key) < 32:
        raise ValueError(f"OTA key in {filename} must contain at least 32 bytes")
    return key


def canonical(document: dict) -> bytes:
    """Canonical fields mirrored by ota.c: project/version/url/sha/size, LF after each."""
    return (f"{document['project']}\n{document['version']}\n{document['firmware_url']}\n"
            f"{document['sha256']}\n{int(document.get('size', 0))}\n").encode("utf-8")


def sign(document: dict, key: bytes) -> dict:
    signed = dict(document)
    signed["signature_algorithm"] = "HMAC-SHA256"
    signed["signature"] = hmac.new(key, canonical(document), hashlib.sha256).hexdigest()
    return signed


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "manifest.json")
    key_path = sys.argv[2] if len(sys.argv) > 2 else None
    with open(path, encoding="utf-8") as fh:
        document = json.load(fh)
    document = sign(document, key_from_file(key_path))
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as fh:
        json.dump(document, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(temporary, path)
    print(f"  signature   : HMAC-SHA256 ({document['signature'][:16]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
