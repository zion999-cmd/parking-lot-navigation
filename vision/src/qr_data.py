"""QR payloads and module matrices for the gate's navigation page.

The LCD consumes modules, not an image.  The same encoder is used by the HTTP
endpoint and tools/make-qr.py so an offline header and a live response cannot
silently use different QR settings.
"""

from __future__ import annotations

import ipaddress
import urllib.parse
from functools import lru_cache

try:
    import qrcode
    from qrcode.constants import ERROR_CORRECT_L
except ImportError:                     # a missing optional dependency must not take serve down
    qrcode = None
    ERROR_CORRECT_L = None


def available() -> bool:
    """False when the QR library is not installed in this interpreter."""
    return qrcode is not None

SCHEMA = "parking_qr/1"


def navigation_url(host: str) -> str:
    """Return the /go URL for a request Host that a phone can also reach.

    The ESP32 asks this endpoint using the address learned from the beacon.  That
    authority is better than a configured PC name: reaching the API has already
    proved that the same address works on this LAN.  Loopback authorities are
    rejected because a phone could not use them.
    """
    candidate = (host or "").strip()
    if not candidate or any(c in candidate for c in "\r\n/#?"):
        raise ValueError("invalid Host header")
    parsed = urllib.parse.urlsplit("//" + candidate)
    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError("invalid Host header") from exc
    if (not hostname or parsed.username is not None or parsed.password is not None or
            parsed.path or parsed.query or parsed.fragment):
        raise ValueError("invalid Host header")
    lowered = hostname.rstrip(".").lower()
    if lowered == "localhost" or lowered.endswith(".localhost"):
        raise ValueError("loopback Host is not reachable by a phone")
    try:
        address = ipaddress.ip_address(lowered.strip("[]"))
    except ValueError:
        address = None
    if address is not None and (address.is_loopback or address.is_unspecified):
        raise ValueError("loopback Host is not reachable by a phone")
    # The service runs on 8010; an omitted port is accepted for test clients but
    # never produces an implicit port-80 URL.
    authority = hostname if port is None else f"{hostname}:{port}"
    if ":" in hostname and not hostname.startswith("["):
        authority = f"[{hostname}]" + (f":{port}" if port is not None else "")
    if port is None:
        authority += ":8010"
    return f"http://{authority}/go"


@lru_cache(maxsize=8)
def make_matrix(payload: str) -> tuple[tuple[bool, ...], ...]:
    """Encode with ECC L and no border; display.c supplies the quiet zone."""
    qr = qrcode.QRCode(error_correction=ERROR_CORRECT_L, border=0)
    qr.add_data(payload)
    qr.make(fit=True)
    return tuple(tuple(bool(cell) for cell in row) for row in qr.get_matrix())


def document(payload: str) -> dict:
    matrix = make_matrix(payload)
    return {
        "schema": SCHEMA,
        "url": payload,
        "size": len(matrix),
        "error_correction": "L",
        "rows": ["".join("1" if cell else "0" for cell in row) for row in matrix],
    }
