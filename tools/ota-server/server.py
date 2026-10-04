#!/usr/bin/env python3
"""
Thin OTA server for the Parking Lot Navigation camera node.

It serves exactly two things, and nothing else:

    GET /manifest.json
    GET /firmware/<version>/parking-lot-cam.bin

No accounts, no login, no database, no device management, no dashboard, no
telemetry, no cloud. Python standard library only - `python3 server.py` is the
whole deployment story.

The device polls /manifest.json, compares the version, and (if newer) downloads
the binary and verifies its SHA-256 before committing it to the spare OTA slot.
"""

import argparse
import functools
import http.server
import os
import socket
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


class OtaHandler(http.server.SimpleHTTPRequestHandler):
    """Static file server rooted at this directory.

    SimpleHTTPRequestHandler already does GET + Range + directory listings, which
    is all this service needs. Two customisations: HTTP/1.1 (what the device's
    esp_http_client speaks) and cache handling, because a cached manifest would
    keep a device stuck on a stale version.
    """

    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def end_headers(self):
        if self.path.split("?")[0] in ("/manifest.json", "/manifest.json/"):
            self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("  %s  %s\n" % (self.address_string(), fmt % args))


class ReusableServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def lan_addresses():
    """Best-effort list of this machine's LAN addresses, for the startup banner."""
    found = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(info[4][0])
    except OSError:
        pass
    # The address the OS would use to reach the internet.
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        found.add(probe.getsockname()[0])
        probe.close()
    except OSError:
        pass
    return sorted(a for a in found if not a.startswith("127."))


def main():
    parser = argparse.ArgumentParser(description="Thin OTA server (manifest + firmware).")
    parser.add_argument("--host", default="0.0.0.0", help="bind address (default: all)")
    parser.add_argument("--port", type=int, default=8000, help="TCP port (default: 8000)")
    args = parser.parse_args()

    manifest = os.path.join(ROOT, "manifest.json")
    if not os.path.exists(manifest):
        print("WARNING: %s does not exist yet." % manifest, file=sys.stderr)
        print("         Run ./publish.sh <version> first, or the device will get a 404.",
              file=sys.stderr)

    handler = functools.partial(OtaHandler)
    with ReusableServer((args.host, args.port), handler) as httpd:
        print("Thin OTA server serving %s" % ROOT)
        print("  manifest : /manifest.json")
        print("  firmware : /firmware/<version>/parking-lot-cam.bin")
        print("  listening: http://%s:%d/" % (args.host, args.port))
        for addr in lan_addresses():
            print("  device   : http://%s:%d/manifest.json" % (addr, args.port))
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
