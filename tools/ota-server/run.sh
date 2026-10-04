#!/usr/bin/env bash
#
# Start the thin OTA server.
#
#   ./run.sh              # port 8000, all interfaces
#   ./run.sh --port 9000
#
set -euo pipefail
cd "$(dirname "$0")"
# -u keeps the startup banner (with the LAN URLs) visible when output is piped
# or redirected into a log.
exec python3 -u server.py "$@"
