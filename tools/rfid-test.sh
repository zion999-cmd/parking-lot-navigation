#!/usr/bin/env bash
#
# RFID presence test: which tags are on the sandbox right now.
#
#   ./tools/rfid-test.sh                        # board from run/camera.txt, or 192.168.3.118
#   ./tools/rfid-test.sh --host 192.168.3.118
#   ./tools/rfid-test.sh --absent-after 3       # seconds of silence before a tag counts as gone
#
# Terminal only on purpose: no page, no DOM, nothing that stops you selecting and copying what you
# see. On a terminal it redraws in place; piped to a file it appends, so the log stays copyable.
#
# If the reader stops answering, the screen says UNKNOWN rather than pretending every tag left.
# The known cause of that state is a sticky multi-antenna inventory command left in the module by an
# earlier experiment: it survives an ESP32 reboot and only clears when the reader is power-cycled.

set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"

PY="${PY_VISION:-}"
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
    if [ -x "/usr/local/Caskroom/miniconda/base/envs/vision-dev/bin/python" ]; then
        PY="/usr/local/Caskroom/miniconda/base/envs/vision-dev/bin/python"
    else
        PY="$(command -v python3)"
    fi
fi

HOST=""
for arg in "$@"; do
    case "$arg" in
        --host) HOST="next" ;;
        --host=*) HOST="${arg#--host=}" ;;
        *) if [ "$HOST" = "next" ]; then HOST="$arg"; fi ;;
    esac
done

if [ -z "$HOST" ] || [ "$HOST" = "next" ]; then
    HOST=""
    if [ -f "$root/run/camera.txt" ]; then
        HOST="$(sed -e 's#^http://##' -e 's#/.*$##' "$root/run/camera.txt" 2>/dev/null || true)"
    fi
    if [ -z "$HOST" ]; then HOST="192.168.3.118"; fi
    exec "$PY" "$root/tools/rfid/rfid_presence.py" --host "$HOST" "$@"
fi

exec "$PY" "$root/tools/rfid/rfid_presence.py" "$@"
