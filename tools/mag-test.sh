#!/usr/bin/env bash
# One command for the magnetometer test page (P0005.2).
#
#   ./tools/mag-test.sh                        # find the board, then open a browser
#   ./tools/mag-test.sh --host 192.168.3.118   # when discovery cannot find it
#
# Does not need the vision stack running: it talks to the board's /api/mag directly and imports the
# detector from vision/src. The board address comes from run/camera.txt (the cache start.sh already
# keeps) unless one is given.

set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"

PORT="${MAG_TEST_PORT:-8040}"
URL="http://127.0.0.1:${PORT}/"

# The vision environment is where numpy/cv2 live; this tool only needs the standard library, but
# using the same interpreter keeps one thing fewer to explain.
PY="${PY_VISION:-}"
if [ -z "$PY" ]; then
    if [ -x "/usr/local/Caskroom/miniconda/base/envs/vision-dev/bin/python" ]; then
        PY="/usr/local/Caskroom/miniconda/base/envs/vision-dev/bin/python"
    else
        PY="$(command -v python3)"
    fi
fi

if [ -z "$PY" ]; then
    echo "error: no python3 found" >&2
    exit 1
fi

# Already running? Just bring the page up again.
if curl -sf --max-time 2 "${URL}api/state" >/dev/null 2>&1; then
    echo "already running: ${URL}"
    if [ "${1:-}" != "--no-open" ]; then open "$URL"; fi
    exit 0
fi

# The server opens the browser itself, so pass --no-open through and let it decide.
# -u so the banner appears immediately even when the output is piped to a log.
exec "$PY" -u "${root}/tools/mag_test_server.py" --port "$PORT" "$@"
