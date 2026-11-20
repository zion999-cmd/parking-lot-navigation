#!/usr/bin/env bash
# Run the vision loop against the camera, in a shell that can actually reach it.
#
# The camera watcher (tools/camera_watch.sh) is meant to do this, but it runs inside
# the assistant's sandbox, and the sandbox has its own network namespace: it can ping
# the router and it can see the PC's own services, but it cannot reach 192.168.3.118.
# The board is there - the browser and a terminal on the same machine can both open
# its stream - so this script is the way to run the loop from a real shell.
#
#   ./tools/run-vision-local.sh                       # use the address in run/camera.txt
#   ./tools/run-vision-local.sh http://<ip>/stream    # or give one
#
# It clears the proxy variables first. The system proxy answers 502 for every LAN
# address, and a 502 from a proxy looks exactly like a broken camera in the log.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY_VISION:-/usr/local/Caskroom/miniconda/base/envs/vision-dev/bin/python}"

SOURCE="${1:-}"
if [ -z "$SOURCE" ]; then
    if [ -f "$ROOT/run/camera.txt" ]; then
        SOURCE="$(cat "$ROOT/run/camera.txt")"
    else
        SOURCE="http://192.168.3.118/stream"
    fi
fi

export PY_VISION="$PY"
export ANNOTATED_PORT="${ANNOTATED_PORT:-8020}"
export RAW_PORT="${RAW_PORT:-8021}"

echo "camera: $SOURCE"
echo "python: $PY"
echo ""

exec "$ROOT/tools/run-vision.sh" "$SOURCE"
