#!/usr/bin/env bash
#
# Run the vision loop, and start it again if it exits.
#
#   tools/run-vision.sh http://<camera>/stream
#
# Why this exists: main.py is a *run*, not a daemon. When the MJPEG stream ends - which
# happens every time the board reboots, and the board reboots whenever someone moves a
# wire - it prints a per-bay summary and returns. That is right for a one-off analysis and
# wrong for a demo, where the consequence is that parking_state.json stops being written
# and the strip has no fresh state to show.
#
# So the thing that produces the state is the one thing start.sh keeps alive. Everything
# else there is a service that runs until it is stopped; this is a program that finishes.
#
# A 2 s pause between attempts so a camera that is down does not turn into a spin loop
# filling the log; the reason each attempt failed is already in the log above it.

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY_VISION="${PY_VISION:-/usr/local/Caskroom/miniconda/base/envs/vision-dev/bin/python}"
ANNOTATED_PORT="${ANNOTATED_PORT:-8020}"
SOURCE="${1:?usage: run-vision.sh http://<camera>/stream}"

child=""
stop() {
    [ -n "$child" ] && kill "$child" 2>/dev/null
    exit 0
}
trap stop TERM INT

while true; do
    "$PY_VISION" -u "$ROOT/vision/src/main.py" \
        --source "$SOURCE" --annotated-port "$ANNOTATED_PORT" &
    child=$!
    wait "$child"
    rc=$?
    child=""
    printf -- '--- main.py exited (rc=%s) at %s; restarting in 2 s\n' \
           "$rc" "$(date '+%H:%M:%S')"
    sleep 2
done
