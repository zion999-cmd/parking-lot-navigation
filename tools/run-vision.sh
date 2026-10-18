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
# The pause between attempts backs off: 2 s, 4, 8, ... up to 60 s, and resets after a run that
# lasted a minute. A fixed 2 s was measured doing real damage - one extra client on the camera
# starved the stream to 3.5 fps, main.py exited on "no frames for 20 s" every ~87 s, and this loop
# restarted it 12,907 times in a day, each restart churning the camera connection that was already
# the problem. The delay is capped so a genuine recovery is still noticed quickly, and the count is
# printed so a restart storm is visible in the log instead of being inferred from its size.

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

delay=2
attempt=0
while true; do
    attempt=$((attempt + 1))
    started=$(date +%s)
    "$PY_VISION" -u "$ROOT/vision/src/main.py" \
        --source "$SOURCE" --annotated-port "$ANNOTATED_PORT" &
    child=$!
    wait "$child"
    rc=$?
    child=""
    ran=$(( $(date +%s) - started ))
    # rc=2 is main.py's "could not open the camera": the board is still booting, or its single
    # stream slot is still held by the connection that just went away. Both are normal for a few
    # seconds and neither is a reason to wait a minute. Measured before this: the loop sat at
    # "after 6s ... restarting in 60 s" for attempts 97..106 while the board had been on the air
    # for nine seconds, which is where the "two minutes before the Wi-Fi settles" came from.
    if [ "$rc" = "2" ]; then
        delay=3
    else
        # A run of a minute or more means the camera was healthy: start the backoff from 2 s again.
        [ "$ran" -ge 60 ] && delay=2
    fi
    printf -- '--- main.py exited (rc=%s) at %s after %ss (attempt %d); restarting in %s s\n' \
           "$rc" "$(date '+%H:%M:%S')" "$ran" "$attempt" "$delay"
    sleep "$delay"
    if [ "$rc" != "2" ]; then
        delay=$(( delay * 2 ))
        [ "$delay" -gt 60 ] && delay=60
    fi
done
