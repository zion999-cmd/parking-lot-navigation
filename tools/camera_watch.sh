#!/usr/bin/env bash
# Camera watcher: keep the vision loop pointed at the camera, wherever it is.
#
# The board gets its address by DHCP, so 192.168.3.118 is only true until the router
# hands out a different one. This watcher scans the local /24 for the board's status
# page (identified by its title, "Parking Lot Camera" - not by its address) and
# restarts run-vision.sh whenever the address changes.
#
# Scanning is deliberately serial-ish: 254 parallel curls saturate the link and the
# router stops answering for everybody, including the camera we are looking for.
#
# An address can also be injected from outside:
#
#   echo -n "http://192.168.3.118/stream" > /tmp/pln-camera-hint
#
# The hint is read every cycle and wins over the sweep. That matters when the sweep
# cannot see the board but something else can - a different machine on the same network,
# or a process that was told the address by hand. The watcher does not care where the
# address came from; it only cares that it changed.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

ROOT=/Users/lancy/WorkBuddy/2026-11-14-10-09-25/parking-lot-navigation
PY=/usr/local/Caskroom/miniconda/base/envs/vision-dev/bin/python
LOG=/tmp/pln-camera-watch.log
HINT=/tmp/pln-camera-hint
RESULT=/tmp/pln-camera-result
FALLBACK="${PLN_CAMERA_URL:-http://192.168.3.118/stream}"

export PY_VISION="$PY"
export ANNOTATED_PORT=8020
export RAW_PORT=8021

say() { printf '%s %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

current_source=""
vision_pid=""

start_vision() {
    local src="$1"
    say "vision -> $src"
    "$ROOT/tools/run-vision.sh" "$src" >> "$LOG" 2>&1 &
    vision_pid=$!
}

stop_vision() {
    [ -n "$vision_pid" ] || return 0
    say "stopping vision (pid $vision_pid)"
    kill "$vision_pid" 2>/dev/null
    pkill -P "$vision_pid" 2>/dev/null
    vision_pid=""
}

# One sweep of the local /24. Prints the stream URL if the board answers, else nothing.
find_camera() {
    local prefix ip
    prefix="$(ipconfig getifaddr en0 2>/dev/null | cut -d. -f1-3)"
    [ -n "$prefix" ] || return 1
    for ip in $(seq 1 254); do
        if curl -s --noproxy '*' --max-time 2 "http://$prefix.$ip/" 2>/dev/null \
             | grep -q "Parking Lot Camera"; then
            printf 'http://%s.%s/stream' "$prefix" "$ip"
            return 0
        fi
        sleep 0.03
    done
    return 1
}

# The address to use right now: an injected hint wins, then the last sweep result,
# then the fallback. The hint is how this works when the sweep cannot reach the
# board but something outside can.
current_address() {
    local hint
    hint="$(cat "$HINT" 2>/dev/null || true)"
    case "$hint" in
        http*) printf '%s' "$hint"; return 0 ;;
    esac
    local swept
    swept="$(cat "$RESULT" 2>/dev/null || true)"
    case "$swept" in
        http*) printf '%s' "$swept"; return 0 ;;
    esac
    printf '%s' "$FALLBACK"
}

say "camera watcher started (fallback $FALLBACK)"

# The sweep runs in the background: it is serial and can take minutes when the camera
# is absent, and every second spent sweeping is a second the loop is not being
# health-checked. It writes its answer to $RESULT; the main loop only reads it.
sweep_loop() {
    while true; do
        local found
        found="$(find_camera)" || found=""
        printf '%s' "$found" > "$RESULT.tmp"
        mv "$RESULT.tmp" "$RESULT"
        sleep 10
    done
}
sweep_loop &
SWEEP_PID=$!

while true; do
    # Liveness first, and every iteration: a loop killed by hand has to come back
    # without waiting for the next sweep to finish.
    if [ -n "$vision_pid" ] && ! kill -0 "$vision_pid" 2>/dev/null; then
        say "vision loop is not running; restarting against ${current_source:-$FALLBACK}"
        vision_pid=""
        start_vision "${current_source:-$FALLBACK}"
    fi

    sleep 5
    found="$(current_address)"

    if [ "$found" != "$current_source" ]; then
        say "camera address: ${current_source:-none} -> $found"
        current_source="$found"
        printf '%s' "$found" > "$ROOT/run/camera.txt"
        stop_vision
        start_vision "$found"
    fi
done
