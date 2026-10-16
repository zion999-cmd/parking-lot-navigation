#!/usr/bin/env bash
#
# Everything this project needs running on the PC, in one command.
#
#   ./start.sh                 start what is not already running
#   ./start.sh status          one screen: what is up, and will the strip work
#   ./start.sh stop            stop what this script started
#   ./start.sh restart
#   ./start.sh logs [name]     follow one log (serve, vision, beacon, ota)
#
# Options (for `start` / `restart`):
#   --source URL     camera stream; default: find the camera on the LAN
#   --no-ota         leave the OTA server alone (only needed for upgrades)
#   --no-beacon      leave the beacon alone (the device then uses the addresses
#                    compiled into it)
#
# Four background services, and what each one is for:
#
#   serve.py    :8010   the state and route documents, the web pages, /go.
#                       THE STRIP AND THE QR DEPEND ON THIS ONE.
#   main.py     :8020   vision: turns the camera stream into parking_state.json.
#                       Without it there is no fresh state, and the strip blanks
#                       itself on purpose (that is the rule, not a bug).
#   beacon.py   :9999   tells the device where this PC is, so moving to another
#                       Wi-Fi does not need a firmware rebuild. Optional by
#                       design: without it the device uses its configured URLs.
#   ota-server  :8000   serves firmware images. Only needed when upgrading.
#
# Logs and pid files land in run/ (gitignored). Running this script twice is safe:
# a service that is already up - including one somebody started by hand - is
# reported and adopted rather than started a second time.
#
# Written for the bash that ships with macOS (3.2): no associative arrays, no
# ${var,,}, no mapfile.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
RUN="$ROOT/run"

# main.py needs OpenCV/numpy, which live in the vision environment; everything else
# is standard library. Override with PY_VISION=... if the environment moves.
PY_VISION="${PY_VISION:-/usr/local/Caskroom/miniconda/base/envs/vision-dev/bin/python}"
PY_SYS="${PY_SYS:-python3}"

VISION_PORT=8010
ANNOTATED_PORT=8020
OTA_PORT=8000
BEACON_PORT=9999

# Set by the option parser; empty means "not given".
OPT_SOURCE=""
OPT_NO_OTA=0
OPT_NO_BEACON=0

# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

say()  { printf '%s\n' "$*"; }
info() { printf '  %s\n' "$*"; }
warn() { printf '  ! %s\n' "$*"; }

pidfile() { printf '%s/%s.pid' "$RUN" "$1"; }
logfile() { printf '%s/%s.log' "$RUN" "$1"; }

pid_of() {                                  # read a pid file, "" if absent/dead
    local f
    f="$(pidfile "$1")"
    [ -f "$f" ] || return 1
    local p
    p="$(cat "$f" 2>/dev/null || true)"
    [ -n "$p" ] || return 1
    kill -0 "$p" 2>/dev/null || return 1
    printf '%s' "$p"
}

# Find a pid for a service this script did not start. Two ways, because port and
# process name are not interchangeable: two of these hold a port, two do not.
# "Not running" is an answer, not a failure. With set -e and pipefail, a pgrep that
# finds nothing or an lsof that finds no listener would otherwise abort the script -
# which is exactly what a status command must never do.
pid_by_port() { lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null | head -1 || true; }
pid_by_cmd()  { pgrep -f "$1" 2>/dev/null | head -1 || true; }

adopt() {                                   # remember a pid we did not start
    local name="$1" pid="$2"
    printf '%s' "$pid" > "$(pidfile "$name")"
    info "$name: already running (pid $pid, not started by this script - adopted)"
}

stop_one() {
    local name="$1"
    local pid
    if ! pid="$(pid_of "$name")"; then
        # A pid file that exists but is dead is worth cleaning up and saying so.
        if [ -f "$(pidfile "$name")" ]; then
            rm -f "$(pidfile "$name")"
        fi
        return 0
    fi
    kill "$pid" 2>/dev/null || true
    local i=0
    while kill -0 "$pid" 2>/dev/null && [ "$i" -lt 25 ]; do
        sleep 0.2
        i=$((i + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
        warn "$name (pid $pid) did not stop; sending SIGKILL"
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$(pidfile "$name")"
    info "$name: stopped"
}

# Start a service in the background, with its own log, and wait until it is
# actually up before saying so - "started" and "working" are different things.
start_one() {
    local name="$1" ready="$2" check="$3"
    shift 3
    local log
    log="$(logfile "$name")"

    if pid_of "$name" >/dev/null; then
        info "$name: already running (pid $(pid_of "$name"))"
        return 0
    fi
    # Somebody else's copy? Adopt it rather than running a second one.
    local existing=""
    case "$check" in
        port:*) existing="$(pid_by_port "${check#port:}")" ;;
        cmd:*)  existing="$(pid_by_cmd  "${check#cmd:}")" ;;
    esac
    if [ -n "$existing" ]; then
        adopt "$name" "$existing"
        return 0
    fi

    say "  starting $name -> $log"
    # shellcheck disable=SC2086
    nohup "$@" >>"$log" 2>&1 &
    printf '%s' "$!" > "$(pidfile "$name")"

    if [ -n "$ready" ]; then
        local i=0
        while [ "$i" -lt 40 ]; do
            if eval "$ready" >/dev/null 2>&1; then
                info "$name: up (pid $(pid_of "$name"))"
                return 0
            fi
            if ! kill -0 "$(cat "$(pidfile "$name")")" 2>/dev/null; then
                warn "$name died while starting; last lines of $log:"
                tail -5 "$log" | sed 's/^/      /'
                rm -f "$(pidfile "$name")"
                return 1
            fi
            sleep 0.25
            i=$((i + 1))
        done
        warn "$name: started (pid $(pid_of "$name")) but did not answer yet - see $log"
        return 0
    fi
    info "$name: started (pid $(pid_of "$name"))"
}

port_answers() {                            # ready-check for the HTTP services
    # Silent, and never a failure: this is a question, and "no" is a valid answer.
    curl -s --max-time 2 -o /dev/null "http://127.0.0.1:$1/" 2>/dev/null || true
}

# --------------------------------------------------------------------------
# Finding the camera
# --------------------------------------------------------------------------

camera_base() { printf '%s' "${1%/stream}"; }

camera_ok() {                               # does this URL answer as our camera
    [ -n "${1:-}" ] || return 1
    curl -s --max-time 2 "$(camera_base "$1")/" 2>/dev/null | grep -q "Parking Lot Camera"
}

find_camera() {
    # A cached answer is tried first: one request, and the cache survives a camera
    # that changed address (it fails the check and probing happens again).
    if [ -f "$RUN/camera.txt" ]; then
        local cached
        cached="$(cat "$RUN/camera.txt" 2>/dev/null || true)"
        if camera_ok "$cached"; then
            printf '%s' "$cached"
            return 0
        fi
        rm -f "$RUN/camera.txt"
    fi

    # Probe this machine's /24 for the camera's own status page. Parallel, so the
    # whole sweep costs about a second; the page's title is what identifies it, not
    # the address, which is the point - the address is the thing that moves.
    local myip prefix tmp i
    myip="$(ipconfig getifaddr en0 2>/dev/null || true)"
    [ -n "$myip" ] || return 1
    prefix="${myip%.*}"
    tmp="$(mktemp -d)"
    for i in $(seq 1 254); do
        (
            if curl -s --max-time 1 "http://$prefix.$i/" 2>/dev/null | grep -q "Parking Lot Camera"; then
                printf '%s' "http://$prefix.$i/stream" > "$tmp/found"
            fi
        ) &
    done
    wait || true

    if [ -s "$tmp/found" ]; then
        local url
        url="$(head -1 "$tmp/found")"
        rm -rf "$tmp"
        printf '%s' "$url"
        return 0
    fi
    rm -rf "$tmp"
    return 1
}

# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

print_services() {
    say ""
    say "  service            port   what breaks without it"
    say "  serve.py           8010   the strip blanks, the QR cannot be followed"
    say "  main.py            8020   parking_state.json goes stale"
    say "  beacon.py          9999   the device falls back to its compiled-in address"
    say "  ota-server         8000   nothing, until you want to upgrade firmware"
    say ""
}

cmd_start() {
    mkdir -p "$RUN"
    say "starting stack (logs in $RUN/)"
    print_services

    if [ "$OPT_NO_OTA" = "0" ]; then
        start_one ota "$(port_answers "$OTA_PORT")" "port:$OTA_PORT" \
            "$PY_SYS" -u "$ROOT/tools/ota-server/server.py" --port "$OTA_PORT" || true
    fi

    start_one beacon "" "cmd:tools/beacon.py" \
        "$PY_SYS" -u "$ROOT/tools/beacon.py" --port "$BEACON_PORT" || true

    # serve.py must be on 0.0.0.0: the device and the phone are other machines.
    start_one serve "$(port_answers "$VISION_PORT")" "port:$VISION_PORT" \
        "$PY_SYS" -u "$ROOT/vision/src/serve.py" --host 0.0.0.0 --port "$VISION_PORT" || true

    local source="$OPT_SOURCE"
    if [ -z "$source" ]; then
        say "  finding the camera on the LAN ..."
        if ! source="$(find_camera)"; then
            warn "no camera found. Is the board powered and on this Wi-Fi?"
            warn "give it explicitly:  ./start.sh --source http://<ip>/stream"
        else
            info "camera: $source"
            printf '%s' "$source" > "$RUN/camera.txt"
        fi
    fi

    if [ -n "$source" ]; then
        if [ ! -x "$PY_VISION" ]; then
            warn "$PY_VISION not found - set PY_VISION to the python that has cv2"
        else
            # No ready-check: main.py has no port of its own unless the annotated
            # view is up, and "it started" is all this can honestly claim.
            start_one vision "" "cmd:src/main.py" \
                "$PY_VISION" -u "$ROOT/vision/src/main.py" \
                --source "$source" --annotated-port "$ANNOTATED_PORT" || true
        fi
    fi

    say ""
    cmd_status
    say ""
    say "  stop:   ./start.sh stop"
    say "  logs:   ./start.sh logs vision"
}

cmd_stop() {
    say "stopping stack"
    stop_one vision
    stop_one serve
    stop_one beacon
    stop_one ota
}

cmd_status() {
    mkdir -p "$RUN"
    say "stack status"

    local name pid state
    for name in serve vision beacon ota; do
        if pid="$(pid_of "$name")"; then
            state="up (pid $pid)"
        else
            case "$name" in
                serve)  pid="$(pid_by_port "$VISION_PORT")" ;;
                ota)    pid="$(pid_by_port "$OTA_PORT")" ;;
                vision) pid="$(pid_by_cmd "src/main.py")" ;;
                beacon) pid="$(pid_by_cmd "tools/beacon.py")" ;;
            esac
            if [ -n "$pid" ]; then
                state="up (pid $pid, not from this script)"
            else
                state="DOWN"
            fi
        fi
        printf '  %-8s %s\n' "$name" "$state"
    done

    say ""
    say "  what the strip needs"

    # The state document, its age, and whether its bays are the ones the strip is
    # mapped to. That last one is the failure that looks like "the strip is dead
    # and the log is clean": a vision process still running an older calibration.
    if ! port_answers "$VISION_PORT"; then
        warn "serve.py is not answering on :$VISION_PORT - the strip has no state source"
    else
        "$PY_SYS" - "$ROOT" "$VISION_PORT" <<'PY' || true
import json, sys, time, urllib.request, os
root, port = sys.argv[1], sys.argv[2]
try:
    doc = json.load(urllib.request.urlopen(
        f"http://127.0.0.1:{port}/out/parking_state.json", timeout=4))
except Exception as exc:
    print(f"  ! cannot read the state document: {exc}")
    raise SystemExit(0)
age = int(time.time()) - int(doc.get("timestamp", 0))
ids = set(doc.get("spots", {}))
print(f"  state document: {age} s old, {len(ids)} bays")
if age > 15:
    print("  ! stale: main.py is not writing (that is what blanks the strip)")
try:
    cfg = json.load(open(os.path.join(root, "vision/config/parking_spots.json")))
    want = {s["id"] for s in cfg["spots"]}
    missing = sorted(want - ids, key=str)[:6]
    if missing:
        print(f"  ! the document and the calibration disagree; missing {missing}")
        print("    (restart main.py after editing the map - the strip would go dark)")
    else:
        print("  bays match the calibration")
except Exception:
    pass
occupied = sum(1 for v in doc.get("spots", {}).values() if v.get("state") == "OCCUPIED")
print(f"  {occupied} occupied, {len(ids) - occupied} empty/unknown")
PY
    fi

    # The camera: cached address, and the firmware it is actually running.
    if [ -f "$RUN/camera.txt" ]; then
        local cam
        cam="$(cat "$RUN/camera.txt")"
        if camera_ok "$cam"; then
            local ver
            ver="$(curl -s --max-time 3 "$(camera_base "$cam")/" 2>/dev/null \
                   | sed -e 's/<[^>]*>/ /g' | tr -s ' \n' ' \n' \
                   | grep -i firmware | head -1 | sed 's/^ *//' || true)"
            info "camera: $cam  ($ver)"
        else
            warn "camera $cam does not answer (cached address; will re-probe on next start)"
        fi
    else
        warn "camera: unknown (pass --source, or let start find it)"
    fi

    if [ -n "$(pid_by_cmd "tools/beacon.py")" ]; then
        info "beacon: running - the device will follow this PC's address"
    else
        info "beacon: not running - the device uses its compiled-in address"
    fi
}

cmd_logs() {
    local name="${1:-serve}"
    local f
    f="$(logfile "$name")"
    if [ ! -f "$f" ]; then
        warn "no log for '$name' (looked for $f); names are: serve vision beacon ota"
        return 1
    fi
    tail -f "$f"
}

usage() {
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
}

# --------------------------------------------------------------------------
# Options and dispatch
# --------------------------------------------------------------------------

cmd=""
while [ $# -gt 0 ]; do
    case "$1" in
        start|stop|restart|status|logs) cmd="$1"; shift ;;
        --source) OPT_SOURCE="${2:?--source needs a URL}"; shift 2 ;;
        --no-ota) OPT_NO_OTA=1; shift ;;
        --no-beacon) OPT_NO_BEACON=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) warn "unknown argument: $1"; usage; exit 2 ;;
    esac
done
[ -n "$cmd" ] || cmd=start

case "$cmd" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    restart) cmd_stop; cmd_start ;;
    status)  cmd_status ;;
    logs)    cmd_logs "${1:-}" ;;
esac
