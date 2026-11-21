#!/usr/bin/env bash
#
# Everything this project needs running on the PC, in one command.
#
#   ./start.sh                 start what is not already running
#   ./start.sh status          one screen: what is up, and will the strip work
#   ./start.sh stop            stop what this script started
#   ./start.sh restart
#   ./start.sh edit [--source URL]   open the bay editor (stops the stack, restarts it on exit)
#   ./start.sh logs [name]     follow one log (serve, vision, beacon, ota, label)
#   ./start.sh label           start the labelling page (:8030) as a managed service
#   ./start.sh beacon          start the discovery beacon (UDP 9999); also started by default
#   ./start.sh ota             start the OTA server (:8000); also started by default
#   ./start.sh stop --all      stop the stack AND every test/experiment process of this
#                              project (tools/hero_lab, vision/tests, the labeller, ...)
#
# Options (for `start` / `restart`):
#   --source URL     camera stream; default: find the camera on the LAN
#   --no-ota         do not start the OTA server (it starts by default)
#   --ota            compatibility flag; OTA already starts by default
#   --no-beacon      do not start the beacon (the device then uses its compiled-in addresses)
#   --beacon         compatibility flag; the beacon already starts by default
#
# Four background services, and what each one is for:
#
#   serve.py    :8010   the state and route documents, the web pages, /go.
#                       THE STRIP AND THE QR DEPEND ON THIS ONE.
#   main.py     :8020   vision: turns the camera stream into parking_state.json.
#                       Without it there is no fresh state, and the strip blanks
#                       itself on purpose (that is the rule, not a bug).
#   beacon.py   :9999   tells the device where this PC is, so the LED state and
#                       route URLs follow a new computer or Wi-Fi automatically.
#   ota-server  :8000   serves firmware images. A published newer image can update
#                       the board automatically; use --no-ota when that is unwanted.
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

# The normal stack is serve + vision + beacon + OTA. label is always opt-in.
# --no-ota / --no-beacon are the explicit escape hatches.
OPT_SOURCE=""
OPT_OTA=1
OPT_BEACON=1
OPT_NO_OTA=0
OPT_NO_BEACON=0

# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

say()  { printf '%s\n' "$*"; }
info() { printf '  %s\n' "$*"; }
warn() { printf '  ! %s\n' "$*"; }

pidfile() { printf '%s/%s.pid' "$RUN" "$1"; }
startedfile() { printf '%s/%s.started' "$RUN" "$1"; }
seconds_since_started() {                   # "" if unknown
    local f
    f="$(startedfile "$1")"
    [ -f "$f" ] || return 0
    local t
    t="$(cat "$f" 2>/dev/null || true)"
    [ -n "$t" ] || return 0
    printf '%s' "$(( $(date +%s) - t ))"
}
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

# Every entry point into this project is supposed to be this script. In practice there are others -
# a labelling server, a one-off eval, a main.py left behind when its supervisor was killed - and they
# all fight over the same camera, whose frame budget is shared between clients. Measured: one stray
# client took the vision stream from ~12 fps to 3.5, which made main.py exit and the supervisor
# restart it 12,907 times in a day.
#
# So: a way to list them, and a way to kill them.
#
# The match is on the script path, and a pid is only killed if it is actually an interpreter
# (python/sh/bash). Without that second test, `pgrep -f tools/beacon.py` also matches the editor that
# has that file open, and `stop --all` would close the editor.
SELF_PIDS=" $$ $PPID "
MANAGED=""

is_interpreter() {
    case "$(ps -o comm= -p "$1" 2>/dev/null | sed 's#.*/##')" in
        python*|sh|bash|env|dash|perl) return 0 ;;
        *) return 1 ;;
    esac
}

# The pids this script is supposed to be running, plus their children. A managed service is not a
# stray: run-vision.sh is in vision.pid and main.py is its child, and listing either of them under
# "not managed" makes the report untrustworthy exactly where it matters.
managed_pids() {
    local f pid
    for f in "$RUN"/*.pid; do
        [ -f "$f" ] || continue
        pid="$(cat "$f" 2>/dev/null)"
        [ -n "$pid" ] && printf '%s\n' "$pid"
    done
}

is_descendant_of_managed() {
    local pid="$1" parent
    while parent="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')"; do
        [ -z "$parent" ] && return 1
        [ "$parent" -le 1 ] && return 1
        case " $MANAGED " in *" $parent "*) return 0 ;; esac
        pid="$parent"
    done
    return 1
}

# pids running one of these scripts, never this script or its parent
stray_pids() {
    local pattern pid
    MANAGED="$(managed_pids | tr '\n' ' ')"
    for pattern in "$@"; do
        for pid in $(pgrep -f "$pattern" 2>/dev/null); do
            case "$SELF_PIDS" in *" $pid "*) continue ;; esac
            case " $MANAGED " in *" $pid "*) continue ;; esac
            is_descendant_of_managed "$pid" && continue
            is_interpreter "$pid" && printf '%s\n' "$pid"
        done
    done | sort -u
}

describe_pid() {
    ps -o pid=,command= -p "$1" 2>/dev/null | cut -c1-100 | sed 's/^ *//'
}

# TERM, wait, then KILL. Says what it did, because "stopped" without a pid is not evidence.
kill_pids() {
    local pid i=0 killed=0
    for pid in "$@"; do
        kill "$pid" 2>/dev/null || continue
        killed=$((killed + 1))
    done
    while [ "$i" -lt 25 ]; do
        local alive=0
        for pid in "$@"; do kill -0 "$pid" 2>/dev/null && alive=1; done
        [ "$alive" -eq 0 ] && break
        sleep 0.2
        i=$((i + 1))
    done
    for pid in "$@"; do
        if kill -0 "$pid" 2>/dev/null; then
            warn "pid $pid ignored SIGTERM; sending SIGKILL"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    printf '%s' "$killed"
}

# The managed services, plus whatever else is running their code. Children before wrappers: killing
# run-vision.sh alone leaves main.py alive and still holding the camera, which is how a "restart"
# ends up with two clients instead of none.
# Both spellings of every script: run-vision.sh starts main.py by its *relative* path
# ("src/main.py"), and the sweep only matched "vision/src/main.py", so an orphan from 01:29 was
# still holding a second camera connection hours later - visible as Clients 2 on the board's status
# page and as two entries in lsof. Second camera clients are exactly what this project cannot have.
# Extra beacon targets, because a plain 255.255.255.255 broadcast is not enough on every network.
# Phone hotspots in particular isolate clients from each other, so the broadcast never arrives and the
# board keeps using the address compiled into the firmware: the QR fetch then times out against a host
# that is not there, every three seconds, forever. A directed broadcast for this subnet, plus a unicast
# to the board itself, gets through where the global broadcast does not - and a unicast is exactly what
# discovery needs, since the device takes the PC's address from the packet's source.
beacon_targets() {
    local pc ip
    pc="$(ipconfig getifaddr en0 2>/dev/null || true)"
    if [ -n "$pc" ]; then printf -- '--to %s.255 ' "${pc%.*}"; fi
    if [ -f "$RUN/camera.txt" ]; then
        ip="$(sed -e 's#^http://##' -e 's#^https://##' -e 's#/.*$##' -e 's#:.*$##' \
                  "$RUN/camera.txt" 2>/dev/null || true)"
        if [ -n "$ip" ]; then printf -- '--to %s ' "$ip"; fi
    fi
}

SERVICE_SCRIPTS="tools/run-vision.sh vision/src/main.py src/main.py vision/src/serve.py src/serve.py tools/beacon.py tools/ota-server"
TEST_SCRIPTS="tools/hero_lab tools/label_server.py vision/tests vision/src/calibration.py vision/src/emit_map.py vision/src/plan.py src/calibration.py src/emit_map.py src/plan.py tools/serial_capture.py tools/watch_device.py"

# What a service's process command line must contain before this script will adopt or
# kill it. Ports and pid files are not proof of identity: a recycled pid, or another
# program that happens to hold :8000, used to be adopted and then killed by `stop`.
script_for() {
    case "$1" in
        serve)  printf 'vision/src/serve.py' ;;
        vision) printf 'run-vision.sh' ;;
        beacon) printf 'tools/beacon.py' ;;
        ota)    printf 'tools/ota-server/server.py' ;;
        label)  printf 'tools/label_server.py' ;;
        *)      printf '' ;;
    esac
}

pid_matches() {                              # pid, expected script fragment
    local pid="$1" expect="$2" cmd comm
    [ -n "$pid" ] && [ -n "$expect" ] || return 1
    cmd="$(ps -o command= -p "$pid" 2>/dev/null || true)"
    [ -n "$cmd" ] || return 1
    case "$cmd" in
        *"$expect"*) ;;
        *) return 1 ;;
    esac
    # The path alone is not proof: an editor with the file open also satisfies it.
    comm="$(ps -o comm= -p "$pid" 2>/dev/null | sed 's#.*/##')"
        # Case-insensitive on purpose: the framework Python reports itself as "Python" with a
        # capital P, so `python*` alone rejected services this same script had started via
        # `start.sh ota` (system interpreter) - the stack then said "not from this script"
        # and refused to stop them.
    case "$comm" in
        [Pp]ython*|sh|bash|zsh|env) return 0 ;;
        *) return 1 ;;
    esac
}

adopt() {                                   # remember a pid we did not start
    local name="$1" pid="$2" expect
    expect="$(script_for "$name")"
    if ! pid_matches "$pid" "$expect"; then
        warn "$name: pid $pid is not this project's $name - not adopting it"
        warn "      command: $(ps -o command= -p "$pid" 2>/dev/null || echo unknown)"
        return 1
    fi
    printf '%s' "$pid" > "$(pidfile "$name")"
    info "$name: already running (pid $pid, not started by this script - adopted)"
}

stop_one() {
    local name="$1" expect
    local pid
    if ! pid="$(pid_of "$name")"; then
        # A pid file that exists but is dead is worth cleaning up and saying so.
        if [ -f "$(pidfile "$name")" ]; then
            rm -f "$(pidfile "$name")"
        fi
        return 0
    fi
    expect="$(script_for "$name")"
    if ! pid_matches "$pid" "$expect"; then
        warn "$name: pid $pid no longer looks like $name - removing the stale pid file"
        rm -f "$(pidfile "$name")"
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
        adopt "$name" "$existing" || return 1
        return 0
    fi

    say "  starting $name -> $log"
    # The log is appended across restarts, so anything older than this start is history.
    # Reading the tail without that anchor once reported the previous run's "Address already
    # in use" as if it were happening now.
    local log_from=0
    [ -f "$log" ] && log_from="$(wc -l < "$log" 2>/dev/null | tr -d ' ')"
    # shellcheck disable=SC2086
    nohup "$@" >>"$log" 2>&1 &
    printf '%s' "$!" > "$(pidfile "$name")"
    date +%s > "$(startedfile "$name")"

    if [ -n "$ready" ]; then
        local i=0
        while [ "$i" -lt 160 ]; do   # up to 40 s: main.py has to open the stream and write once
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
            # The supervisor survives main.py exiting, so "pid is alive" is not progress. If the log
            # already says the camera could not be opened, waiting the full forty seconds only hides
            # the reason behind a spinner.
            # A port still held by the previous instance, or a camera that does not answer: both are
            # terminal for this start, and neither gets better by waiting another 250 ms.
            local fresh=""
            if [ -f "$log" ]; then
                fresh="$(tail -n "+$((log_from + 1))" "$log" 2>/dev/null | tail -20)"
            fi
            if printf '%s' "$fresh" | grep -q "Address already in use"; then
                warn "$name cannot start: $(printf '%s' "$fresh" | grep -m1 "Address already in use" | cut -c1-100)"
                warn "  a previous instance still holds its port - stop the stack and wait a moment:"
                warn "  ./start.sh stop --all; sleep 3; ./start.sh"
                return 1
            fi
            if [ "$name" = "vision" ] && printf '%s' "$fresh" | grep -q "could not open"; then
                warn "vision cannot start: $(printf '%s' "$fresh" | grep -m1 "could not open" | cut -c1-110)"
                warn "  the camera in --source is not answering - check it, or pass the right one:"
                warn "  ./start.sh --source http://<board-ip>/stream"
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

# Is the state document being written right now? This is the only honest readiness
# signal the vision loop has: it has no port of its own, and "the process started" is
# not the same as "the strip has data" - that difference is what made a freshly
# started stack look broken.
state_fresh() {
    # curl and arithmetic, not a python3 -c: this runs inside the readiness loop every 250 ms, and
    # a process whose name changes the terminal title twice a second for forty seconds looks like a
    # script stuck in a loop, which is exactly how it was reported.
    local doc ts now
    doc="$(curl -sf --max-time 3 "http://127.0.0.1:8010/out/parking_state.json" 2>/dev/null || true)"
    [ -n "$doc" ] || return 1
    ts="$(printf '%s' "$doc" | sed -n 's/.*"timestamp"[[:space:]]*:[[:space:]]*\([0-9]*\).*/\1/p' | head -1)"
    [ -n "$ts" ] || return 1
    now="$(date +%s)"
    [ $((now - ts)) -lt 5 ]
}

port_answers() {                            # ready-check for the HTTP services
    # Silent, but the exit status is the answer: -f turns an HTTP error into failure too.
    # Without -f this returned 0 for a closed port as well, so every readiness check and
    # the "serve.py is not answering" warning were dead.
    curl -sf --max-time 2 -o /dev/null "http://127.0.0.1:$1/" 2>/dev/null
}

# --------------------------------------------------------------------------
# Finding the camera
# --------------------------------------------------------------------------

camera_base() { printf '%s' "${1%/stream}"; }

# Does this URL answer as our camera?
#
# The timeout is a parameter, and generous by default, because of what happens when
# the camera is already streaming: the MJPEG stream saturates the link (a ping to the
# board measures 70-95 ms instead of the 2-5 ms it measures when idle), and a status
# request that normally takes 80 ms can occasionally need seconds. Reporting "does not
# answer" for a camera that is streaming perfectly is worse than reporting nothing,
# because it sends someone to debug a camera that is working.
camera_ok() {
    local url="${1:-}" timeout="${2:-4}"
    [ -n "$url" ] || return 1
    curl -s --max-time "$timeout" "$(camera_base "$url")/" 2>/dev/null \
        | grep -q "Parking Lot Camera"
}

find_camera() {
    # A cached answer is tried first: one request, and the cache survives a camera
    # that changed address (it fails the check and probing happens again).
    if [ -f "$RUN/camera.txt" ]; then
        local cached
        cached="$(cat "$RUN/camera.txt" 2>/dev/null || true)"
        # Short timeout first because this is the common path, then a long one: a
        # streaming camera can miss the short one and still be perfectly fine.
        if camera_ok "$cached" 3 || camera_ok "$cached" 8; then
            printf '%s' "$cached"
            return 0
        fi
        warn "cached camera ${cached:0:48} did not answer; looking again" >&2
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
            if curl -s --max-time 3 "http://$prefix.$i/" 2>/dev/null | grep -q "Parking Lot Camera"; then
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

# One command to fix the bays.
#
# The editor is the same main.py, but it needs a window to take keys ('e' to move
# bays, 'r' for a reference). The background vision holds the camera's single
# stream slot, so it has to step aside first - otherwise the editor would be the
# second viewer and the board would (correctly) refuse it with 409.
cmd_edit() {
    local source="$OPT_SOURCE"
    if [ -z "$source" ]; then
        say "  finding the camera on the LAN ..."
        source="$(find_camera)" || source=""
        if [ -z "$source" ]; then
            warn "no camera found. give it explicitly:  ./start.sh edit --source http://<ip>/stream"
            return 1
        fi
    fi

    cmd_stop 0 >/dev/null 2>&1 || true

    say ""
    say "  ---------------------------------------------------------------"
    say "   editor window opening - it is the only viewer, so nothing is"
    say "   being split. Keys:  e = move bays   r = capture reference"
    say "                       a = auto-detect  q = quit"
    say "   When you quit, the background stack restarts by itself."
    say "  ---------------------------------------------------------------"
    say ""
    "$PY_VISION" -u "$ROOT/vision/src/main.py" --source "$source" || true

    say ""
    say "  editor closed - restarting the background stack"
    cmd_start
}

cmd_start() {
    mkdir -p "$RUN"
    say "starting stack (logs in $RUN/)"
    print_services

    # OTA is part of the normal stack. Publishing a newer image while it is running
    # can update the board automatically; --no-ota is the explicit way to prevent that.
    if [ "$OPT_OTA" = "1" ] && [ "$OPT_NO_OTA" = "0" ]; then
        start_one ota "port_answers $OTA_PORT" "port:$OTA_PORT" \
            "$PY_SYS" -u "$ROOT/tools/ota-server/server.py" --port "$OTA_PORT" || true
    fi

    # The beacon is required when the PC address can change: the LED state and
    # route URLs follow its source address. --no-beacon restores the old behaviour
    # of using only the addresses compiled into the firmware.
    if [ "$OPT_BEACON" = "1" ] && [ "$OPT_NO_BEACON" = "0" ]; then
        # shellcheck disable=SC2046  (targets never contain spaces)
        start_one beacon "" "cmd:tools/beacon.py" \
            "$PY_SYS" -u "$ROOT/tools/beacon.py" --port "$BEACON_PORT" $(beacon_targets) || true
    fi

    # serve.py must be on 0.0.0.0: the device and the phone are other machines.
        # serve.py imports the contract module, which imports road_network, which imports
        # numpy - so it needs the vision environment, not the system interpreter. It ran on
        # $PY_SYS until someone started the stack from a shell whose python3 had no numpy,
        # and then serve died with ModuleNotFoundError and the strip lost its state source.
    start_one serve "port_answers $VISION_PORT" "port:$VISION_PORT" \
        "$PY_VISION" -u "$ROOT/vision/src/serve.py" --host 0.0.0.0 --port "$VISION_PORT" || true

    local source="$OPT_SOURCE"
    if [ -z "$source" ]; then
        say "  finding the camera on the LAN ..."
        # find_camera writes exactly one URL to stdout; anything else in there is a bug, and it
        # has happened - a warning was printed to stdout and captured, so the "URL" handed to
        # main.py was the warning's text, which failed as an HTTP URL and took the service down
        # into the restart loop. Validate rather than trust.
        source="$(find_camera)" || source=""
        case "$source" in
            http*) ;;
            "")    ;;
            *)     warn "camera resolution returned something that is not a URL: $source" >&2
                   source="" ;;
        esac
        if [ -z "$source" ]; then
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
            # Ready means "the state document went fresh", not "the process exists".
            # If it never does, the reason is in its own log (usually the camera
            # stream), so that is what gets printed.
            #
            # Wrapped in run-vision.sh because main.py finishes: the camera rebooting
            # ends the stream, main.py returns after printing its summary, and the
            # state document stops being written. It is the only service here that
            # exits on purpose, so it is the only one that needs restarting.
            start_one vision "state_fresh" "cmd:run-vision.sh" \
                "$ROOT/tools/run-vision.sh" "$source" || true
        fi
    fi

    say ""
    cmd_status
    say ""
    say "  stop:   ./start.sh stop"
    say "  logs:   ./start.sh logs vision"
}

cmd_stop() {
    local all="${1:-0}"
    say "stopping stack"
    stop_one vision
    stop_one serve
    stop_one beacon
    stop_one ota
    stop_one label

    # The pid files cover the processes this script started; these cover everything else running the
    # same code - an adopted copy, a second main.py from a killed supervisor, a service started by
    # hand. Without this, "stopped" means "stopped the ones I knew about".
    local strays
    # shellcheck disable=SC2086
    strays="$(stray_pids $SERVICE_SCRIPTS)"
    if [ -n "$strays" ]; then
        say "  also stopping:"
        for pid in $strays; do printf '    %s\n' "$(describe_pid "$pid")"; done
        kill_pids $strays >/dev/null
    fi

    if [ "$all" = "1" ]; then
        # shellcheck disable=SC2086
        strays="$(stray_pids $TEST_SCRIPTS)"
        if [ -n "$strays" ]; then
            say "  stopping test and experiment processes (--all):"
            for pid in $strays; do printf '    %s\n' "$(describe_pid "$pid")"; done
            kill_pids $strays >/dev/null
        fi
        rm -f "$RUN"/label.pid
    fi

    local left
    # shellcheck disable=SC2086
    left="$(stray_pids $SERVICE_SCRIPTS)"
    if [ -n "$left" ]; then
        warn "still running:"
        for pid in $left; do printf '    %s\n' "$(describe_pid "$pid")"; done
        return 1
    fi
    say "  nothing of this project is left running"
}

# The labelling page is an entry point like any other, so it is started and stopped here rather than
# with a remembered python command. It needs the camera to itself: see the warning in the tool itself.
# Start the OTA server on its own. It serves /manifest.json and the published binaries; the board
# polls it in a loop, so while this is running a published build reaches the hardware without anyone
# asking again. That is useful during an upgrade and a hazard the rest of the time.
# The discovery beacon: how the device learns this PC's address when the network moves underneath
# it. Useful, and currently the thing the board does not survive indefinitely, so it is started
# deliberately and stopped when the demo does not need it.
cmd_beacon() {
    if pid="$(pid_of beacon)"; then
        info "beacon: already running (pid $pid)"
        return 0
    fi
    start_one beacon "" "cmd:tools/beacon.py" \
        "$PY_SYS" -u "$ROOT/tools/beacon.py" --port "$BEACON_PORT" $(beacon_targets) || true
    say "  broadcasting every 2 s on UDP $BEACON_PORT; watch the board: ./start.sh status"
}

cmd_ota() {
    if pid="$(pid_of ota)"; then
        info "ota: already running (pid $pid)"
        return 0
    fi
    start_one ota "port_answers $OTA_PORT" "port:$OTA_PORT" \
        "$PY_SYS" -u "$ROOT/tools/ota-server/server.py" --port "$OTA_PORT" || true
    local mf="$ROOT/tools/ota-server/manifest.json"
    if [ -f "$mf" ]; then
        say "  publishing: $(sed -n 's/.*"version": *"\([^"]*\)".*/\1/p' "$mf" | head -1)"
    fi
    say "  the board polls this; stop it with ./start.sh stop when the upgrade is done"
}

cmd_label() {
    local port="${LABEL_PORT:-8030}"
    if pid="$(pid_of label)"; then
        info "label: already running (pid $pid) - http://127.0.0.1:$port"
        return 0
    fi
    # The labeller reads the PC-side RAW relay (:8021), not the board. The board has one
    # frame queue and allows one viewer (vision); :8021 is vision re-serving the unannotated
    # picture, so opening the labeller costs the board nothing and needs nothing stopped.
    local lsrc="${OPT_SOURCE:-http://127.0.0.1:${RAW_PORT:-8021}/stream}"
    if ! port_answers "${RAW_PORT:-8021}"; then
        warn "no raw relay on :${RAW_PORT:-8021} - start the stack first: ./start.sh start"
    fi
    start_one label "" "cmd:tools/label_server.py" \
        "$PY_VISION" "$ROOT/tools/label_server.py" --port "$port" --source "$lsrc" || true
    say "  open: http://127.0.0.1:$port"
}

cmd_status() {
    mkdir -p "$RUN"
    say "stack status"

    local name pid state
    for name in serve vision beacon ota label; do
        if pid="$(pid_of "$name")"; then
            state="up (pid $pid)"
        else
            case "$name" in
                serve)  pid="$(pid_by_port "$VISION_PORT")" ;;
                ota)    pid="$(pid_by_port "$OTA_PORT")" ;;
                vision) pid="$(pid_by_cmd "run-vision.sh")" ;;
                beacon) pid="$(pid_by_cmd "tools/beacon.py")" ;;
            esac
            if [ -n "$pid" ]; then
                state="up (pid $pid, not from this script)"
            else
                state="DOWN"
            fi
        fi
        printf '  %-8s %s\n' "$name" "$state"
        # A bare "DOWN" sends people to look at the wrong thing. A supervisor that is alive but whose
        # main.py keeps exiting shows as DOWN here, and the reason is in its log - usually the camera
        # being unreachable.
        if [ "$name" = "vision" ] && [ "$state" = "DOWN" ] && [ -f "$RUN/vision.log" ]; then
            local why tries
            # The reason first, the exit code only as a fallback: the last line of the log is always
            # "rc=2", which says nothing about why.
            why="$(grep -aoE 'error: could not open[^\n]*|could not read video stream[^\n]*' "$RUN/vision.log" 2>/dev/null | tail -1)"
            [ -n "$why" ] || why="$(grep -aoE 'rc=[0-9]+' "$RUN/vision.log" 2>/dev/null | tail -1)"
            [ -n "$why" ] && printf '  %-8s   why: %s\n' "" "${why:0:96}"
            tries="$(grep -ac 'main.py exited' "$RUN/vision.log" 2>/dev/null || echo 0)"
            if [ "${tries:-0}" -gt 0 ]; then
                printf '  %-8s   main.py has exited and restarted %s time(s) - see run/vision.log\n' "" "$tries"
            fi
        fi
    done

    say ""
    # Not managed, but still running this project's code - and therefore still a client on the
    # camera, which is the resource every one of these competes for.
    local strays
    # shellcheck disable=SC2086
    strays="$(stray_pids $SERVICE_SCRIPTS $TEST_SCRIPTS)"
    if [ -n "$strays" ]; then
        say "  not managed by this script (they count as camera clients):"
        for pid in $strays; do printf '    %s\n' "$(describe_pid "$pid")"; done
        say "    ./start.sh stop --all   stops these too"
    fi

    say "  what the strip needs"

    # The state document, its age, and whether its bays are the ones the strip is
    # mapped to. That last one is the failure that looks like "the strip is dead
    # and the log is clean": a vision process still running an older calibration.
    local v_age
    v_age="$(seconds_since_started vision)"

    # Code or calibration edited after the process started: the change is not running. This has
    # already fooled us twice - a restart at 06:44 with an edit at 06:53 - and a stale process
    # looks exactly like a logic fault, which is the expensive kind of wrong. One line of output
    # is cheaper than remembering.
    if [ -n "${v_age:-}" ]; then
        local newest started
        newest="$(find "$ROOT/vision/src" "$ROOT/vision/config" \( -name '*.py' -o -name '*.json' \) -print 2>/dev/null \
                 | xargs stat -f %m 2>/dev/null | sort -rn | head -1)"
        started="$(( $(date +%s) - v_age ))"
        if [ -n "$newest" ] && [ "$newest" -gt "$started" ]; then
            warn "vision code or calibration is newer than the running vision process"
            say "     → the fix is not running:  ./start.sh restart"
        fi
    fi

    if ! port_answers "$VISION_PORT"; then
        warn "serve.py is not answering on :$VISION_PORT - the strip has no state source"
    else
        "$PY_SYS" - "$ROOT" "$VISION_PORT" "${v_age:-}" <<'PY' || true
import json, sys, time, urllib.request, os
root, port = sys.argv[1], sys.argv[2]
just_started = sys.argv[3] if len(sys.argv) > 3 else ""
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
    # A stack that was just started is not broken, it is early: main.py needs to open
    # the stream and write once, which takes seconds. Saying "not writing" at that
    # moment is how a working stack gets diagnosed as a broken one.
    if just_started and int(just_started) < 60:
        print(f"  ... waiting for the first state document "
              f"(vision started {just_started} s ago)")
    else:
        print("  ! stale: nothing is writing parking_state.json")
        print("    (that blanks the strip by design - check: ./start.sh logs vision)")
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
# Counted separately, because EMPTY and UNKNOWN are not the same thing on the strip:
# EMPTY is green, UNKNOWN is dark. Lumping them together hides the difference between
# "the lot is empty" and "the detector has not seen these bays yet".
states = {}
for v in doc.get("spots", {}).values():
    states[v.get("state", "?")] = states.get(v.get("state", "?"), 0) + 1
print("  {occ} occupied, {emp} empty, {unk} unknown".format(
    occ=states.get("OCCUPIED", 0), emp=states.get("EMPTY", 0), unk=states.get("UNKNOWN", 0)))
if states.get("UNKNOWN", 0) and age < 20:
    print("    (unknown right after a start is normal: no frame has been seen yet,"
          " and the strip shows those dark)")
elif states.get("UNKNOWN", 0):
    print("    ! bays have been unknown for a while: the detector is not seeing them")
PY
    fi

    # The camera: cached address, and the firmware it is actually running.
    if [ -f "$RUN/camera.txt" ]; then
        local cam
        cam="$(cat "$RUN/camera.txt" 2>/dev/null)"
        case "$cam" in http*) ;; *) cam="" ;; esac
        if camera_ok "$cam" 4; then
            local ver
            ver="$(curl -s --max-time 3 "$(camera_base "$cam")/" 2>/dev/null \
                   | sed -e 's/<[^>]*>/ /g' | tr -s ' \n' ' \n' \
                   | grep -i firmware | head -1 | sed 's/^ *//' || true)"
            info "camera: $cam  ($ver)"
        else
            warn "camera $cam did not answer in 4 s"
            warn "  (a busy camera can be slow to answer while streaming; ./start.sh status to retry)"
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
        warn "no log for '$name' (looked for $f); names are: serve vision beacon ota label"
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
LOG_NAME=""
while [ $# -gt 0 ]; do
    case "$1" in
        start|stop|restart|status|label|edit) cmd="$1"; shift ;;
        # `logs` takes an optional service name. It used to consume the word and leave
        # nothing for cmd_logs, so `./start.sh logs vision` was rejected as unknown.
        logs)
            cmd="logs"
            shift
            if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
                LOG_NAME="$1"
                shift
            fi
            ;;
        --all|-all) OPT_ALL=1; shift ;;
        --ota) OPT_OTA=1; shift ;;
        --beacon) OPT_BEACON=1; shift ;;
        beacon) cmd="beacon"; shift ;;
        ota) cmd="ota"; shift ;;
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
    stop)    cmd_stop "${OPT_ALL:-0}" ;;
    restart) cmd_stop "${OPT_ALL:-0}"; cmd_start ;;
    edit)    cmd_edit ;;
    label)   cmd_label ;;
    ota)     cmd_ota ;;
    beacon)  cmd_beacon ;;
    status)  cmd_status ;;
    logs)    cmd_logs "${LOG_NAME:-}" ;;
esac
