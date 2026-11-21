#!/usr/bin/env bash
#
# Run main.py by hand, in the conda environment that has cv2 and numpy.
#
#   ./run.sh                              # default camera, windowed
#   ./run.sh --source http://192.168.3.118/stream
#   ./run.sh --headless --duration 10     # no window, print the per-bay summary and exit
#   ./run.sh                              # stops the background stack if it is running, runs main.py
#                                         # in the foreground, restarts the stack when it exits
#   ./run.sh --no-take-over               # refuse instead of taking over (old behaviour)
#   ./run.sh --help                       # every main.py flag
#
# This is the *manual* entry point: it runs main.py in the foreground, shows the OpenCV window, and
# returns when main.py does. The system stack - serve.py, the restart supervisor, the relays, the
# pid files, the logs, beacon and the OTA server - is ./start.sh. Do not run both: main.py writes
# vision/out/parking_state.json, and two writers make the tracked vehicle jump on every page, so
# main.py refuses to start while another live copy owns that file.
#
#   ./run.sh          <- one main.py, foreground, you watch it
#   ./start.sh        <- the whole system, supervised, logs in run/

set -euo pipefail

VISION_PYTHON="${PY_VISION:-/usr/local/Caskroom/miniconda/base/envs/vision-dev/bin/python}"
PROJECT="/Users/bx/Esp32/parking-lot-navigation"
MAIN="$PROJECT/vision/src/main.py"

if [ ! -x "$VISION_PYTHON" ]; then
    echo "error: interpreter not found: $VISION_PYTHON" >&2
    echo "       override with PY_VISION=/path/to/vision-dev/bin/python" >&2
    exit 1
fi
if [ ! -f "$MAIN" ]; then
    echo "error: main.py not found: $MAIN" >&2
    exit 1
fi

# --take-over: the stack holds the camera and writes the same state file, so it has to step aside
# first. This is the one-command version of "stop the stack, run main.py yourself, start it again" -
# ./start.sh edit does the same thing for the bay editor.
# Take over by default. The guard in main.py exists because two writers make the vehicle jump, but
# refusing to start while the stack runs meant the answer to "run main.py" was always "no" - asked
# four times in a row. Taking over is what the person typing this wants: the stack steps aside, and
# comes back when main.py exits.
TAKE_OVER=1
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --take-over)    TAKE_OVER=1 ;;
        --no-take-over) TAKE_OVER=0 ;;
        *)              ARGS+=("$arg") ;;
    esac
done
set -- ${ARGS+"${ARGS[@]}"}

if [ "$TAKE_OVER" = "1" ]; then
    # Only the background VISION has to step aside - it holds the camera and writes the state file.
    # Stopping the whole stack also stopped serve.py, so /web/board.html could not be opened while a
    # manual main.py was running, and the page kept showing the state from before the stop, which is
    # why the hero looked wrong there while the local window was right.
    #
    # main.py's own guard already does exactly this (supervisor first, then the children), so there is
    # nothing to stop here: it takes over on startup and says so.
    echo "note: the background vision will step aside automatically; serve/beacon/ota keep running"
    restore() {
        echo
        echo "main.py finished - bringing the background vision back"
        "$PROJECT/start.sh" >/dev/null 2>&1 || true
    }
    trap restore EXIT INT TERM
fi

# Default camera: whatever start.sh last found, else the board's usual address. An explicit
# --source on the command line wins, because it comes after these.
SOURCE="${PARKING_SOURCE:-}"
if [ -z "$SOURCE" ] && [ -f "$PROJECT/run/camera.txt" ]; then
    SOURCE="$(cat "$PROJECT/run/camera.txt")"
fi
if [ -z "$SOURCE" ]; then
    SOURCE="http://192.168.3.118/stream"
fi

for arg in "$@"; do
    if [ "$arg" = "--source" ] || [ "${arg#--source=}" != "$arg" ]; then
        SOURCE=""                      # already given; do not add a second one
        break
    fi
done

cd "$PROJECT"
if [ -n "$SOURCE" ]; then
    set -- --source "$SOURCE" "$@"
fi

if [ "$TAKE_OVER" = "1" ]; then
    # Not exec: replacing the shell would discard the trap, and the stack would stay down after
    # main.py exits - which is exactly what happened on the first version of this.
    "$VISION_PYTHON" "$MAIN" "$@"
else
    exec "$VISION_PYTHON" "$MAIN" "$@"
fi
