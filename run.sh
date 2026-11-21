#!/usr/bin/env bash
#
# Run main.py by hand, in the conda environment that has cv2 and numpy.
#
#   ./run.sh                              # default camera, windowed
#   ./run.sh --source http://192.168.3.118/stream
#   ./run.sh --headless --duration 10     # no window, print the per-bay summary and exit
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
    exec "$VISION_PYTHON" "$MAIN" --source "$SOURCE" "$@"
fi
exec "$VISION_PYTHON" "$MAIN" "$@"
