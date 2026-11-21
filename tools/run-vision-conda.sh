#!/usr/bin/env bash
#
# Run the vision viewer with the conda vision-dev environment.
#
#   tools/run-vision-conda.sh
#   tools/run-vision-conda.sh --source http://192.168.3.118/stream
#   tools/run-vision-conda.sh --headless --duration 10 --mag-log data/mag_compare.csv
#
# All arguments go straight to vision/src/main.py, so every flag in
# `tools/run-vision-conda.sh --help` works.
#
# Paths are fixed on purpose - this is a personal launcher for one machine, and deriving them
# produced a script that ran /Users/bx/Esp32/vision/src/main.py when a copy was placed elsewhere.

set -euo pipefail

VISION_PYTHON="/usr/local/Caskroom/miniconda/base/envs/vision-dev/bin/python"
MAIN="/Users/bx/Esp32/parking-lot-navigation/vision/src/main.py"
PROJECT="/Users/bx/Esp32/parking-lot-navigation"

if [ ! -x "$VISION_PYTHON" ]; then
    echo "error: interpreter not found: $VISION_PYTHON" >&2
    echo "       (override with PY_VISION=/path/to/vision-dev/bin/python)" >&2
    exit 1
fi
VISION_PYTHON="${PY_VISION:-$VISION_PYTHON}"

if [ ! -f "$MAIN" ]; then
    echo "error: main.py not found: $MAIN" >&2
    exit 1
fi

# main.py resolves its own paths relative to the project, so run from there.
cd "$PROJECT"
exec "$VISION_PYTHON" "$MAIN" "$@"
