#!/usr/bin/env bash
#
# Run the vision viewer with the conda environment that has cv2 and numpy.
#
#   tools/run-vision-conda.sh                                  # default source
#   tools/run-vision-conda.sh --source http://192.168.3.118/stream
#   tools/run-vision-conda.sh --headless --duration 10 --mag-log data/mag_compare.csv
#
# Any arguments are passed straight through to vision/src/main.py, so the flags in
# `python vision/src/main.py --help` all work here.
#
# Why this exists: main.py imports numpy, so it has to run in the vision-dev environment -
# running it with the system python3 fails at import time. This script makes that choice for
# you instead of requiring the right interpreter path to be typed every time.
#
# Note: this runs main.py once and returns, like the program does. The restart loop the demo
# uses is tools/run-vision.sh; this is the "just run it and watch" entry point.

set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"

VISION_PYTHON="${PY_VISION:-/usr/local/Caskroom/miniconda/base/envs/vision-dev/bin/python}"

# Fall back to `conda run` if the environment lives somewhere else, and only then to any python3.
if [ ! -x "$VISION_PYTHON" ]; then
    if command -v conda >/dev/null 2>&1 && conda env list 2>/dev/null | grep -q '/vision-dev$'; then
        echo "note: $VISION_PYTHON not found, using 'conda run -n vision-dev'" >&2
        cd "$root"
        exec conda run -n vision-dev python vision/src/main.py "$@"
    fi
    if command -v python3 >/dev/null 2>&1; then
        echo "warning: vision-dev environment not found; falling back to $(command -v python3)" >&2
        echo "         if this fails on 'import numpy' or 'import cv2', point PY_VISION at the env" >&2
        VISION_PYTHON="$(command -v python3)"
    else
        echo "error: no python found" >&2
        echo "       set PY_VISION=/path/to/vision-dev/bin/python" >&2
        exit 1
    fi
fi

cd "$root"
exec "$VISION_PYTHON" vision/src/main.py "$@"
