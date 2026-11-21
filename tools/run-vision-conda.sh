#!/usr/bin/env bash
#
# Start the stack with the conda vision-dev environment. Thin wrapper around ./start.sh.
#
#   tools/run-vision-conda.sh
#   tools/run-vision-conda.sh --source http://192.168.3.118/stream
#   tools/run-vision-conda.sh status
#   tools/run-vision-conda.sh stop --all
#
# All arguments go to start.sh, which already knows how to find the camera, start the supervisor,
# open the annotated and raw relays, keep the pid files and write the logs.
#
# It used to run main.py directly, in parallel with whatever start.sh had already started, with
# different arguments. That produced two writers of vision/out/parking_state.json and a vehicle that
# jumped on every page - main.py now refuses to be that second writer, and this wrapper no longer
# offers a way to become one.

set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"

export PY_VISION="${PY_VISION:-/usr/local/Caskroom/miniconda/base/envs/vision-dev/bin/python}"
exec "$root/start.sh" "$@"
