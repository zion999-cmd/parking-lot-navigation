#!/usr/bin/env bash
#
# Start the stack. This is a thin wrapper around ./start.sh and nothing else.
#
#   ./run.sh                       # whole stack: serve + vision (+ relays 8020/8021) + beacon + ota
#   ./run.sh --source http://x/stream
#   ./run.sh stop --all
#   ./run.sh status
#
# It used to `exec main.py` itself, with different arguments from the ones start.sh uses: no
# --annotated-port/--raw-port, no log file, no restart loop, and no record in the stack's pid files.
# That is how two main.py processes ended up writing vision/out/parking_state.json at the same time
# and the tracked vehicle appeared to jump between two positions on board.html. One entry point, one
# set of arguments, one owner of the state file.

set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
exec "$here/start.sh" "$@"
