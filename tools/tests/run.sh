#!/usr/bin/env bash
#
# Host tests for this project. Currently just the OTA version rule.
#
#   tools/tests/run.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
MAIN="$HERE/../../firmware/esp32-s3-cam/main"
OUT="$(mktemp -d)"

CC="${CC:-cc}"

echo "compiling $MAIN/version.c + version_test.c with $CC"
"$CC" -std=c17 -Wall -Wextra -Werror -I"$MAIN" \
      "$MAIN/version.c" "$HERE/version_test.c" -o "$OUT/version_test"

"$OUT/version_test"
rc=$?
rm -rf "$OUT"
exit $rc
