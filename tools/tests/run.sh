#!/usr/bin/env bash
#
# Host tests for this project: the OTA version rule, and the URL host rewrite
# that dynamic server discovery depends on.
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

echo "compiling $MAIN/urlhost.c + urlhost_test.c with $CC"
"$CC" -std=c17 -Wall -Wextra -Werror -I"$MAIN" \
      "$MAIN/urlhost.c" "$HERE/urlhost_test.c" -o "$OUT/urlhost_test"

rc=0
"$OUT/version_test" || rc=1
"$OUT/urlhost_test" || rc=1
rm -rf "$OUT"
exit $rc
