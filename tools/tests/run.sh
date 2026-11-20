#!/usr/bin/env bash
#
# Host tests: OTA versions, discovery URL rewriting, and QR matrix validation.
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

echo "compiling $MAIN/qr_matrix.c + qr_matrix_test.c with $CC"
"$CC" -std=c17 -Wall -Wextra -Werror -I"$MAIN" \
      "$MAIN/qr_matrix.c" "$HERE/qr_matrix_test.c" -o "$OUT/qr_matrix_test"

rc=0
"$OUT/version_test" || rc=1
"$OUT/urlhost_test" || rc=1
"$OUT/qr_matrix_test" || rc=1
python3 "$HERE/ota_manifest_test.py" || rc=1
rm -rf "$OUT"
exit $rc
