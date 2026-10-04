#!/usr/bin/env bash
#
# Publish a built firmware image as an OTA release.
#
#   ./publish.sh 0.2.1
#   ./publish.sh 0.2.1 path/to/parking-lot-cam.bin
#   ./publish.sh --sha256 <64-hex> 0.2.1        # deliberately wrong digest
#
# Writes:
#   firmware/<version>/parking-lot-cam.bin
#   manifest.json
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_BIN="$HERE/../../firmware/esp32-s3-cam/build/parking-lot-cam.bin"

SHA_OVERRIDE=""
if [ "${1:-}" = "--sha256" ]; then
    SHA_OVERRIDE="${2:?--sha256 needs a value}"
    shift 2
fi

VERSION="${1:?usage: $0 [--sha256 HEX] <version> [path/to/parking-lot-cam.bin]}"
BIN="${2:-$DEFAULT_BIN}"

if [ ! -f "$BIN" ]; then
    echo "error: firmware image not found: $BIN" >&2
    echo "       build it first:  cd firmware/esp32-s3-cam && idf.py build" >&2
    exit 1
fi

case "$VERSION" in
    *[!0-9.]*) echo "error: version must look like major.minor.patch" >&2; exit 1 ;;
esac

DEST_DIR="$HERE/firmware/$VERSION"
DEST="$DEST_DIR/parking-lot-cam.bin"
mkdir -p "$DEST_DIR"
cp "$BIN" "$DEST"

SIZE=$(wc -c < "$DEST" | tr -d ' ')
SHA=$(shasum -a 256 "$DEST" | awk '{print $1}')
[ -n "$SHA_OVERRIDE" ] && SHA="$SHA_OVERRIDE"

cat > "$HERE/manifest.json" <<EOF
{
  "project": "parking-lot-cam",
  "version": "$VERSION",
  "firmware_url": "/firmware/$VERSION/parking-lot-cam.bin",
  "sha256": "$SHA",
  "size": $SIZE
}
EOF

echo "published $VERSION"
echo "  binary   : $DEST ($SIZE bytes)"
echo "  sha256   : $SHA"
[ -n "$SHA_OVERRIDE" ] && echo "  NOTE: digest was overridden - the device is expected to REJECT this image."
echo "  manifest : $HERE/manifest.json"
