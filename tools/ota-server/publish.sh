#!/usr/bin/env bash
#
# Publish a built firmware image as an OTA release.
#
#   ./publish.sh 0.2.3
#   ./publish.sh 0.2.3 path/to/parking-lot-cam.bin
#   ./publish.sh --sha256 <64-hex> 0.2.3        # deliberately wrong digest
#   ./publish.sh --allow-version-mismatch 0.9.9 path/to/bin
#
# Writes:
#   firmware/<version>/parking-lot-cam.bin
#   manifest.json
#
# The version you type is checked against the version actually embedded in the
# binary. Publishing 0.2.3 from a 0.2.2 image is how a manifest ends up
# advertising a digest that the served file does not have, which the device then
# rejects as a SHA256 mismatch. Refusing to do that is cheaper than debugging it.
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_BIN="$HERE/../../firmware/esp32-s3-cam/build/parking-lot-cam.bin"

SHA_OVERRIDE=""
ALLOW_MISMATCH=""
while [ $# -gt 0 ]; do
    case "$1" in
        --sha256)
            SHA_OVERRIDE="${2:?--sha256 needs a value}"; shift 2 ;;
        --allow-version-mismatch)
            ALLOW_MISMATCH=1; shift ;;
        *)
            break ;;
    esac
done

VERSION="${1:?usage: $0 [--sha256 HEX] [--allow-version-mismatch] <version> [path/to/parking-lot-cam.bin]}"
BIN="${2:-$DEFAULT_BIN}"

if [ ! -f "$BIN" ]; then
    echo "error: firmware image not found: $BIN" >&2
    echo "       build it first:  cd firmware/esp32-s3-cam && idf.py build" >&2
    exit 1
fi

case "$VERSION" in
    *[!0-9.]*) echo "error: version must look like major.minor.patch" >&2; exit 1 ;;
esac

# The app version lives in esp_app_desc_t: 32 bytes at offset 0x30 of the image.
BIN_VERSION="$(python3 -c '
import sys
d = open(sys.argv[1], "rb").read(0x50)
if len(d) < 0x50:
    sys.exit("image too short")
print(d[0x30:0x50].split(b"\x00")[0].decode("ascii", "replace"))
' "$BIN")"

if [ "$BIN_VERSION" != "$VERSION" ]; then
    if [ -z "$ALLOW_MISMATCH" ]; then
        echo "error: version mismatch" >&2
        echo "       you asked to publish : $VERSION" >&2
        echo "       binary actually is   : $BIN_VERSION  ($BIN)" >&2
        echo "       Set PROJECT_VER in firmware/esp32-s3-cam/CMakeLists.txt and rebuild," >&2
        echo "       or pass --allow-version-mismatch if you really mean it." >&2
        exit 1
    fi
    echo "warning: publishing $VERSION from a $BIN_VERSION binary (--allow-version-mismatch)" >&2
fi

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
echo "  binary      : $DEST ($SIZE bytes, embedded version $BIN_VERSION)"
echo "  sha256      : $SHA"
[ -n "$SHA_OVERRIDE" ] && echo "  NOTE: digest was overridden - the device is expected to REJECT this image."
echo "  manifest    : $HERE/manifest.json"
