#!/usr/bin/env python3
"""Generate the fixed test QR matrix that firmware/esp32-s3-cam/main/qr_code.h holds.

    python3 tools/make-qr.py "http://<PC-IP>:8010/go" > firmware/esp32-s3-cam/main/qr_code.h

Why a baked matrix instead of an encoder on the device: the payload changes about
once per installation, so an encoder would be flash spent on a decision already made -
and one more thing to get wrong. The cost is that changing the URL is a rebuild plus an
OTA, which is measured in minutes and happens once. The
panel draws the modules as filled squares, so what the firmware needs is the
module grid, which is exactly what this prints.

Regenerate whenever the payload changes. The README section on the QR screen says
so, and the generated header says so in its own comment.

Kept in tools/ next to make-aruco-marker.py, for the same reason: the thing that
produces a constant belongs with the constant it produces.
"""

import sys

import qrcode
from qrcode.constants import ERROR_CORRECT_L


def main() -> int:
    # The payload is required rather than defaulted: a default here would be a second
    # copy of the PC's address, and the copy in the generated header is the one the
    # firmware actually uses. Two copies is how one of them goes stale.
    if len(sys.argv) < 2:
        sys.stderr.write(
            'usage: make-qr.py "<url or text>"\n'
            '   eg: make-qr.py "http://192.168.0.10:8010/go"\n'
            '       > firmware/esp32-s3-cam/main/qr_code.h\n')
        return 2
    payload = sys.argv[1]

    # Error correction L, and the reason is module size rather than robustness: on a
    # 240 px panel the scale is what decides whether a phone can read it. L keeps a
    # 16 character payload at version 1 (21 modules), which allows 8 px per module
    # plus a full 4 module quiet zone. M would push it to version 2 (25 modules) and
    # 7 px - a smaller, denser code on the same screen.
    qr = qrcode.QRCode(error_correction=ERROR_CORRECT_L, border=0)
    qr.add_data(payload)
    qr.make(fit=True)

    matrix = qr.get_matrix()
    n = len(matrix)
    assert all(len(row) == n for row in matrix), "matrix is not square"

    rows = []
    for row in matrix:
        bits = 0
        for x, dark in enumerate(row):
            if dark:
                bits |= 1 << x
        rows.append(bits)

    per_row = (n + 7) // 8
    # The payload is pasted into a C string literal and a comment. Escape it so a quote,
    # backslash, newline or "*/" in the URL cannot produce invalid (or injected) source.
    c_payload = (payload.replace("\\", "\\\\").replace('"', '\\"')
                 .replace("\n", "\\n").replace("\r", "\\r"))
    shown_payload = payload.replace("*/", "* /").replace("\n", " ").replace("\r", " ")
    print("/*")
    print(" * QR code for the gate panel - GENERATED, do not edit by hand.")
    print(" *")
    print(f' *   payload : "{shown_payload}" ({len(payload)} characters)')
    print(f" *   version : {(n - 17) // 4}  ({n} x {n} modules)")
    print(" *   ECC     : L")
    print(" *")
    print(" * Regenerate with:")
    print(f' *     python3 tools/make-qr.py "{shown_payload}" \\')
    print(" *         > firmware/esp32-s3-cam/main/qr_code.h")
    print(" *")
    print(" * One bit per module, bit 0 = leftmost column of that row, which is what")
    print(" * display.c walks when it paints the modules.")
    print(" */")
    print("#pragma once")
    print()
    print(f"#define QR_PAYLOAD \"{c_payload}\"")
    print(f"#define QR_SIZE    {n}")
    print(f"#define QR_ROW_BYTES {per_row}")
    print()
    print("static const uint8_t qr_rows[QR_SIZE][QR_ROW_BYTES] = {")
    for y, bits in enumerate(rows):
        packed = [(bits >> (8 * i)) & 0xFF for i in range(per_row)]
        body = ", ".join(f"0x{b:02X}" for b in packed)
        print(f"    {{ {body} }},   /* row {y:2d} */")
    print("};")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
