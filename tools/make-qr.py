#!/usr/bin/env python3
"""Offline QR matrix/header generator using the production encoder.

The running firmware gets its matrix from GET /api/qr and no longer compiles a
fixed header. This tool remains useful for diagnostics and host-side tests.

    python3 tools/make-qr.py "http://<PC-IP>:8010/go" > /tmp/qr_code.h
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "vision", "src"))
from qr_data import make_matrix                                  # noqa: E402


def main() -> int:
    # The payload is required rather than defaulted: a default here would be a second
    # copy of the PC's address, and the copy in the generated header is the one the
    # firmware actually uses. Two copies is how one of them goes stale.
    if len(sys.argv) < 2:
        sys.stderr.write(
            'usage: make-qr.py "<url or text>"\n'
            '   eg: make-qr.py "http://192.168.0.10:8010/go"\n'
            '       > /tmp/qr_code.h\n')
        return 2
    payload = sys.argv[1]

    # ECC L and no border are shared with the live endpoint; display.c adds the
    # four-module quiet zone when it paints the matrix.
    matrix = make_matrix(payload)
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
    print(" *         > /tmp/qr_code.h")
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
