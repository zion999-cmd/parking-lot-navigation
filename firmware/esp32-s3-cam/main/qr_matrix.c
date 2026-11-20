#include "qr_matrix.h"

#include <stddef.h>

static bool module(const uint8_t *rows, int row_bytes, int x, int y)
{
    return (rows[y * row_bytes + x / 8] >> (x % 8)) & 1;
}

static bool finder_ok(const uint8_t *rows, int row_bytes, int ox, int oy)
{
    for (int y = 0; y < 7; y++) {
        for (int x = 0; x < 7; x++) {
            bool expected = x == 0 || x == 6 || y == 0 || y == 6 ||
                            (x >= 2 && x <= 4 && y >= 2 && y <= 4);
            if (module(rows, row_bytes, ox + x, oy + y) != expected) {
                return false;
            }
        }
    }
    return true;
}

bool qr_matrix_valid(const uint8_t *rows, int size, int row_bytes)
{
    if (rows == NULL || size < QR_MATRIX_MIN_SIZE || size > QR_MATRIX_MAX_SIZE ||
            (size - QR_MATRIX_MIN_SIZE) % 4 != 0 || row_bytes != (size + 7) / 8) {
        return false;
    }
    if (size % 8 != 0) {
        uint8_t used = (uint8_t)((1U << (size % 8)) - 1U);
        for (int y = 0; y < size; y++) {
            if ((rows[y * row_bytes + row_bytes - 1] & (uint8_t)~used) != 0) {
                return false;
            }
        }
    }
    return finder_ok(rows, row_bytes, 0, 0) &&
           finder_ok(rows, row_bytes, size - 7, 0) &&
           finder_ok(rows, row_bytes, 0, size - 7);
}
