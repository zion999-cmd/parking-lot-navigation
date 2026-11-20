#include <stdio.h>
#include <string.h>

#include "qr_matrix.h"

static int failures;

static void expect(const char *name, bool condition)
{
    printf("%s %s\n", condition ? "ok" : "FAIL", name);
    if (!condition) failures++;
}

static void finder(uint8_t *rows, int row_bytes, int ox, int oy)
{
    for (int y = 0; y < 7; y++) {
        for (int x = 0; x < 7; x++) {
            if (x == 0 || x == 6 || y == 0 || y == 6 ||
                    (x >= 2 && x <= 4 && y >= 2 && y <= 4)) {
                rows[(oy + y) * row_bytes + (ox + x) / 8] |= 1U << ((ox + x) % 8);
            }
        }
    }
}

int main(void)
{
    uint8_t rows[QR_MATRIX_MAX_SIZE][QR_MATRIX_ROW_BYTES];
    memset(rows, 0, sizeof(rows));
    const int size = 29, row_bytes = 4;
    finder(&rows[0][0], row_bytes, 0, 0);
    finder(&rows[0][0], row_bytes, size - 7, 0);
    finder(&rows[0][0], row_bytes, 0, size - 7);

    expect("valid 29x29 matrix", qr_matrix_valid(&rows[0][0], size, row_bytes));
    expect("wrong row stride", !qr_matrix_valid(&rows[0][0], size, 3));
    expect("too small", !qr_matrix_valid(&rows[0][0], 17, 3));
    expect("not a QR version size", !qr_matrix_valid(&rows[0][0], 30, 4));
    expect("NULL", !qr_matrix_valid(NULL, size, row_bytes));

    rows[0][0] &= ~1U;
    expect("damaged top-left finder", !qr_matrix_valid(&rows[0][0], size, row_bytes));
    rows[0][0] |= 1U;
    ((uint8_t *)rows)[28 * row_bytes + 3] |= 0x80U;  /* padding bits 29..31 */
    expect("nonzero padding", !qr_matrix_valid(&rows[0][0], size, row_bytes));
    ((uint8_t *)rows)[28 * row_bytes + 3] &= ~0x80U;
    expect("restored matrix", qr_matrix_valid(&rows[0][0], size, row_bytes));

    printf("%s\n", failures ? "FAILURES" : "all qr_matrix checks passed");
    return failures ? 1 : 0;
}
