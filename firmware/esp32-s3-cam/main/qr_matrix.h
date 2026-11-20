#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QR_MATRIX_MIN_SIZE 21
#define QR_MATRIX_MAX_SIZE 45
#define QR_MATRIX_ROW_BYTES ((QR_MATRIX_MAX_SIZE + 7) / 8)

/* Validate size, padding and all three 7x7 finder patterns. Rows are packed
 * least-significant bit first, exactly as display.c paints them. */
bool qr_matrix_valid(const uint8_t *rows, int size, int row_bytes);

#ifdef __cplusplus
}
#endif
