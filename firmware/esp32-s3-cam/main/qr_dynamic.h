#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "qr_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QR_URL_MAX 160
#define QR_SOURCE_URL_MAX 192
#define QR_ERROR_MAX 96

typedef struct {
    bool valid;
    int size;
    int row_bytes;
    /* Packed contiguously using row_bytes, so validation and drawing share layout. */
    uint8_t rows[QR_MATRIX_MAX_SIZE * QR_MATRIX_ROW_BYTES];
    char url[QR_URL_MAX];                 /* payload encoded by rows, valid only when valid */
    char source_url[QR_SOURCE_URL_MAX];   /* API used for the most recent attempt */
    int64_t last_attempt_us;              /* device uptime, for diagnostics */
    int64_t last_success_us;              /* last validated matrix, retained after failure */
    char error[QR_ERROR_MAX];             /* empty when the last attempt succeeded */
} qr_snapshot_t;

/* Start the independent HTTP worker. Safe to call once after Wi-Fi is up. */
bool qr_dynamic_start(void);

/* Copy a consistent RAM snapshot. No HTTP or LCD work happens here. */
void qr_dynamic_snapshot(qr_snapshot_t *out);

#ifdef __cplusplus
}
#endif
