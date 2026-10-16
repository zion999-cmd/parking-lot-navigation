/*
 * Fixed test QR code - GENERATED, do not edit by hand.
 *
 *   payload : "PARKING-TEST-001" (16 characters)
 *   version : 1  (21 x 21 modules)
 *   ECC     : L
 *
 * Regenerate with:
 *     python3 tools/make-qr-test.py "PARKING-TEST-001" \
 *         > firmware/esp32-s3-cam/main/qr_test.h
 *
 * One bit per module, bit 0 = leftmost column of that row, which is what
 * display.c walks when it paints the modules.
 */
#pragma once

#define QR_TEST_PAYLOAD "PARKING-TEST-001"
#define QR_TEST_SIZE    21
#define QR_TEST_ROW_BYTES 3

static const uint8_t qr_test_rows[QR_TEST_SIZE][QR_TEST_ROW_BYTES] = {
    { 0x7F, 0xD4, 0x1F },   /* row  0 */
    { 0x41, 0x55, 0x10 },   /* row  1 */
    { 0x5D, 0x4D, 0x17 },   /* row  2 */
    { 0x5D, 0x50, 0x17 },   /* row  3 */
    { 0x5D, 0x5F, 0x17 },   /* row  4 */
    { 0x41, 0x47, 0x10 },   /* row  5 */
    { 0x7F, 0xD5, 0x1F },   /* row  6 */
    { 0x00, 0x01, 0x00 },   /* row  7 */
    { 0xCB, 0xDC, 0x0D },   /* row  8 */
    { 0x0C, 0xAD, 0x17 },   /* row  9 */
    { 0xD7, 0x48, 0x07 },   /* row 10 */
    { 0xA4, 0x70, 0x0D },   /* row 11 */
    { 0x6B, 0xAC, 0x19 },   /* row 12 */
    { 0x00, 0xEB, 0x00 },   /* row 13 */
    { 0x7F, 0x1F, 0x0B },   /* row 14 */
    { 0x41, 0x4E, 0x0C },   /* row 15 */
    { 0x5D, 0xD0, 0x0F },   /* row 16 */
    { 0x5D, 0xC7, 0x18 },   /* row 17 */
    { 0x5D, 0xC6, 0x14 },   /* row 18 */
    { 0x41, 0x0F, 0x10 },   /* row 19 */
    { 0x7F, 0x9D, 0x05 },   /* row 20 */
};
