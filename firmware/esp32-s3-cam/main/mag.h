/* Four geomagnetic sensors behind a TCA9548A I2C multiplexer.

   HMC5883L has a fixed address (0x1E), so four of them cannot share a bus. The TCA9548A sits
   between the ESP32 and the sensors and connects exactly one channel at a time; every sensor keeps
   address 0x1E and only the selected one is visible.

   Hardware, as wired:
     ESP32 GPIO41 -> mux SDA   (I2C port 1; port 0 belongs to the camera's SCCB bus)
     ESP32 GPIO42 -> mux SCL
     mux  A0/A1/A2 -> GND, so the mux answers at 0x70
     mux  VIN -> 3V3, never 5V (HMC5883L dies above 3.6 V)
     sensor VIN -> 3V3 directly; the mux switches SDA/SCL only, it does not pass power

   This module is best-effort on purpose: a magnetometer that stops answering must not take the
   camera, the bay lights or the gate with it.
*/
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"

typedef enum {
    MAG_CHIP_NONE = 0,
    MAG_CHIP_HMC5883L,
    MAG_CHIP_QMC5883L,
    MAG_CHIP_QMC5883P,   /* 0x2C, chip id 0x80 - not register compatible with either */
} mag_chip_t;

typedef struct {
    bool ok;               /* last read succeeded */
    mag_chip_t chip;
    uint8_t channel;       /* TCA9548A channel, 0..3 */
    int16_t x, y, z;       /* raw counts */
    float field_ut;        /* |B| in microtesla, scaled by the chip's LSB size */
    float ut_per_lsb;      /* which range was inferred, so the number can be checked */
    uint32_t reads;        /* successful reads since boot */
    uint32_t errors;       /* failed reads since boot */
} mag_reading_t;

esp_err_t mag_start(void);
bool mag_available(void);
uint8_t mag_count(void);
bool mag_get(uint8_t index, mag_reading_t *out);
const char *mag_diag(void);
