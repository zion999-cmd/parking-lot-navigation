#pragma once

#include <stdbool.h>
#include <esp_err.h>

/**
 * Initialise the camera on the ESP32-S3-CAM (KZB V2.0) board.
 *
 * Pin map, XCLK frequency, PSRAM placement and sensor orientation are reused
 * verbatim from the already proven reference project. See camera.c for the
 * provenance of every single value.
 */
esp_err_t camera_init(void);

/** True once camera_init() has succeeded. */
bool camera_ready(void);

/** Sensor model reported by the driver, e.g. "OV3660". Never NULL. */
const char *camera_sensor_name(void);

/** Human readable frame size, e.g. "800x600 (SVGA)". Never NULL. */
const char *camera_resolution_name(void);

/** Pixel format name, e.g. "JPEG". Never NULL. */
const char *camera_pixel_format_name(void);

/** Configured JPEG quality (0-63, lower means better). */
int camera_jpeg_quality(void);
