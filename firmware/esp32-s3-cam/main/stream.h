#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <esp_err.h>

/** Path of the MJPEG stream endpoint. */
#define STREAM_PATH "/stream"

/** Path that asks the OTA client to check for an update right now. */
#define STREAM_OTA_CHECK_PATH "/api/ota/check"

/** Camera-only benchmark: takes no video path at all, only fb_get/fb_return. */
#define STREAM_BENCH_PATH "/api/bench"

/** Start the HTTP server: "/" status page and STREAM_PATH MJPEG stream. */
esp_err_t stream_server_start(void);

/**
 * Stop the HTTP server and release port 80.
 *
 * The provisioning captive portal also binds port 80, so the application has to
 * call this before the SoftAP is raised. The two never run at the same time.
 */
void stream_server_stop(void);

bool stream_server_is_running(void);

/** Total number of JPEG frames handed to clients since boot. */
uint32_t stream_frames_sent(void);

/**
 * Smoothed frames-per-second actually delivered to clients.
 *
 * Measured across all connected clients, so it is N times the capture rate when
 * N clients are streaming. The acceptance test uses exactly one client.
 */
float stream_fps(void);

/** Number of clients currently attached to STREAM_PATH. */
int stream_client_count(void);
