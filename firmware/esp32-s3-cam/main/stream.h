#pragma once

#include <stdint.h>
#include <esp_err.h>

/** Path of the MJPEG stream endpoint. */
#define STREAM_PATH "/stream"

/** Start the HTTP server: "/" status page and STREAM_PATH MJPEG stream. */
esp_err_t stream_server_start(void);

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
