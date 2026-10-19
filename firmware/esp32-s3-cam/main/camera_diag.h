#pragma once
#include "esp_err.h"
#include "esp_http_server.h"

esp_err_t camera_diag_start(int vsync_pin);
esp_err_t camera_diag_register(httpd_handle_t server);

/*
 * Read-only edge snapshot for the on-screen diagnostics page. No network request,
 * no side effects: it copies the counters the ISR is already maintaining.
 */
void camera_diag_edges(uint32_t *rising, uint32_t *intervals, uint64_t *total_us);
