#pragma once

#include <stdbool.h>
#include <esp_err.h>

/**
 * Bring up Wi-Fi in station mode and block until an IPv4 address is obtained.
 *
 * Credentials are read at compile time from the gitignored
 * main/wifi_credentials.h - see main/wifi_credentials.h.example.
 *
 * After this returns the driver keeps reconnecting on its own if the link
 * drops; check wifi_connected() to know the current state.
 */
esp_err_t wifi_sta_start(void);

/** True while the station holds an IPv4 address. */
bool wifi_connected(void);

/** Current IPv4 address as a string; "0.0.0.0" before the first lease. */
const char *wifi_ip_string(void);

/** Current RSSI in dBm, or 0 when not connected. */
int wifi_rssi(void);
