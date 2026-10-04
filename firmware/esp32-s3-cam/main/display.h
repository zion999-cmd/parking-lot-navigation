#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Thin status-display interface (P0001.2).
 *
 * The TFT is a *device status panel*, not a UI. It answers "what should a human
 * do next" and makes the device IP visible without a serial cable.
 *
 * Architectural rule: camera / Wi-Fi / OTA / stream code only ever *reports*
 * state through these functions. No LCD code lives in those modules, and no
 * module touches esp_lcd, the SPI bus or the backlight.
 *
 * Every function below is safe to call from any task, including Wi-Fi event
 * callbacks: they only post a small message to a queue (and drop it if the
 * display task is behind). Rendering happens in the display task.
 */

/** Bring up SPI3, the ST7789 panel and the backlight, then start the task. */
bool display_init(void);

/** True once the panel is usable; all show_* calls are no-ops if false. */
bool display_ready(void);

void display_show_boot(const char *version);

/** First-run / fallback screen: how to join the SoftAP and where to go. */
void display_show_provisioning(const char *ap_ssid, const char *ap_url);

void display_show_wifi_connecting(const char *ssid);

/** Normal running state. 'ip' is the important bit - keep it readable. */
void display_show_ready(const char *ip);

void display_show_camera_error(void);

void display_show_wifi_error(void);

void display_show_ota_checking(void);

/** version_from -> version_to, percent 0..100. */
void display_show_ota_progress(const char *version_from, const char *version_to, int percent);

/** ok == true: "Update complete / Rebooting..."; false: "Update failed". */
void display_show_ota_result(bool ok);

/** Escape hatch for anything not covered above. */
void display_show_error(const char *title, const char *hint);

#ifdef __cplusplus
}
#endif
