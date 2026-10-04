#pragma once

#include <stdbool.h>
#include <esp_err.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Thin C adapter over the reused provisioning component (78/esp-wifi-connect).
 *
 * The component is C++ (WifiManager / WifiStation / WifiConfigurationAp /
 * SsidManager). This header is the only surface the rest of the firmware sees,
 * so camera/stream/main stay plain C.
 */

/** Events forwarded from the Wi-Fi manager. Mirrors the component's WifiEvent. */
typedef enum {
    WIFI_EVT_SCANNING,
    WIFI_EVT_CONNECTING,
    WIFI_EVT_CONNECTED,
    WIFI_EVT_DISCONNECTED,
    WIFI_EVT_CONFIG_MODE_ENTER,
    WIFI_EVT_CONFIG_MODE_EXIT,
} wifi_mgr_event_t;

typedef void (*wifi_event_cb_t)(wifi_mgr_event_t event, const char *data);

/**
 * Initialise NVS, netif, the default event loop, the Wi-Fi driver and the
 * manager. Safe to call once at boot.
 */
esp_err_t wifi_init(void);

/** Register the single event callback used by the provisioning state machine. */
void wifi_set_event_callback(wifi_event_cb_t cb);

/** True when NVS holds at least one saved SSID. */
bool wifi_has_credentials(void);

/** Number of SSIDs currently stored in NVS (namespace "wifi"). */
int wifi_saved_count(void);

/** Erase every saved SSID/password from NVS. Backs the serial `clearwifi`. */
esp_err_t wifi_clear_credentials(void);

/* Station and SoftAP control. All four are idempotent. */
void wifi_start_station(void);
void wifi_stop_station(void);
void wifi_start_config_ap(void);
void wifi_stop_config_ap(void);

bool wifi_is_connected(void);
bool wifi_is_config_mode(void);

/** Current station IPv4 address, or "0.0.0.0" before the first lease. */
const char *wifi_ip_string(void);

/** SSID of the currently associated access point, or "". */
const char *wifi_ssid(void);

/** RSSI in dBm, or 0 when not associated. */
int wifi_rssi(void);

/** SSID of the provisioning SoftAP, or "" when not in config mode. */
const char *wifi_ap_ssid(void);

/** URL of the provisioning web page, e.g. "http://192.168.4.1". */
const char *wifi_ap_url(void);

#ifdef __cplusplus
}
#endif
