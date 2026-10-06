#include "wifi.h"

#include <cstdio>
#include <cstring>
#include <string>

#include <esp_log.h>
#include <esp_netif.h>
#include <esp_wifi.h>
#include <esp_timer.h>

#include "ssid_manager.h"
#include "wifi_manager.h"

/*
 * C adapter over the reused provisioning component.
 *
 * Everything radio-related is delegated to 78/esp-wifi-connect, which is the
 * reference project's provisioner (SoftAP + captive portal + NVS credentials).
 * The two things this file adds on top:
 *
 *  1. a C API, so camera/stream/main stay plain C;
 *  2. the hand-off from "portal connected the station" back to normal station
 *     mode. The component connects the station itself from its /submit handler
 *     while the SoftAP is up, and WifiManager never learns about that, so this
 *     file watches IP_EVENT_STA_GOT_IP directly and, after a grace period for
 *     the phone to load the success page, leaves config mode properly.
 */

static const char *TAG = "wifi";

#define AP_SSID_PREFIX "ParkingCam"
#define AP_LANGUAGE "en-US"

/*
 * The reference project ships 10 s .. 300 s scan intervals. A camera node that
 * loses its access point should come back within a minute, not five, so the
 * upper bound is lowered. The lower bound is kept as-is.
 */
#define SCAN_INTERVAL_MIN_S 10
#define SCAN_INTERVAL_MAX_S 60

/* Let the phone finish loading /done.html before the SoftAP disappears. */
#define LEAVE_CONFIG_MODE_DELAY_MS 4000

static wifi_event_cb_t s_callback = NULL;
static char s_ip[16] = "0.0.0.0";
static esp_timer_handle_t s_leave_config_timer = NULL;
static bool s_initialized = false;

static void notify(wifi_mgr_event_t event, const char *data)
{
    if (s_callback != NULL) {
        s_callback(event, data ? data : "");
    }
}

static void on_manager_event(WifiEvent event, const std::string &data)
{
    switch (event) {
    case WifiEvent::Scanning:
        notify(WIFI_EVT_SCANNING, "");
        break;
    case WifiEvent::Connecting:
        notify(WIFI_EVT_CONNECTING, data.c_str());
        break;
    case WifiEvent::Connected:
        notify(WIFI_EVT_CONNECTED, data.c_str());
        break;
    case WifiEvent::Disconnected:
        notify(WIFI_EVT_DISCONNECTED, data.c_str());
        break;
    case WifiEvent::ConfigModeEnter:
        notify(WIFI_EVT_CONFIG_MODE_ENTER, "");
        break;
    case WifiEvent::ConfigModeExit:
        notify(WIFI_EVT_CONFIG_MODE_EXIT, "");
        break;
    }
}

static void leave_config_mode_cb(void *arg)
{
    (void)arg;
    if (!WifiManager::GetInstance().IsConfigMode()) {
        return;
    }
    ESP_LOGI(TAG, "Portal did not request exit, leaving config mode anyway");
    WifiManager::GetInstance().StopConfigAp();
    WifiManager::GetInstance().StartStation();
}

static void on_ip_event(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    (void)arg;
    if (base != IP_EVENT || id != IP_EVENT_STA_GOT_IP) {
        return;
    }

    const ip_event_got_ip_t *ev = (const ip_event_got_ip_t *)data;
    snprintf(s_ip, sizeof(s_ip), IPSTR, IP2STR(&ev->ip_info.ip));

    if (WifiManager::GetInstance().IsConfigMode()) {
        /*
         * The provisioning portal just connected the station. Give the browser
         * time to reach /done.html and POST /exit (which does the clean
         * hand-off); if that never happens, do it ourselves.
         */
        if (s_leave_config_timer != NULL) {
            esp_timer_stop(s_leave_config_timer);
            esp_timer_start_once(s_leave_config_timer, LEAVE_CONFIG_MODE_DELAY_MS * 1000ULL);
        }
    }
}

static void on_wifi_event(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    (void)arg;
    (void)data;
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        strlcpy(s_ip, "0.0.0.0", sizeof(s_ip));
    }
}

esp_err_t wifi_init(void)
{
    if (s_initialized) {
        return ESP_OK;
    }

    WifiManagerConfig config;
    config.ssid_prefix = AP_SSID_PREFIX;
    config.language = AP_LANGUAGE;
    config.station_scan_min_interval_seconds = SCAN_INTERVAL_MIN_S;
    config.station_scan_max_interval_seconds = SCAN_INTERVAL_MAX_S;

    /* Does NVS init, esp_netif_init, default event loop and esp_wifi_init. */
    if (!WifiManager::GetInstance().Initialize(config)) {
        ESP_LOGE(TAG, "WifiManager::Initialize failed");
        return ESP_FAIL;
    }

    WifiManager::GetInstance().SetEventCallback(on_manager_event);

    esp_timer_create_args_t timer_args = {};
    timer_args.callback = leave_config_mode_cb;
    timer_args.name = "leave_cfg_ap";
    ESP_ERROR_CHECK(esp_timer_create(&timer_args, &s_leave_config_timer));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, on_ip_event, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, WIFI_EVENT_STA_DISCONNECTED, on_wifi_event, NULL, NULL));

    wifi_disable_power_save();

    s_initialized = true;
    ESP_LOGI(TAG, "WiFi manager initialised (SoftAP prefix \"%s\")", AP_SSID_PREFIX);
    return ESP_OK;
}

int wifi_power_save_mode(void)
{
    wifi_ps_type_t mode = WIFI_PS_NONE;
    if (esp_wifi_get_ps(&mode) != ESP_OK) {
        return -1;
    }
    return (int)mode;
}

const char *wifi_power_save_name(void)
{
    switch (wifi_power_save_mode()) {
    case WIFI_PS_NONE:      return "none";
    case WIFI_PS_MIN_MODEM: return "min_modem";
    case WIFI_PS_MAX_MODEM: return "max_modem";
    default:                return "unknown";
    }
}

int wifi_disable_power_save(void)
{
    /*
     * WIFI_PS_NONE - and, the part that was missing, a readback of what the driver
     * actually did.
     *
     * Setting this once is not enough on this board. It runs after esp_wifi_init,
     * again on every connect, and is re-checked every 10 s, because
     * esp_wifi_set_ps() can return ESP_OK while the driver stays in modem sleep, and
     * the driver re-applies its own mode on reconnect.
     *
     * Why it matters, measured: with the board idle and associated, ping to it
     * averaged 555 ms with 15% loss while the router answered the same Mac in 2 ms
     * with 0% loss. The best ping to the board was 5.4 ms, so the RF path is fine -
     * the radio was simply asleep. That latency, not the camera and not the TCP
     * buffers, is what caps the stream around 130 KB/s: the earlier 13.8 fps and
     * today's 4.3 fps are the same byte rate with different JPEG sizes.
     */
    esp_err_t err = esp_wifi_set_ps(WIFI_PS_NONE);
    int actual = wifi_power_save_mode();

    if (err != ESP_OK) {
        ESP_LOGW(TAG, "esp_wifi_set_ps(WIFI_PS_NONE) failed: %s", esp_err_to_name(err));
    }
    if (actual < 0) {
        ESP_LOGW(TAG, "could not read back the WiFi power save mode");
    } else if (actual != WIFI_PS_NONE) {
        ESP_LOGW(TAG, "WiFi power save is STILL %s after asking for none - the radio will sleep",
                 wifi_power_save_name());
    } else {
        ESP_LOGI(TAG, "WiFi power save: off (readback verified)");
    }
    return actual;
}

void wifi_link_summary(char *out, size_t n)
{
    if (out == NULL || n == 0) {
        return;
    }

    wifi_ap_record_t ap = {};
    if (esp_wifi_sta_get_ap_info(&ap) != ESP_OK) {
        snprintf(out, n, "not associated");
        return;
    }

    const char *phy = "?";
    wifi_phy_mode_t mode;
    if (esp_wifi_sta_get_negotiated_phymode(&mode) == ESP_OK) {
        switch (mode) {
        case WIFI_PHY_MODE_LR:   phy = "11LR";     break;
        case WIFI_PHY_MODE_11B:  phy = "802.11b";  break;
        case WIFI_PHY_MODE_11G:  phy = "802.11g";  break;
        case WIFI_PHY_MODE_11A:  phy = "802.11a";  break;
        case WIFI_PHY_MODE_HT20: phy = "11n/HT20"; break;
        case WIFI_PHY_MODE_HT40: phy = "11n/HT40"; break;
        case WIFI_PHY_MODE_HE20: phy = "11ax/HE20"; break;
        case WIFI_PHY_MODE_VHT20: phy = "11ac/VHT20"; break;
        default: break;
        }
    }

    snprintf(out, n, "ch%u+%d %s %d dBm", (unsigned)ap.primary, (int)ap.second, phy, ap.rssi);
}

void wifi_set_event_callback(wifi_event_cb_t cb)
{
    s_callback = cb;
}

bool wifi_has_credentials(void)
{
    return !SsidManager::GetInstance().GetSsidList().empty();
}

int wifi_saved_count(void)
{
    return (int)SsidManager::GetInstance().GetSsidList().size();
}

esp_err_t wifi_clear_credentials(void)
{
    ESP_LOGW(TAG, "Clearing all saved Wi-Fi credentials from NVS");
    SsidManager::GetInstance().Clear();
    strlcpy(s_ip, "0.0.0.0", sizeof(s_ip));
    return ESP_OK;
}

void wifi_start_station(void)
{
    WifiManager::GetInstance().StartStation();
}

void wifi_stop_station(void)
{
    WifiManager::GetInstance().StopStation();
}

void wifi_start_config_ap(void)
{
    WifiManager::GetInstance().StartConfigAp();
}

void wifi_stop_config_ap(void)
{
    WifiManager::GetInstance().StopConfigAp();
}

bool wifi_is_connected(void)
{
    return WifiManager::GetInstance().IsConnected();
}

bool wifi_is_config_mode(void)
{
    return WifiManager::GetInstance().IsConfigMode();
}

const char *wifi_ip_string(void)
{
    return s_ip;
}

static char s_ssid[33] = "";

const char *wifi_ssid(void)
{
    std::string ssid = WifiManager::GetInstance().GetSsid();
    strlcpy(s_ssid, ssid.c_str(), sizeof(s_ssid));
    return s_ssid;
}

int wifi_rssi(void)
{
    return WifiManager::GetInstance().GetRssi();
}

static char s_ap_ssid[33] = "";
static char s_ap_url[32] = "";

const char *wifi_ap_ssid(void)
{
    std::string ssid = WifiManager::GetInstance().GetApSsid();
    strlcpy(s_ap_ssid, ssid.c_str(), sizeof(s_ap_ssid));
    return s_ap_ssid;
}

const char *wifi_ap_url(void)
{
    std::string url = WifiManager::GetInstance().GetApWebUrl();
    strlcpy(s_ap_url, url.c_str(), sizeof(s_ap_url));
    return s_ap_url;
}
