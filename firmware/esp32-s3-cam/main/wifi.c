#include "wifi.h"

#include <stdio.h>
#include <string.h>

#include <freertos/FreeRTOS.h>
#include <freertos/event_groups.h>

#include <esp_event.h>
#include <esp_log.h>
#include <esp_netif.h>
#include <esp_wifi.h>

/*
 * Credentials live in a gitignored header so they can never be committed.
 * The example file is committed instead. See README "Wi-Fi configuration".
 */
#if __has_include("wifi_credentials.h")
#include "wifi_credentials.h"
#else
#error "main/wifi_credentials.h is missing. Copy main/wifi_credentials.h.example to main/wifi_credentials.h and fill in WIFI_SSID / WIFI_PASSWORD."
#endif

static const char *TAG = "wifi";

#define WIFI_GOT_IP_BIT BIT0

/* How long to wait for a DHCP lease before giving up. */
#define WIFI_CONNECT_TIMEOUT_MS 60000

static EventGroupHandle_t s_events;
static bool s_connected = false;
static char s_ip[16] = "0.0.0.0";

static void on_wifi_event(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
        return;
    }

    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        const wifi_event_sta_disconnected_t *ev = (const wifi_event_sta_disconnected_t *)data;
        s_connected = false;
        strlcpy(s_ip, "0.0.0.0", sizeof(s_ip));
        ESP_LOGW(TAG, "WiFi disconnected (reason=%d), reconnecting", ev->reason);
        esp_wifi_connect();
        return;
    }

    if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        const ip_event_got_ip_t *ev = (const ip_event_got_ip_t *)data;
        snprintf(s_ip, sizeof(s_ip), IPSTR, IP2STR(&ev->ip_info.ip));
        s_connected = true;

        /* These two lines are part of the P0001 acceptance criteria. */
        ESP_LOGI(TAG, "WiFi connected");
        ESP_LOGI(TAG, "IP address: %s", s_ip);

        xEventGroupSetBits(s_events, WIFI_GOT_IP_BIT);
    }
}

esp_err_t wifi_sta_start(void)
{
    s_events = xEventGroupCreate();
    if (s_events == NULL) {
        return ESP_ERR_NO_MEM;
    }

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t init_cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init_cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, on_wifi_event, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, on_wifi_event, NULL, NULL));

    wifi_config_t wifi_cfg = {0};
    strlcpy((char *)wifi_cfg.sta.ssid, WIFI_SSID, sizeof(wifi_cfg.sta.ssid));
    strlcpy((char *)wifi_cfg.sta.password, WIFI_PASSWORD, sizeof(wifi_cfg.sta.password));
    wifi_cfg.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_cfg));
    ESP_ERROR_CHECK(esp_wifi_start());

    /*
     * Disable Wi-Fi power save. With power save enabled the access point buffers
     * frames until the next DTIM beacon, which adds tens to hundreds of
     * milliseconds of jitter to a video stream. Latency matters more than the
     * small current saving here.
     */
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));

    ESP_LOGI(TAG, "Connecting to SSID \"%s\"...", WIFI_SSID);

    EventBits_t bits = xEventGroupWaitBits(s_events, WIFI_GOT_IP_BIT,
                                           pdFALSE, pdFALSE,
                                           pdMS_TO_TICKS(WIFI_CONNECT_TIMEOUT_MS));
    if ((bits & WIFI_GOT_IP_BIT) == 0) {
        ESP_LOGE(TAG, "No IP address after %d s - check WIFI_SSID / WIFI_PASSWORD "
                      "and that the access point is in range",
                 WIFI_CONNECT_TIMEOUT_MS / 1000);
        return ESP_ERR_TIMEOUT;
    }

    return ESP_OK;
}

bool wifi_connected(void)
{
    return s_connected;
}

const char *wifi_ip_string(void)
{
    return s_ip;
}

int wifi_rssi(void)
{
    if (!s_connected) {
        return 0;
    }

    wifi_ap_record_t ap = {0};
    if (esp_wifi_sta_get_ap_info(&ap) != ESP_OK) {
        return 0;
    }
    return ap.rssi;
}
