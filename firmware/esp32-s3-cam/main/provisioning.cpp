#include "provisioning.h"

#include <cstdio>
#include <cstring>

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include <esp_console.h>
#include <esp_log.h>
#include <esp_timer.h>

#include "wifi.h"

static const char *TAG = "prov";

/* Give up on the first association attempt after this long and open the portal. */
#define STA_CONNECT_TIMEOUT_MS 60000

/*
 * If the node was connected and then loses its access point for this long, fall
 * back to the portal instead of retrying forever. Long enough that a router
 * reboot does not interrupt a running demo.
 */
#define RECONNECT_FALLBACK_MS 300000

typedef enum {
    PROV_BOOT,
    PROV_STA_CONNECTING,
    PROV_CONNECTED,
    PROV_RECONNECTING,
    PROV_PROVISIONING,
} prov_state_t;

static prov_state_t s_state = PROV_BOOT;
static esp_timer_handle_t s_timeout_timer = NULL;
static provisioning_hook_t s_on_connected = NULL;
static provisioning_hook_t s_on_disconnected = NULL;
static provisioning_hook_t s_on_before_config_ap = NULL;

static const char *state_name(prov_state_t s)
{
    switch (s) {
    case PROV_BOOT:           return "BOOT";
    case PROV_STA_CONNECTING: return "STA_CONNECTING";
    case PROV_CONNECTED:      return "CONNECTED";
    case PROV_RECONNECTING:   return "STA_RECONNECTING";
    case PROV_PROVISIONING:   return "PROVISIONING";
    }
    return "?";
}

const char *provisioning_state_name(void)
{
    return state_name(s_state);
}

bool provisioning_in_config_mode(void)
{
    return s_state == PROV_PROVISIONING;
}

static void set_state(prov_state_t s)
{
    if (s_state == s) {
        return;
    }
    ESP_LOGI(TAG, "state %s -> %s", state_name(s_state), state_name(s));
    s_state = s;
}

static void cancel_timeout(void)
{
    if (s_timeout_timer != NULL) {
        esp_timer_stop(s_timeout_timer);
    }
}

static void arm_timeout(uint64_t ms)
{
    if (s_timeout_timer == NULL) {
        return;
    }
    esp_timer_stop(s_timeout_timer);
    esp_timer_start_once(s_timeout_timer, ms * 1000ULL);
}

static void enter_provisioning(void)
{
    if (s_state == PROV_PROVISIONING) {
        return;
    }

    /* The portal needs port 80, so let the application shut its server down. */
    if (s_on_before_config_ap != NULL) {
        s_on_before_config_ap();
    }

    set_state(PROV_PROVISIONING);
    wifi_start_config_ap();

    ESP_LOGW(TAG, "Wi-Fi provisioning mode");
    ESP_LOGW(TAG, "  1. join the access point : %s", wifi_ap_ssid());
    ESP_LOGW(TAG, "  2. open in a browser    : %s", wifi_ap_url());
    ESP_LOGW(TAG, "  3. pick your network and enter its password");
}

static void timeout_cb(void *arg)
{
    (void)arg;
    if (s_state == PROV_STA_CONNECTING) {
        ESP_LOGW(TAG, "Could not connect within %d s", STA_CONNECT_TIMEOUT_MS / 1000);
        wifi_stop_station();
        enter_provisioning();
    } else if (s_state == PROV_RECONNECTING) {
        ESP_LOGW(TAG, "Disconnected for %d s", RECONNECT_FALLBACK_MS / 1000);
        wifi_stop_station();
        enter_provisioning();
    }
}

static void start_station_attempt(void)
{
    if (!wifi_has_credentials()) {
        enter_provisioning();
        return;
    }

    set_state(PROV_STA_CONNECTING);
    arm_timeout(STA_CONNECT_TIMEOUT_MS);
    wifi_start_station();
}

static void on_wifi_event(wifi_mgr_event_t event, const char *data)
{
    switch (event) {
    case WIFI_EVT_SCANNING:
        ESP_LOGD(TAG, "scanning");
        break;

    case WIFI_EVT_CONNECTING:
        ESP_LOGI(TAG, "connecting to \"%s\"", data);
        break;

    case WIFI_EVT_CONNECTED:
        cancel_timeout();
        set_state(PROV_CONNECTED);
        ESP_LOGI(TAG, "WiFi connected");
        ESP_LOGI(TAG, "IP address: %s", wifi_ip_string());
        ESP_LOGI(TAG, "SSID: %s, RSSI: %d dBm", data, wifi_rssi());
        if (s_on_connected != NULL) {
            s_on_connected();
        }
        break;

    case WIFI_EVT_DISCONNECTED:
        /*
         * While still trying to associate, ignore this: the station driver does
         * its own retries and the STA_CONNECTING timeout covers the whole
         * attempt. Only a drop *after* a working connection arms the fallback.
         */
        if (s_state == PROV_CONNECTED) {
            ESP_LOGW(TAG, "WiFi link lost, reconnecting (fallback to provisioning in %d s)",
                     RECONNECT_FALLBACK_MS / 1000);
            set_state(PROV_RECONNECTING);
            arm_timeout(RECONNECT_FALLBACK_MS);
        }
        if (s_on_disconnected != NULL) {
            s_on_disconnected();
        }
        break;

    case WIFI_EVT_CONFIG_MODE_ENTER:
        set_state(PROV_PROVISIONING);
        break;

    case WIFI_EVT_CONFIG_MODE_EXIT:
        /* The portal is down; connect with whatever was just saved. */
        start_station_attempt();
        break;
    }
}

void provisioning_set_hooks(provisioning_hook_t on_connected,
                            provisioning_hook_t on_disconnected,
                            provisioning_hook_t on_before_config_ap)
{
    s_on_connected = on_connected;
    s_on_disconnected = on_disconnected;
    s_on_before_config_ap = on_before_config_ap;
}

void provisioning_start(void)
{
    esp_timer_create_args_t timer_args = {};
    timer_args.callback = timeout_cb;
    timer_args.name = "prov_timeout";
    ESP_ERROR_CHECK(esp_timer_create(&timer_args, &s_timeout_timer));

    wifi_set_event_callback(on_wifi_event);

    ESP_LOGI(TAG, "Saved Wi-Fi credentials in NVS: %d", wifi_saved_count());
    start_station_attempt();
}

void provisioning_clear_credentials(void)
{
    wifi_clear_credentials();

    if (wifi_is_config_mode()) {
        ESP_LOGW(TAG, "Credentials cleared, portal is already up");
        return;
    }

    cancel_timeout();
    wifi_stop_station();
    enter_provisioning();
}

/* ------------------------------------------------------------------------- */
/* Serial re-provisioning command                                             */
/* ------------------------------------------------------------------------- */

static int cmd_clearwifi(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    ESP_LOGW(TAG, "'clearwifi' requested");
    provisioning_clear_credentials();
    return 0;
}

void provisioning_start_serial_console(void)
{
    /*
     * Use the ESP-IDF console component rather than poking the UART VFS by
     * hand. A hand-rolled "uart_vfs_dev_use_driver(0) + fgets(stdin)" is only
     * half the setup - esp_console_new_repl_uart() also drains stdout, sets the
     * line endings, installs the UART driver and switches the port to
     * non-blocking polling. Skipping that panicked in newlib's stdio lock.
     */
    esp_console_repl_config_t repl_cfg = ESP_CONSOLE_REPL_CONFIG_DEFAULT();
    repl_cfg.prompt = "parking> ";
    repl_cfg.max_cmdline_length = 64;

    esp_console_dev_uart_config_t uart_cfg = ESP_CONSOLE_DEV_UART_CONFIG_DEFAULT();

    esp_console_cmd_t cmd = {};
    cmd.command = "clearwifi";
    cmd.help = "Erase the saved Wi-Fi network and re-enter provisioning mode";
    cmd.func = cmd_clearwifi;

    esp_console_repl_t *repl = NULL;
    ESP_ERROR_CHECK(esp_console_new_repl_uart(&uart_cfg, &repl_cfg, &repl));
    ESP_ERROR_CHECK(esp_console_cmd_register(&cmd));
    ESP_ERROR_CHECK(esp_console_start_repl(repl));

    ESP_LOGI(TAG, "Serial console ready: type 'clearwifi' + Enter to erase the "
                  "saved network and re-enter provisioning mode.");
}
