#include <stdio.h>

#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <freertos/task.h>

#include <esp_heap_caps.h>
#include <esp_idf_version.h>
#include <esp_log.h>
#include <esp_system.h>
#include <esp_timer.h>
#include <sdkconfig.h>

#include "camera.h"
#include "discovery.h"
#include "display.h"
#include "gate.h"
#include "led.h"
#include "ota.h"
#include "provisioning.h"
#include "stream.h"
#include "esp_wifi.h"
#include "wifi.h"

static const char *TAG = "app";

/* How often the runtime health line is printed. */
#define DIAGNOSTICS_PERIOD_MS 10000

/* Pause before rebooting when a mandatory subsystem could not be brought up. */
#define FAILURE_RESTART_DELAY_MS 5000

/*
 * Boot order, and why it is this order:
 *
 *   wifi_init + provisioning_start   -> the node is either on the LAN or
 *                                       offering its configuration portal
 *   camera + stream server           -> started by network_task once an IPv4
 *                                       address exists, so the camera and the
 *                                       PSRAM frame buffers are not powered
 *                                       while the device is merely waiting to
 *                                       be configured, and so port 80 is free
 *                                       for the captive portal.
 *
 * That keeps the P0001 startup log contract intact:
 *   WiFi connected / IP address / Camera initialized / Stream server started
 */

#if CONFIG_QR_SHOW_ON_BOOT
/*
 * One-shot timer that takes the QR overlay down. A timer rather than a task
 * because there is nothing to wait for: display_restore() only posts to the
 * display queue, which is safe from the esp_timer task.
 */
static esp_timer_handle_t s_qr_timer = NULL;

static void qr_overlay_off(void *arg)
{
    (void)arg;
    display_restore();
}

static void qr_overlay_start(void)
{
    if (s_qr_timer == NULL) {
        const esp_timer_create_args_t args = {
            .callback = qr_overlay_off,
            .name = "qr_panel",
        };
        if (esp_timer_create(&args, &s_qr_timer) != ESP_OK) {
            ESP_LOGW(TAG, "could not create the QR timer - the code stays on screen");
            return;
        }
    }
    esp_timer_start_once(s_qr_timer, (uint64_t)CONFIG_QR_BOOT_SECONDS * 1000000ULL);
    ESP_LOGI(TAG, "test QR shown for %d s", CONFIG_QR_BOOT_SECONDS);
}
#endif

static SemaphoreHandle_t s_connected_sem = NULL;
static SemaphoreHandle_t s_disconnected_sem = NULL;

/* These three run in the Wi-Fi event task, so they only signal. */
static void on_connected(void)
{
    xSemaphoreGive(s_connected_sem);
}

static void on_disconnected(void)
{
    xSemaphoreGive(s_disconnected_sem);
}

static void on_before_config_ap(void)
{
    /* The captive portal needs port 80; the MJPEG server owns it right now. */
    stream_server_stop();
}

static void restart_after_failure(const char *what)
{
    ESP_LOGE(TAG, "%s could not be initialised - restarting in %d s",
             what, FAILURE_RESTART_DELAY_MS / 1000);
    vTaskDelay(pdMS_TO_TICKS(FAILURE_RESTART_DELAY_MS));
    esp_restart();
}

static void network_task(void *arg)
{
    (void)arg;
    bool camera_up = false;
#if CONFIG_LED_ENABLE
    bool led_started = false;
#endif

    while (true) {
        xSemaphoreTake(s_connected_sem, portMAX_DELAY);

        if (!camera_up) {
            if (camera_init() != ESP_OK) {
                display_show_camera_error();
                restart_after_failure("Camera");
            }
            camera_up = true;

            /*
             * Rollback confirmation. This is the first point at which the new
             * image has proven it can do its job, so cancel the pending
             * rollback here (see README "OTA and rollback").
             */
            ota_mark_current_valid();
        }

        if (stream_server_start() != ESP_OK) {
            restart_after_failure("HTTP stream server");
        }

#if CONFIG_LED_ENABLE
        /*
         * Started here rather than at boot because it polls the PC over HTTP:
         * with no network there is nothing to read and the strip stays dark
         * anyway. Best effort - a missing strip never stops the camera node.
         */
        if (!led_started) {
            led_start();
            led_started = true;
        }
#endif

        ESP_LOGI(TAG, "Status page : http://%s/", wifi_ip_string());
        ESP_LOGI(TAG, "MJPEG stream: http://%s%s", wifi_ip_string(), STREAM_PATH);
        display_show_ready(wifi_ip_string(), ota_running_version());

        /* Drop any stale disconnect signal from the association phase. */
        xSemaphoreTake(s_disconnected_sem, 0);
        xSemaphoreTake(s_disconnected_sem, portMAX_DELAY);

        ESP_LOGW(TAG, "Not reachable any more; waiting for the network to come back");
    }
}

static void diagnostics_task(void *arg)
{
    (void)arg;
    const TickType_t period = pdMS_TO_TICKS(DIAGNOSTICS_PERIOD_MS);
    TickType_t last_wake = xTaskGetTickCount();
    bool stats_dumped = false;
    int stats_window = 0;

    while (true) {
        vTaskDelayUntil(&last_wake, period);

        /*
         * One-shot, first tick only: print the driver's own statistics.
         *
         * esp_wifi_statis_dump() is the official ESP-IDF API for this; nothing is
         * hooked, replaced or reconfigured. It is called exactly once per boot so
         * the log contains one unmodified dump, and no counter is disturbed.
         */
        /*
         * Two measurement windows for the RTT comparison, first tick only:
         * dumps at uptime ~60 s, ~90 s and ~120 s give window 1 (60-90 s) and
         * window 2 (90-120 s). The host pings continuously across both and splits
         * its samples at these markers. Same official call, no new instrumentation.
         */
        if (stats_dumped && stats_window < 3) {
            const int64_t up_s = esp_timer_get_time() / 1000000;
            if (up_s >= 60 + 30 * stats_window) {
                stats_window++;
                esp_log_level_set("wifi", ESP_LOG_VERBOSE);
                ESP_LOGI(TAG, "=== STATS WINDOW %d BEGIN (uptime %lld s) ===",
                         stats_window, (long long)up_s);
                esp_err_t win_err = esp_wifi_statis_dump(WIFI_STATIS_ALL);
                esp_log_level_set("wifi", ESP_LOG_INFO);
                ESP_LOGI(TAG, "=== STATS WINDOW %d END (%s) ===",
                         stats_window, esp_err_to_name(win_err));
            }
        }

        if (!stats_dumped) {
            stats_dumped = true;
            /*
             * One-shot, first tick only: the driver's own statistics.
             *
             * The dump produced no output at the default tag level, which is what
             * ESP-IDF's internal dumps usually look like, so the "wifi" tag is
             * raised to VERBOSE for the call and put straight back afterwards.
             * Only the log level is touched - no Wi-Fi/TCP/camera/task/buffer
             * setting is changed, and nothing is hooked or replaced.
             */
            const esp_log_level_t wifi_level_before = esp_log_level_get("wifi");
            esp_log_level_set("wifi", ESP_LOG_VERBOSE);
            ESP_LOGI(TAG, "=== wifi stats BEGIN (level %d -> verbose) ===", (int)wifi_level_before);
            esp_err_t stats_err = esp_wifi_statis_dump(WIFI_STATIS_ALL);
            ESP_LOGI(TAG, "=== wifi stats dump returned %s ===", esp_err_to_name(stats_err));
            esp_log_level_set("wifi", wifi_level_before);
            ESP_LOGI(TAG, "=== wifi stats END (level restored to %d) ===", (int)wifi_level_before);
        }

        /*
         * One line, every 10 s, with everything the stability test asks for:
         * uptime (did it restart?), FPS, frame count (did output stop?), clients,
         * RSSI (did Wi-Fi drop?), free/min heap (memory growth?), plus the
         * firmware version and slot so an OTA reboot is obvious in the log.
         */
        /*
         * Self-healing. The driver re-applies its own power save mode on reconnect,
         * and esp_wifi_set_ps() can return ESP_OK without the mode actually changing,
         * so the mode is checked and corrected here instead of being trusted to the
         * connect path. This is why the log line shows it: if "ps=" ever reads
         * anything but "none", the link is sleeping and the stream will crawl.
         */
        if (wifi_is_connected() && wifi_power_save_mode() != 0) {
            ESP_LOGW(TAG, "modem sleep is on (%s) - turning it off again", wifi_power_save_name());
            wifi_disable_power_save();
        }
        if (wifi_is_connected() && wifi_max_tx_power() != CONFIG_WIFI_MAX_TX_POWER_DBM * 4) {
            /*
             * The provisioning component restores its own transmit power from NVS on
             * every station start, so this is re-checked rather than trusted.
             *
             * The configured cap is deliberately below what the hardware can do: the
             * 3.3 V rail cannot carry 20 dBm transmit current spikes, and a sagging
             * rail corrupts the uplink while the RSSI on this same line still reads
             * strong. See CONFIG_WIFI_MAX_TX_POWER_DBM.
             */
            wifi_force_max_tx_power();
        }

        char link[48];
        wifi_link_summary(link, sizeof(link));

        ESP_LOGI(TAG, "uptime=%us fw=%s fps=%.1f frames=%u clients=%d rssi=%d dBm net=%s ip=%s "
                      "ps=%s tx=%.2fdBm link=\"%s\" disp=%u led=%s gate=%s heap_internal=%u dma_free=%u "
                      "heap_psram=%u heap_min=%u",
                 (unsigned)(esp_timer_get_time() / 1000000),
                 ota_running_version(),
                 (double)stream_fps(),
                 (unsigned)stream_frames_sent(),
                 stream_client_count(),
                 wifi_rssi(),
                 provisioning_state_name(),
                 wifi_ip_string(),
                 wifi_power_save_name(),
                 wifi_max_tx_power() / 4.0,
                 link,
                 (unsigned)display_flush_count(),
#if CONFIG_LED_ENABLE
                 led_status(),
#else
                 "absent",
#endif
#if CONFIG_GATE_ENABLE
                 gate_status(),
#else
                 "absent",
#endif
                 (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                 (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA),
                 (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
                 (unsigned)esp_get_minimum_free_heap_size());
    }
}

void app_main(void)
{
    ESP_LOGI(TAG, "Parking Lot Navigation - ESP32-S3-CAM network camera");
    ESP_LOGI(TAG, "Firmware %s, ESP-IDF %s", ota_running_version(), esp_get_idf_version());

    ota_log_boot_info();

    /*
     * Bring the status panel up first: from here on the device can tell a human
     * what it is doing and, most importantly, which IP address it ended up with.
     * The display is best effort - a missing or broken panel never stops the
     * camera node from working.
     */
    display_init();
    display_show_boot(ota_running_version());

#if CONFIG_QR_SHOW_ON_BOOT
    /*
     * The test QR is shown at every boot for CONFIG_QR_BOOT_SECONDS, then the panel
     * goes back to whatever it was showing. Two reasons for a timed overlay rather
     * than a permanent screen: the panel's original job is still the device IP (see
     * README 12), and someone standing in front of a sandbox with a phone needs a
     * window they can rely on rather than one that ends when Wi-Fi happens to
     * connect.
     *
     * Displayed before Wi-Fi is up on purpose: a QR code that only works on a
     * healthy network is the one thing you cannot check when the network is what
     * you are debugging.
     */
    display_show_qr(NULL);      /* NULL = the payload baked into qr_code.h */
    qr_overlay_start();
#endif

    s_connected_sem = xSemaphoreCreateBinary();
    s_disconnected_sem = xSemaphoreCreateBinary();
    if (s_connected_sem == NULL || s_disconnected_sem == NULL) {
        ESP_LOGE(TAG, "could not create the network semaphores");
        esp_restart();
    }

    /* Does NVS, netif, the default event loop and the Wi-Fi driver. */
    ESP_ERROR_CHECK(wifi_init());

    /*
     * AFTER wifi_init(), and this is not a stylistic preference.
     *
     * wifi_init() is what runs esp_netif_init() and therefore tcpip_init(); before
     * it, any lwIP call is a call into a stack that does not exist yet. Starting the
     * discovery task first looked harmless - the socket binds either way, and until a
     * beacon arrives every address behaves as configured - but the task runs at a
     * higher priority than app_main, so it called socket() immediately, and lwIP
     * answered with
     *
     *     assert failed: tcpip_send_msg_wait_sem (Invalid mbox)
     *
     * which panics before Wi-Fi and therefore before this firmware can call
     * ota_mark_current_valid(). An image that panics before confirming itself is
     * rolled back by the bootloader, the old image finds the newer manifest again,
     * downloads it again, and the device loops between OTA and rollback until the
     * flash wears out or someone stops the server. That is exactly what happened.
     */
#if CONFIG_DISCOVERY_ENABLE
    discovery_start();
#endif

    provisioning_set_hooks(on_connected, on_disconnected, on_before_config_ap);
    provisioning_start_serial_console();
    provisioning_start();

    /*
     * Rollback confirmation for the case where the node is healthy but has no
     * network to join yet: offering the provisioning portal is a working state.
     * The other confirmation point is in network_task(), after the camera is up.
     */
    if (provisioning_in_config_mode()) {
        ota_mark_current_valid();
    }

    ota_start();

#if CONFIG_GATE_ENABLE
    /*
     * Deliberately here and not in network_task(): the gate is local automation,
     * so it must work with no network at all. It never reads the parking state
     * and it publishes nothing.
     */
    gate_start();
#endif

    if (xTaskCreate(network_task, "network_app", 6144, NULL, 5, NULL) != pdPASS) {
        ESP_LOGE(TAG, "could not start the network task");
        esp_restart();
    }
    xTaskCreate(diagnostics_task, "diagnostics", 4096, NULL, 3, NULL);
}
