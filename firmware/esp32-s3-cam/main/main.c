#include <stdio.h>

#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <freertos/task.h>

#include <esp_heap_caps.h>
#include <esp_idf_version.h>
#include <esp_log.h>
#include <esp_system.h>
#include <esp_timer.h>

#include "camera.h"
#include "display.h"
#include "ota.h"
#include "provisioning.h"
#include "stream.h"
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

        ESP_LOGI(TAG, "Status page : http://%s/", wifi_ip_string());
        ESP_LOGI(TAG, "MJPEG stream: http://%s%s", wifi_ip_string(), STREAM_PATH);
        display_show_ready(wifi_ip_string());

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

    while (true) {
        vTaskDelayUntil(&last_wake, period);

        /*
         * One line, every 10 s, with everything the stability test asks for:
         * uptime (did it restart?), FPS, frame count (did output stop?), clients,
         * RSSI (did Wi-Fi drop?), free/min heap (memory growth?), plus the
         * firmware version and slot so an OTA reboot is obvious in the log.
         */
        ESP_LOGI(TAG, "uptime=%us fw=%s fps=%.1f frames=%u clients=%d rssi=%d dBm "
                      "net=%s ip=%s heap_internal=%u heap_psram=%u heap_min=%u",
                 (unsigned)(esp_timer_get_time() / 1000000),
                 ota_running_version(),
                 (double)stream_fps(),
                 (unsigned)stream_frames_sent(),
                 stream_client_count(),
                 wifi_rssi(),
                 provisioning_state_name(),
                 wifi_ip_string(),
                 (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
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

    s_connected_sem = xSemaphoreCreateBinary();
    s_disconnected_sem = xSemaphoreCreateBinary();
    if (s_connected_sem == NULL || s_disconnected_sem == NULL) {
        ESP_LOGE(TAG, "could not create the network semaphores");
        esp_restart();
    }

    /* Does NVS, netif, the default event loop and the Wi-Fi driver. */
    ESP_ERROR_CHECK(wifi_init());

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

    if (xTaskCreate(network_task, "network_app", 6144, NULL, 5, NULL) != pdPASS) {
        ESP_LOGE(TAG, "could not start the network task");
        esp_restart();
    }
    xTaskCreate(diagnostics_task, "diagnostics", 4096, NULL, 3, NULL);
}
