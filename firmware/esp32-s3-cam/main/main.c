#include <stdio.h>

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include <esp_heap_caps.h>
#include <esp_idf_version.h>
#include <esp_log.h>
#include <esp_system.h>
#include <esp_timer.h>
#include <nvs_flash.h>

#include "camera.h"
#include "stream.h"
#include "wifi.h"

static const char *TAG = "app";

/* How often the runtime health line is printed. */
#define DIAGNOSTICS_PERIOD_MS 10000

/* Pause before rebooting when a mandatory subsystem could not be brought up. */
#define FAILURE_RESTART_DELAY_MS 5000

static void diagnostics_task(void *arg)
{
    const TickType_t period = pdMS_TO_TICKS(DIAGNOSTICS_PERIOD_MS);
    TickType_t last_wake = xTaskGetTickCount();

    while (true) {
        vTaskDelayUntil(&last_wake, period);

        /*
         * One line, every 10 s, with everything the P0001 stability test asks
         * for: uptime (did it restart?), FPS, frame count (did output stop?),
         * clients, RSSI (did Wi-Fi drop?), and free/min heap (memory growth?).
         */
        ESP_LOGI(TAG, "uptime=%us fps=%.1f frames=%u clients=%d rssi=%d dBm "
                      "heap_internal=%u heap_psram=%u heap_min=%u",
                 (unsigned)(esp_timer_get_time() / 1000000),
                 (double)stream_fps(),
                 (unsigned)stream_frames_sent(),
                 stream_client_count(),
                 wifi_rssi(),
                 (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                 (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
                 (unsigned)esp_get_minimum_free_heap_size());
    }
}

static esp_err_t init_nvs(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_LOGW(TAG, "NVS needs to be erased (%s), erasing", esp_err_to_name(err));
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    return err;
}

static void restart_after_failure(const char *what)
{
    ESP_LOGE(TAG, "%s could not be initialised - restarting in %d s",
             what, FAILURE_RESTART_DELAY_MS / 1000);
    vTaskDelay(pdMS_TO_TICKS(FAILURE_RESTART_DELAY_MS));
    esp_restart();
}

void app_main(void)
{
    ESP_LOGI(TAG, "Parking Lot Navigation - ESP32-S3-CAM network camera (P0001)");
    ESP_LOGI(TAG, "Built with ESP-IDF %s", esp_get_idf_version());

    ESP_ERROR_CHECK(init_nvs());

    if (wifi_sta_start() != ESP_OK) {
        /*
         * Wi-Fi is the only external dependency, so a failure here is not fatal:
         * the driver keeps retrying in the background and the stream server
         * binds to 0.0.0.0, so the node becomes reachable the moment a lease is
         * obtained. Being reachable-later beats a reboot loop, and it keeps the
         * camera diagnostics below visible.
         */
        ESP_LOGW(TAG, "WiFi not connected yet - continuing anyway, the driver "
                      "keeps retrying in the background");
    }

    if (camera_init() != ESP_OK) {
        restart_after_failure("Camera");
    }

    if (stream_server_start() != ESP_OK) {
        restart_after_failure("HTTP stream server");
    }

    ESP_LOGI(TAG, "Status page : http://%s/", wifi_ip_string());
    ESP_LOGI(TAG, "MJPEG stream: http://%s%s", wifi_ip_string(), STREAM_PATH);

    xTaskCreate(diagnostics_task, "diagnostics", 4096, NULL, 3, NULL);
}
