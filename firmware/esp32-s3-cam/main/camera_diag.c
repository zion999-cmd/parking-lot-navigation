/* Passive diagnostic taps. Do not change sensor registers or capture settings. */
#include "camera_diag.h"
#include "camera.h"
#include "cam_hal.h"
#include "driver/gpio.h"
#include "esp_timer.h"
#include "esp_app_desc.h"
#include "cJSON.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static portMUX_TYPE edge_mux = portMUX_INITIALIZER_UNLOCKED;
static struct {
    uint32_t rising, falling, intervals, min_us, max_us;
    int64_t last_us, total_us;
} edges;
static int pin_number;

static void vsync_edge(void *arg)
{
    int level = gpio_get_level(pin_number);
    int64_t now = esp_timer_get_time();
    portENTER_CRITICAL_ISR(&edge_mux);
    if (level) {
        edges.rising++;
        if (edges.last_us) {
            uint32_t dt = (uint32_t)(now - edges.last_us);
            if (!edges.intervals || dt < edges.min_us) edges.min_us = dt;
            if (dt > edges.max_us) edges.max_us = dt;
            edges.total_us += dt;
            edges.intervals++;
        }
        edges.last_us = now;
    } else {
        edges.falling++;
    }
    portEXIT_CRITICAL_ISR(&edge_mux);
}

esp_err_t camera_diag_start(int vsync_pin)
{
    pin_number = vsync_pin;
    /* The camera already configured this input. Leave its GPIO matrix routing alone. */
    esp_err_t err = gpio_install_isr_service(0);
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) return err;
    err = gpio_set_intr_type(vsync_pin, GPIO_INTR_ANYEDGE);
    if (err != ESP_OK) return err;
    return gpio_isr_handler_add(vsync_pin, vsync_edge, NULL);
}

static esp_err_t diagnostic_handler(httpd_req_t *req)
{
    uint32_t counts[CAM_DIAG_COUNT];
    cam_diag_snapshot(counts);
    portENTER_CRITICAL(&edge_mux);
    typeof(edges) e = edges;
    portEXIT_CRITICAL(&edge_mux);
    const char *names[CAM_DIAG_COUNT] = {
        "vsync_isr", "dma_isr", "vsync_queue_overflow", "dma_queue_overflow",
        "capture_start", "start_failed", "frame_complete", "empty_vsync",
        "fb_overflow", "dma_overflow", "bad_soi", "bad_eoi",
        "queue_replaced", "queue_error", "delivered", "take_timeout", "null_dma"
    };
    cJSON *root = cJSON_CreateObject();
    if (!root) return ESP_ERR_NO_MEM;
    cJSON_AddStringToObject(root, "firmware", esp_app_get_description()->version);
    cJSON_AddNumberToObject(root, "sample_us", esp_timer_get_time());
    cJSON_AddNumberToObject(root, "gpio", pin_number);
    cJSON_AddNumberToObject(root, "gpio_rising", e.rising);
    cJSON_AddNumberToObject(root, "gpio_falling", e.falling);
    cJSON_AddNumberToObject(root, "gpio_intervals", e.intervals);
    cJSON_AddNumberToObject(root, "gpio_total_interval_us", e.total_us);
    cJSON_AddNumberToObject(root, "gpio_min_interval_us", e.min_us);
    cJSON_AddNumberToObject(root, "gpio_max_interval_us", e.max_us);
    cJSON *driver = cJSON_AddObjectToObject(root, "driver");
    for (int i = 0; i < CAM_DIAG_COUNT; i++) cJSON_AddNumberToObject(driver, names[i], counts[i]);
    /* Sequential SCCB reads, not an atomic frame snapshot. -1 remains a read error. */
    const uint16_t regs[] = {
        0x3000,0x3001,0x3002,0x3003,0x3004,0x3005,0x3006,0x3007,
        0x3034,0x3035,0x3036,0x3037,0x3039,0x3103,0x3108,
        0x3500,0x3501,0x3502,0x3503,0x350c,0x350d,
        0x380c,0x380d,0x380e,0x380f,0x3814,0x3815,0x3820,0x3821,0x3824,
        0x3a00,0x3a02,0x3a03,0x3a05,0x3a08,0x3a09,0x3a0a,0x3a0b,0x3a0d,0x3a0e,0x3a14,0x3a15,
        0x4201,0x4202,0x4400,0x4401,0x4402,0x4407,0x4417,
        0x4600,0x4602,0x4603,0x4604,0x4605,0x460c,
        0x4713,0x471d,0x471f,0x4740,0x4901,0x4902,0x5001
    };
    sensor_t *sensor = esp_camera_sensor_get();
    cJSON *r = cJSON_AddObjectToObject(root, "regs");
    for (unsigned i = 0; i < sizeof(regs)/sizeof(regs[0]); i++) {
        char key[5];
        snprintf(key, sizeof(key), "%04x", regs[i]);
        cJSON_AddNumberToObject(r, key, sensor ? sensor->get_reg(sensor, regs[i], 0xff) : -1);
    }
    char *body = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    if (!body) return ESP_ERR_NO_MEM;
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    esp_err_t err = httpd_resp_sendstr(req, body);
    free(body);
    return err;
}

void camera_diag_edges(uint32_t *rising, uint32_t *intervals, uint64_t *total_us)
{
    portENTER_CRITICAL(&edge_mux);
    if (rising) *rising = edges.rising;
    if (intervals) *intervals = edges.intervals;
    if (total_us) *total_us = edges.total_us;
    portEXIT_CRITICAL(&edge_mux);
}

esp_err_t camera_diag_register(httpd_handle_t server)
{
    const httpd_uri_t uri = {.uri="/api/camera-diag", .method=HTTP_GET, .handler=diagnostic_handler};
    return httpd_register_uri_handler(server, &uri);
}
