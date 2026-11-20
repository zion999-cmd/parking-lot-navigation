#include "qr_dynamic.h"

#include <stdio.h>
#include <string.h>

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include <cJSON.h>
#include <esp_heap_caps.h>
#include <esp_http_client.h>
#include <esp_log.h>
#include <esp_timer.h>
#include <sdkconfig.h>

#include "discovery.h"
#include "display.h"

/* Network I/O belongs here, never in the gate or display task. A snapshot is
 * valid only after URL and matrix validation; every failure hides an old QR. */
static const char *TAG = "qr";
#define QR_BODY_CAP (16 * 1024)

static portMUX_TYPE s_lock = portMUX_INITIALIZER_UNLOCKED;
static qr_snapshot_t s_state = { .error = "not fetched yet" };

void qr_dynamic_snapshot(qr_snapshot_t *out)
{
    if (out == NULL) {
        return;
    }
    portENTER_CRITICAL(&s_lock);
    *out = s_state;
    portEXIT_CRITICAL(&s_lock);
}

static void fail(const char *source, const char *reason)
{
    portENTER_CRITICAL(&s_lock);
    s_state.valid = false;
    s_state.last_attempt_us = esp_timer_get_time();
    snprintf(s_state.source_url, sizeof(s_state.source_url), "%s", source);
    snprintf(s_state.error, sizeof(s_state.error), "%s", reason);
    portEXIT_CRITICAL(&s_lock);
    ESP_LOGW(TAG, "QR unavailable: %s", reason);
    display_qr_data_changed();
}

static bool expected_navigation_url(const char *source, const char *payload)
{
    const char *api = strstr(source, "/api/qr");
    if (api == NULL || api[7] != '\0' || strncmp(source, "http://", 7) != 0) {
        return false;
    }
    char expected[QR_URL_MAX];
    int n = snprintf(expected, sizeof(expected), "%.*s/go", (int)(api - source), source);
    return n > 0 && n < (int)sizeof(expected) && strcmp(expected, payload) == 0;
}

static bool parse_document(const char *body, size_t len, const char *source,
                           qr_snapshot_t *next, char *reason, size_t reason_size)
{
    cJSON *root = cJSON_ParseWithLength(body, len);
    if (root == NULL) {
        snprintf(reason, reason_size, "invalid JSON");
        return false;
    }
    bool ok = false;
    const cJSON *schema = cJSON_GetObjectItemCaseSensitive(root, "schema");
    const cJSON *url = cJSON_GetObjectItemCaseSensitive(root, "url");
    const cJSON *size = cJSON_GetObjectItemCaseSensitive(root, "size");
    const cJSON *rows = cJSON_GetObjectItemCaseSensitive(root, "rows");
    if (!cJSON_IsString(schema) || strcmp(schema->valuestring, "parking_qr/1") != 0) {
        snprintf(reason, reason_size, "wrong schema");
        goto done;
    }
    if (!cJSON_IsString(url) || !expected_navigation_url(source, url->valuestring)) {
        snprintf(reason, reason_size, "payload does not point to the serving PC");
        goto done;
    }
    if (!cJSON_IsNumber(size) || !cJSON_IsArray(rows) ||
            cJSON_GetArraySize(rows) != size->valueint) {
        snprintf(reason, reason_size, "size/rows mismatch");
        goto done;
    }

    memset(next, 0, sizeof(*next));
    next->size = size->valueint;
    next->row_bytes = (next->size + 7) / 8;
    if (next->size < QR_MATRIX_MIN_SIZE || next->size > QR_MATRIX_MAX_SIZE ||
            next->row_bytes > QR_MATRIX_ROW_BYTES) {
        snprintf(reason, reason_size, "unsupported matrix size %d", next->size);
        goto done;
    }
    for (int y = 0; y < next->size; y++) {
        const cJSON *row = cJSON_GetArrayItem(rows, y);
        if (!cJSON_IsString(row) || strlen(row->valuestring) != (size_t)next->size) {
            snprintf(reason, reason_size, "bad row %d length", y);
            goto done;
        }
        for (int x = 0; x < next->size; x++) {
            char bit = row->valuestring[x];
            if (bit != '0' && bit != '1') {
                snprintf(reason, reason_size, "bad bit at %d,%d", x, y);
                goto done;
            }
            if (bit == '1') {
                next->rows[y * next->row_bytes + x / 8] |= (uint8_t)(1U << (x % 8));
            }
        }
    }
    if (!qr_matrix_valid(next->rows, next->size, next->row_bytes)) {
        snprintf(reason, reason_size, "matrix finder/padding check failed");
        goto done;
    }
    snprintf(next->url, sizeof(next->url), "%s", url->valuestring);
    snprintf(next->source_url, sizeof(next->source_url), "%s", source);
    next->valid = true;
    next->last_attempt_us = esp_timer_get_time();
    next->last_success_us = next->last_attempt_us;
    ok = true;

done:
    cJSON_Delete(root);
    return ok;
}

static void fetch_once(const char *source)
{
    char body_error[96];
    esp_http_client_config_t config = {
        .url = source,
        .timeout_ms = CONFIG_QR_HTTP_TIMEOUT_MS,
        .disable_auto_redirect = true,
    };
    esp_http_client_handle_t client = esp_http_client_init(&config);
    if (client == NULL) {
        fail(source, "HTTP client allocation failed");
        return;
    }
    char *body = heap_caps_malloc(QR_BODY_CAP, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (body == NULL) {
        body = malloc(QR_BODY_CAP);
    }
    if (body == NULL) {
        esp_http_client_cleanup(client);
        fail(source, "no memory for response");
        return;
    }

    esp_err_t err = esp_http_client_open(client, 0);
    if (err != ESP_OK) {
        snprintf(body_error, sizeof(body_error), "connect: %s", esp_err_to_name(err));
        fail(source, body_error);
        goto done;
    }
    int content_len = esp_http_client_fetch_headers(client);
    int status = esp_http_client_get_status_code(client);
    if (status != 200) {
        snprintf(body_error, sizeof(body_error), "HTTP %d", status);
        fail(source, body_error);
        goto done;
    }
    if (content_len > QR_BODY_CAP - 1) {
        fail(source, "response too large");
        goto done;
    }
    size_t total = 0;
    while (total < QR_BODY_CAP - 1) {
        int n = esp_http_client_read(client, body + total, QR_BODY_CAP - 1 - total);
        if (n < 0) {
            fail(source, "response read failed");
            goto done;
        }
        if (n == 0) {
            break;
        }
        total += (size_t)n;
        if (content_len > 0 && total >= (size_t)content_len) {
            break;
        }
    }
    if (total == 0) {
        fail(source, "empty response");
        goto done;
    }

    qr_snapshot_t next;
    if (!parse_document(body, total, source, &next, body_error, sizeof(body_error))) {
        fail(source, body_error);
        goto done;
    }

    bool changed;
    portENTER_CRITICAL(&s_lock);
    changed = !s_state.valid || strcmp(s_state.url, next.url) != 0;
    next.last_success_us = esp_timer_get_time();
    s_state = next;
    portEXIT_CRITICAL(&s_lock);
    if (changed) {
        ESP_LOGI(TAG, "QR updated: %s (%d x %d)", next.url, next.size, next.size);
        display_qr_data_changed();
    }

done:
    free(body);
    esp_http_client_close(client);
    esp_http_client_cleanup(client);
}

static void qr_task(void *arg)
{
    (void)arg;
    char previous_source[QR_SOURCE_URL_MAX] = "";
    while (true) {
        char source[QR_SOURCE_URL_MAX];
        discovery_url(CONFIG_QR_DATA_URL, source, sizeof(source));
        if (strcmp(previous_source, source) != 0) {
            /* A new PC must never inherit the old PC's visible QR while its first
             * request is still in flight. */
            if (previous_source[0] != '\0') {
                fail(source, "serving PC changed");
            }
            snprintf(previous_source, sizeof(previous_source), "%s", source);
        }
        fetch_once(source);
        vTaskDelay(pdMS_TO_TICKS(CONFIG_QR_POLL_MS));
    }
}

bool qr_dynamic_start(void)
{
    if (xTaskCreate(qr_task, "qr_dynamic", 6144, NULL, 2, NULL) != pdPASS) {
        ESP_LOGW(TAG, "could not start QR worker; no dynamic QR will be shown");
        return false;
    }
    return true;
}
