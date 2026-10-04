#include "ota.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include <cJSON.h>
#include <esp_app_desc.h>
#include <esp_crt_bundle.h>
#include <esp_http_client.h>
#include <esp_log.h>
#include <esp_ota_ops.h>
#include <esp_system.h>
#include <mbedtls/sha256.h>
#include <sdkconfig.h>

#include "display.h"
#include "wifi.h"
#include "version.h"

/*
 * Thin OTA client - device pull model.
 *
 *   1. GET the manifest (CONFIG_OTA_MANIFEST_URL)
 *   2. compare its "version" against the running version as integers
 *   3. if newer: GET the firmware, hashing the stream with SHA-256
 *   4. compare the digest with the manifest, then esp_ota_write into the
 *      inactive slot and esp_ota_set_boot_partition
 *
 * The esp_ota_* call sequence and the mark-valid check are the ones the
 * reference project uses in main/ota.cc (MarkCurrentVersionValid / Upgrade);
 * only the cloud protocol around them is gone, replaced by the manifest above.
 *
 * Nothing here touches the running slot, so a failure at any step leaves the
 * currently bootable firmware untouched.
 */

static const char *TAG = "ota";

#define OTA_DOWNLOAD_BUF 4096
#define OTA_MANIFEST_MAX 1024

/*
 * Wait this long after boot before the automatic check, so the camera and the
 * stream server are up and the network has settled.
 */
#define OTA_BOOT_CHECK_DELAY_MS 30000

static TaskHandle_t s_task = NULL;
static volatile bool s_in_progress = false;
static char s_available_version[16] = "-";
static char s_last_result[96] = "not checked yet";

const char *ota_running_version(void)
{
    return esp_app_get_description()->version;
}

const char *ota_available_version(void)
{
    return s_available_version;
}

const char *ota_last_result(void)
{
    return s_last_result;
}

bool ota_in_progress(void)
{
    return s_in_progress;
}

static void set_result(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(s_last_result, sizeof(s_last_result), fmt, ap);
    va_end(ap);
    ESP_LOGI(TAG, "%s", s_last_result);
}

/* ------------------------------------------------------------------------- */
/* Rollback confirmation                                                      */
/* ------------------------------------------------------------------------- */

static const char *ota_state_name(esp_ota_img_states_t state)
{
    switch (state) {
    case ESP_OTA_IMG_NEW:            return "NEW";
    case ESP_OTA_IMG_PENDING_VERIFY: return "PENDING_VERIFY";
    case ESP_OTA_IMG_VALID:          return "VALID";
    case ESP_OTA_IMG_INVALID:        return "INVALID";
    case ESP_OTA_IMG_ABORTED:        return "ABORTED";
    case ESP_OTA_IMG_UNDEFINED:      return "UNDEFINED";
    default:                         return "?";
    }
}

void ota_log_boot_info(void)
{
    const esp_partition_t *running = esp_ota_get_running_partition();
    const esp_partition_t *next = esp_ota_get_next_update_partition(NULL);

    esp_ota_img_states_t state = ESP_OTA_IMG_UNDEFINED;
    esp_ota_get_state_partition(running, &state);

    ESP_LOGI(TAG, "running slot=%s (%s) state=%s, next OTA target=%s",
             running->label, ota_running_version(), ota_state_name(state),
             next ? next->label : "none");

    /* Print the configured server up front. A wrong address here used to be
     * invisible until an OTA check timed out 30 s after boot. */
    ESP_LOGI(TAG, "manifest URL  = %s", CONFIG_OTA_MANIFEST_URL);
}

void ota_mark_current_valid(void)
{
    const esp_partition_t *running = esp_ota_get_running_partition();

    if (strcmp(running->label, "factory") == 0) {
        return;
    }

    esp_ota_img_states_t state;
    if (esp_ota_get_state_partition(running, &state) != ESP_OK) {
        ESP_LOGE(TAG, "could not read the state of %s", running->label);
        return;
    }

    if (state == ESP_OTA_IMG_PENDING_VERIFY) {
        ESP_LOGI(TAG, "Firmware %s on %s is working, cancelling rollback",
                 ota_running_version(), running->label);
        esp_err_t err = esp_ota_mark_app_valid_cancel_rollback();
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "esp_ota_mark_app_valid_cancel_rollback failed: %s",
                     esp_err_to_name(err));
        }
    }
}

/* ------------------------------------------------------------------------- */
/* Version comparison is in version.c so it can be unit tested on a PC.       */
/* ------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------- */
/* Manifest                                                                   */
/* ------------------------------------------------------------------------- */

typedef struct {
    char version[16];
    char firmware_url[192];
    char sha256[65];
    size_t size;
} ota_manifest_t;

static bool http_get_to_buffer(const char *url, char *buf, size_t buf_len, size_t *out_len)
{
    esp_http_client_config_t cfg = {};
    cfg.url = url;
    cfg.timeout_ms = 15000;
    cfg.buffer_size = 1024;
    /*
     * Only consulted when the URL is https://; for plain HTTP it is ignored, so
     * the LAN development setup needs no PKI work at all.
     */
    cfg.crt_bundle_attach = esp_crt_bundle_attach;

    esp_http_client_handle_t client = esp_http_client_init(&cfg);
    if (client == NULL) {
        return false;
    }

    bool ok = false;
    if (esp_http_client_open(client, 0) == ESP_OK) {
        int status = esp_http_client_get_status_code(client);
        if (status != 200) {
            ESP_LOGE(TAG, "GET %s -> HTTP %d", url, status);
        } else {
            int total = 0;
            while (total < (int)buf_len - 1) {
                int n = esp_http_client_read(client, buf + total, buf_len - 1 - total);
                if (n <= 0) {
                    break;
                }
                total += n;
            }
            buf[total] = '\0';
            *out_len = (size_t)total;
            ok = total > 0;
        }
    } else {
        ESP_LOGE(TAG, "could not open %s", url);
    }

    esp_http_client_close(client);
    esp_http_client_cleanup(client);
    return ok;
}

static bool fetch_manifest(ota_manifest_t *m)
{
    char *body = malloc(OTA_MANIFEST_MAX);
    if (body == NULL) {
        return false;
    }

    memset(m, 0, sizeof(*m));
    bool ok = false;
    size_t len = 0;

    if (!http_get_to_buffer(CONFIG_OTA_MANIFEST_URL, body, OTA_MANIFEST_MAX, &len)) {
        set_result("manifest: fetch failed");
        goto done;
    }

    cJSON *root = cJSON_Parse(body);
    if (root == NULL) {
        set_result("manifest: invalid JSON");
        goto done;
    }

    cJSON *jv = cJSON_GetObjectItemCaseSensitive(root, "version");
    cJSON *ju = cJSON_GetObjectItemCaseSensitive(root, "firmware_url");
    cJSON *js = cJSON_GetObjectItemCaseSensitive(root, "sha256");
    cJSON *jsz = cJSON_GetObjectItemCaseSensitive(root, "size");

    if (!cJSON_IsString(jv) || !cJSON_IsString(ju) || !cJSON_IsString(js)) {
        set_result("manifest: missing version/firmware_url/sha256");
        cJSON_Delete(root);
        goto done;
    }

    snprintf(m->version, sizeof(m->version), "%s", jv->valuestring);
    snprintf(m->firmware_url, sizeof(m->firmware_url), "%s", ju->valuestring);
    snprintf(m->sha256, sizeof(m->sha256), "%s", js->valuestring);
    m->size = cJSON_IsNumber(jsz) ? (size_t)jsz->valuedouble : 0;

    cJSON_Delete(root);
    ok = true;

done:
    free(body);
    return ok;
}

/* Turn a manifest-relative firmware_url into an absolute one. */
static void make_firmware_url(const char *base_manifest_url, const char *rel,
                              char *out, size_t out_len)
{
    if (strncmp(rel, "http://", 7) == 0 || strncmp(rel, "https://", 8) == 0) {
        snprintf(out, out_len, "%s", rel);
        return;
    }

    /* base is scheme://host[:port]/manifest.json -> keep up to the third '/' */
    const char *scheme = strstr(base_manifest_url, "://");
    const char *host = scheme ? scheme + 3 : base_manifest_url;
    const char *slash = strchr(host, '/');
    size_t prefix = slash ? (size_t)(slash - base_manifest_url) : strlen(base_manifest_url);

    snprintf(out, out_len, "%.*s%s", (int)prefix, base_manifest_url, rel);
}

/* ------------------------------------------------------------------------- */
/* Download + flash                                                           */
/* ------------------------------------------------------------------------- */

static bool download_and_flash(const char *url, const ota_manifest_t *m)
{
    esp_http_client_config_t cfg = {};
    cfg.url = url;
    cfg.timeout_ms = 30000;
    cfg.buffer_size = OTA_DOWNLOAD_BUF;
    cfg.keep_alive_enable = false;
    cfg.crt_bundle_attach = esp_crt_bundle_attach;

    esp_http_client_handle_t client = esp_http_client_init(&cfg);
    if (client == NULL) {
        set_result("download: client init failed");
        return false;
    }

    uint8_t *buf = malloc(OTA_DOWNLOAD_BUF);
    if (buf == NULL) {
        esp_http_client_cleanup(client);
        set_result("download: out of memory");
        return false;
    }

    const esp_partition_t *target = esp_ota_get_next_update_partition(NULL);
    if (target == NULL) {
        free(buf);
        esp_http_client_cleanup(client);
        set_result("download: no OTA partition");
        return false;
    }

    bool ok = false;
    esp_ota_handle_t handle = 0;
    bool ota_started = false;
    mbedtls_sha256_context sha;
    size_t written = 0;
    int last_logged = -1;

    mbedtls_sha256_init(&sha);
    mbedtls_sha256_starts(&sha, 0);

    if (esp_http_client_open(client, 0) != ESP_OK) {
        set_result("download: connection failed");
        goto cleanup;
    }
    if (esp_http_client_get_status_code(client) != 200) {
        set_result("download: HTTP %d", esp_http_client_get_status_code(client));
        goto cleanup;
    }

    ESP_LOGI(TAG, "downloading into slot %s (%u bytes expected)",
             target->label, (unsigned)m->size);

    if (esp_ota_begin(target, m->size ? m->size : OTA_SIZE_UNKNOWN, &handle) != ESP_OK) {
        set_result("download: esp_ota_begin failed");
        goto cleanup;
    }
    ota_started = true;

    while (true) {
        int n = esp_http_client_read(client, (char *)buf, OTA_DOWNLOAD_BUF);
        if (n < 0) {
            set_result("download: read error");
            goto cleanup;
        }
        if (n == 0) {
            break;
        }

        mbedtls_sha256_update(&sha, buf, (size_t)n);

        esp_err_t err = esp_ota_write(handle, buf, (size_t)n);
        if (err != ESP_OK) {
            set_result("download: esp_ota_write failed (%s)", esp_err_to_name(err));
            goto cleanup;
        }

        written += (size_t)n;
        if (m->size > 0) {
            int bucket = (int)((written * 5) / m->size);
            if (bucket != last_logged) {
                last_logged = bucket;
                int pct = (int)((written * 100) / m->size);
                ESP_LOGI(TAG, "downloaded %u / %u bytes (%d%%)",
                         (unsigned)written, (unsigned)m->size, pct);
                display_show_ota_progress(ota_running_version(), m->version, pct);
            }
        }
    }

    if (m->size > 0 && written != m->size) {
        set_result("download: short read, got %u of %u bytes",
                   (unsigned)written, (unsigned)m->size);
        goto cleanup;
    }

    /* Integrity check before anything is committed (SC-13). */
    {
        unsigned char digest[32];
        char hex[65];

        mbedtls_sha256_finish(&sha, digest);
        for (int i = 0; i < 32; i++) {
            snprintf(hex + i * 2, 3, "%02x", digest[i]);
        }
        hex[64] = '\0';

        if (strcasecmp(hex, m->sha256) != 0) {
            set_result("SHA256 mismatch: expected %.16s... got %.16s...", m->sha256, hex);
            goto cleanup;
        }
        ESP_LOGI(TAG, "SHA256 verified: %s", hex);
    }

    if (esp_ota_end(handle) != ESP_OK) {
        ota_started = false;
        set_result("esp_ota_end failed (image rejected)");
        goto cleanup;
    }
    ota_started = false;

    if (esp_ota_set_boot_partition(target) != ESP_OK) {
        set_result("esp_ota_set_boot_partition failed");
        goto cleanup;
    }

    set_result("OTA ok: %s -> %s, rebooting", ota_running_version(), m->version);
    ok = true;

cleanup:
    if (ota_started) {
        esp_ota_abort(handle);
    }
    mbedtls_sha256_free(&sha);
    esp_http_client_close(client);
    esp_http_client_cleanup(client);
    free(buf);
    return ok;
}

/* ------------------------------------------------------------------------- */
/* Worker task                                                                */
/* ------------------------------------------------------------------------- */

static void check_once(void)
{
    ota_manifest_t manifest;

    if (!wifi_is_connected()) {
        set_result("skipped: no network");
        return;                     /* no overlay was shown, nothing to undo */
    }

    ESP_LOGI(TAG, "checking %s (running %s)", CONFIG_OTA_MANIFEST_URL, ota_running_version());
    display_show_ota_checking();

    if (!fetch_manifest(&manifest)) {
        goto done;                  /* reason is already in s_last_result */
    }
    snprintf(s_available_version, sizeof(s_available_version), "%s", manifest.version);

    if (version_cmp(manifest.version, ota_running_version()) <= 0) {
        set_result("up to date (running %s, server %s)",
                   ota_running_version(), manifest.version);
        goto done;
    }

    ESP_LOGI(TAG, "update available: %s -> %s", ota_running_version(), manifest.version);

    char url[256];
    make_firmware_url(CONFIG_OTA_MANIFEST_URL, manifest.firmware_url, url, sizeof(url));

    if (download_and_flash(url, &manifest)) {
        display_show_ota_result(true);
        ESP_LOGW(TAG, "rebooting into %s", manifest.version);
        vTaskDelay(pdMS_TO_TICKS(1500));
        esp_restart();
    }

    /*
     * Download, SHA256 or esp_ota_* failed. The running firmware is untouched,
     * so tell the person looking at the device that nothing was lost, give them
     * a moment to read it, and then go back to the normal screen. The detailed
     * reason stays in the serial log.
     */
    display_show_ota_result(false);
    vTaskDelay(pdMS_TO_TICKS(4000));

 done:
    /*
     * Every non-reboot exit restores the previous screen. Without this the panel
     * would sit on "Checking update..." indefinitely and hide the device IP -
     * which is the one thing the status display exists to show.
     */
    vTaskDelay(pdMS_TO_TICKS(800));
    display_restore();
}

static void ota_task(void *arg)
{
    (void)arg;

    /* One automatic check shortly after boot, then only on request. */
    ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(OTA_BOOT_CHECK_DELAY_MS));

    while (true) {
        s_in_progress = true;
        check_once();
        s_in_progress = false;

        /* Sleep until someone asks for another check. */
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    }
}

void ota_start(void)
{
    if (xTaskCreate(ota_task, "ota", 8192, NULL, 4, &s_task) != pdPASS) {
        ESP_LOGE(TAG, "could not start the OTA task");
        s_task = NULL;
    }
}

bool ota_request_check(void)
{
    if (s_task == NULL || s_in_progress) {
        return false;
    }
    xTaskNotifyGive(s_task);
    return true;
}
