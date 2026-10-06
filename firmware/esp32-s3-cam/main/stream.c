#include "stream.h"

#include <stdio.h>
#include <string.h>

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include <esp_app_desc.h>
#include <esp_camera.h>
#include <esp_http_server.h>
#include <esp_log.h>
#include <esp_ota_ops.h>
#include <esp_timer.h>
#include <sdkconfig.h>

#include "camera.h"
#include "ota.h"
#include "provisioning.h"
#include "wifi.h"

static const char *TAG = "stream";

#define PART_BOUNDARY "parkinglotcamboundary"

static const char *STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
/*
 * Boundary and part header in one write.
 *
 * Nagle holds a small segment back while earlier data is still unacknowledged, and a
 * 40 KB JPEG is always in flight, so each of the boundary and part-header writes used
 * to cost a round trip. Sending them as a single prefix halves the number of small
 * writes; the JPEG itself is large enough to go out immediately.
 */
#define STREAM_PREFIX_FMT \
    "\r\n--" PART_BOUNDARY "\r\n" \
    "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n"

/*
 * Where does the time go?
 *
 * The board delivers ~4 fps where ~13 was expected, and from outside there was no way
 * to tell whether the camera or the network was stalling. Each frame is therefore
 * split into "waiting for the sensor" (grab) and "pushing it down the socket" (send),
 * and the JPEG size is tracked too - because a link caps the byte rate, so the frame
 * rate is really bytes per second divided by the JPEG size. Both are reported.
 *
 * The counters are locals of the streaming task, not file statics: two clients on the
 * stream at once would otherwise corrupt each other's numbers.
 */
#define STREAM_STATS_EVERY 50

/* Stack for the per-client streaming task. */
#define STREAM_TASK_STACK 4096
#define STREAM_TASK_PRIO 5

static httpd_handle_t s_server = NULL;
static uint32_t s_frames_sent = 0;
static volatile float s_fps = 0.0f;
static int64_t s_last_frame_us = 0;
static volatile int s_clients = 0;

/* ------------------------------------------------------------------------- */
/* Status page                                                                */
/* ------------------------------------------------------------------------- */

static esp_err_t index_handler(httpd_req_t *req)
{
    static char page[3072];

    const esp_partition_t *running = esp_ota_get_running_partition();

    int n = snprintf(page, sizeof(page),
        "<!DOCTYPE html>\n"
        "<html><head><meta charset=\"utf-8\">"
        "<title>Parking Lot Camera</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:1.5em;}"
        "img{max-width:100%%;height:auto;border:1px solid #ccc;}"
        "td{padding:2px 12px 2px 0;}td:first-child{color:#666;}"
        "button{padding:6px 12px;}</style></head>\n"
        "<body>\n"
        "<h1>Parking Lot Camera</h1>\n"
        "<table>\n"
        "<tr><td>IP</td><td>%s</td></tr>\n"
        "<tr><td>Sensor</td><td>%s</td></tr>\n"
        "<tr><td>Resolution</td><td>%s</td></tr>\n"
        "<tr><td>Format</td><td>%s (quality %d)</td></tr>\n"
        "<tr><td>Stream</td><td><a href=\"%s\">%s</a></td></tr>\n"
        "<tr><td>FPS</td><td>%.1f</td></tr>\n"
        "<tr><td>Uptime</td><td>%u s</td></tr>\n"
        "<tr><td>Clients</td><td>%d</td></tr>\n"
        "<tr><td>WiFi</td><td>%s, RSSI %d dBm</td></tr>\n"
        "<tr><td>Firmware</td><td>%s (slot %s)</td></tr>\n"
        "<tr><td>Net state</td><td>%s</td></tr>\n"
        "<tr><td>OTA</td><td>%s%s</td></tr>\n"
        "<tr><td>OTA server</td><td>%s</td></tr>\n"
        "</table>\n"
        "<p><img src=\"%s\" alt=\"live stream\"></p>\n"
        "<p><button onclick=\"fetch('%s',{method:'POST'})\">Check for firmware update</button></p>\n"
        "</body></html>\n",
        wifi_ip_string(),
        camera_sensor_name(),
        camera_resolution_name(),
        camera_pixel_format_name(),
        camera_jpeg_quality(),
        STREAM_PATH, STREAM_PATH,
        (double)stream_fps(),
        (unsigned)(esp_timer_get_time() / 1000000),
        s_clients,
        wifi_ssid(),
        wifi_rssi(),
        esp_app_get_description()->version, running->label,
        provisioning_state_name(),
        ota_in_progress() ? "checking... " : "",
        ota_last_result(),
        /* Shown on purpose: a wrong OTA server address is then obvious from the
         * status page instead of only showing up as a timeout in the log. */
        CONFIG_OTA_MANIFEST_URL,
        STREAM_PATH,
        STREAM_OTA_CHECK_PATH);

    if (n <= 0 || (size_t)n >= sizeof(page)) {
        return httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "page too long");
    }

    httpd_resp_set_type(req, "text/html");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    return httpd_resp_send(req, page, n);
}

/* ------------------------------------------------------------------------- */
/* Manual OTA trigger                                                         */
/* ------------------------------------------------------------------------- */

static esp_err_t ota_check_handler(httpd_req_t *req)
{
    /*
     * Only wakes the OTA worker; the download happens there so this request
     * returns immediately. 202 = accepted, 409 = a check is already running.
     */
    bool accepted = ota_request_check();

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    httpd_resp_set_status(req, accepted ? "202 Accepted" : "409 Conflict");

    char body[160];
    int n = snprintf(body, sizeof(body),
                     "{\"accepted\":%s,\"running_version\":\"%s\",\"last_result\":\"%s\"}\n",
                     accepted ? "true" : "false", ota_running_version(), ota_last_result());
    return httpd_resp_send(req, body, n);
}

/* ------------------------------------------------------------------------- */
/* MJPEG stream                                                               */
/* ------------------------------------------------------------------------- */

static void stream_task(void *arg)
{
    httpd_req_t *req = (httpd_req_t *)arg;
    esp_err_t res;

    res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
    if (res == ESP_OK) {
        httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
        httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    }

    ESP_LOGI(TAG, "client connected (%d active)", s_clients);

    uint32_t stat_frames = 0;
    int64_t grab_us = 0, send_us = 0, grab_max_us = 0, send_max_us = 0;
    size_t jpeg_sum = 0, jpeg_min = 0, jpeg_max = 0;
    int64_t window_start_us = esp_timer_get_time();

    while (res == ESP_OK) {
        int64_t t_grab0 = esp_timer_get_time();
        camera_fb_t *fb = esp_camera_fb_get();
        int64_t t_grab1 = esp_timer_get_time();
        if (fb == NULL) {
            ESP_LOGE(TAG, "esp_camera_fb_get() failed, closing stream");
            res = ESP_FAIL;
            break;
        }

        if (fb->format != PIXFORMAT_JPEG) {
            ESP_LOGE(TAG, "unexpected pixel format %d, closing stream", fb->format);
            esp_camera_fb_return(fb);
            res = ESP_FAIL;
            break;
        }

        /*
         * fb->len must be read before the buffer goes back to the driver. After
         * esp_camera_fb_return() the DMA is free to start filling that buffer with
         * the next frame, so reading fb->len (or fb->buf) afterwards is a race.
         */
        size_t len = fb->len;

        char prefix[128];
        int plen = snprintf(prefix, sizeof(prefix), STREAM_PREFIX_FMT, (unsigned)len);

        res = httpd_resp_send_chunk(req, prefix, plen);
        if (res == ESP_OK) {
            res = httpd_resp_send_chunk(req, (const char *)fb->buf, len);
        }

        esp_camera_fb_return(fb);
        int64_t t_send = esp_timer_get_time();
        grab_us += t_grab1 - t_grab0;
        send_us += t_send - t_grab1;
        if (t_grab1 - t_grab0 > grab_max_us) grab_max_us = t_grab1 - t_grab0;
        if (t_send - t_grab1 > send_max_us) send_max_us = t_send - t_grab1;
        jpeg_sum += len;
        if (jpeg_min == 0 || len < jpeg_min) jpeg_min = len;
        if (len > jpeg_max) jpeg_max = len;
        if (++stat_frames % STREAM_STATS_EVERY == 0) {
            int64_t now_us = esp_timer_get_time();
            double window_s = (double)(now_us - window_start_us) / 1000000.0;
            if (window_s <= 0.0) {
                window_s = 0.001;
            }
            ESP_LOGI(TAG, "%u frames in %.1f s (%.1f fps) | grab avg %.1f ms (max %.1f) | "
                          "send avg %.1f ms (max %.1f) | jpeg avg %.1f KB (min %.1f max %.1f) | "
                          "%.0f KB/s",
                     (unsigned)stat_frames, window_s, (double)stat_frames / window_s,
                     (double)grab_us / 1000.0 / stat_frames, (double)grab_max_us / 1000.0,
                     (double)send_us / 1000.0 / stat_frames, (double)send_max_us / 1000.0,
                     (double)jpeg_sum / 1024.0 / stat_frames,
                     (double)jpeg_min / 1024.0, (double)jpeg_max / 1024.0,
                     (double)jpeg_sum / 1024.0 / window_s);
            stat_frames = 0;
            grab_us = send_us = grab_max_us = send_max_us = 0;
            jpeg_sum = jpeg_min = jpeg_max = 0;
            window_start_us = now_us;
        }

        if (res != ESP_OK) {
            /* Client went away; this is the normal way a stream ends. */
            break;
        }

        s_frames_sent++;

        int64_t now = esp_timer_get_time();
        if (s_last_frame_us != 0) {
            float dt = (float)(now - s_last_frame_us) / 1000000.0f;
            if (dt > 5.0f) {
                s_fps = 0.0f;              /* long gap: restart the average */
            } else if (dt > 0.0001f) {
                float inst = 1.0f / dt;
                s_fps = (s_fps <= 0.0f) ? inst : (0.9f * s_fps + 0.1f * inst);
            }
        }
        s_last_frame_us = now;
    }

    ESP_LOGI(TAG, "client disconnected (0x%x), stream task stack high-water mark %u bytes",
             res, (unsigned)(STREAM_TASK_STACK - uxTaskGetStackHighWaterMark(NULL)));

    /* Finish the chunked response and release the async request / socket. */
    httpd_resp_send_chunk(req, NULL, 0);
    httpd_req_async_handler_complete(req);

    s_clients--;
    s_last_frame_us = 0;
    s_fps = 0.0f;

    vTaskDelete(NULL);
}

static esp_err_t stream_handler(httpd_req_t *req)
{
    /*
     * Hand the request to a dedicated task and return immediately. The HTTP
     * server worker is then free to serve the status page (and further stream
     * clients) while this task pushes frames. This is the pattern used by the
     * official Espressif camera_web_server example.
     */
    httpd_req_t *copy = NULL;
    esp_err_t res = httpd_req_async_handler_begin(req, &copy);
    if (res != ESP_OK) {
        ESP_LOGE(TAG, "httpd_req_async_handler_begin failed: %s", esp_err_to_name(res));
        return res;
    }

    s_clients++;

    if (xTaskCreatePinnedToCore(stream_task, "mjpeg_stream", STREAM_TASK_STACK,
                                copy, STREAM_TASK_PRIO, NULL, tskNO_AFFINITY) != pdPASS) {
        s_clients--;
        ESP_LOGE(TAG, "failed to create stream task");
        httpd_resp_send_err(copy, HTTPD_500_INTERNAL_SERVER_ERROR, "out of memory");
        httpd_req_async_handler_complete(copy);
        return ESP_FAIL;
    }

    return ESP_OK;
}

/* ------------------------------------------------------------------------- */
/* Server                                                                     */
/* ------------------------------------------------------------------------- */

esp_err_t stream_server_start(void)
{
    if (s_server != NULL) {
        return ESP_OK;
    }

    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;
    config.max_uri_handlers = 8;
    config.stack_size = 8192;
    /* Recycle the least recently used connection instead of refusing new ones. */
    config.lru_purge_enable = true;
    config.recv_wait_timeout = 10;
    config.send_wait_timeout = 10;

    esp_err_t err = httpd_start(&s_server, &config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "httpd_start failed: %s", esp_err_to_name(err));
        s_server = NULL;
        return err;
    }

    httpd_uri_t index_uri = {
        .uri = "/",
        .method = HTTP_GET,
        .handler = index_handler,
        .user_ctx = NULL,
    };
    httpd_uri_t stream_uri = {
        .uri = STREAM_PATH,
        .method = HTTP_GET,
        .handler = stream_handler,
        .user_ctx = NULL,
    };
    httpd_uri_t ota_uri = {
        .uri = STREAM_OTA_CHECK_PATH,
        .method = HTTP_POST,
        .handler = ota_check_handler,
        .user_ctx = NULL,
    };

    ESP_ERROR_CHECK(httpd_register_uri_handler(s_server, &index_uri));
    ESP_ERROR_CHECK(httpd_register_uri_handler(s_server, &stream_uri));
    ESP_ERROR_CHECK(httpd_register_uri_handler(s_server, &ota_uri));

    ESP_LOGI(TAG, "Stream server started");
    ESP_LOGI(TAG, "Stream server listening on port 80");

    return ESP_OK;
}

void stream_server_stop(void)
{
    if (s_server == NULL) {
        return;
    }

    ESP_LOGW(TAG, "Stopping stream server to release port 80 for the portal");
    httpd_stop(s_server);
    s_server = NULL;
    s_fps = 0.0f;
    s_last_frame_us = 0;
}

bool stream_server_is_running(void)
{
    return s_server != NULL;
}

uint32_t stream_frames_sent(void)
{
    return s_frames_sent;
}

float stream_fps(void)
{
    return s_fps;
}

int stream_client_count(void)
{
    return s_clients;
}
