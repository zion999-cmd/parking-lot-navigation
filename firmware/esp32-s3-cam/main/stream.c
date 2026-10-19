#include "stream.h"

#include <stdio.h>
#include <string.h>

#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include <esp_app_desc.h>
#include <esp_camera.h>
#include <esp_heap_caps.h>
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

/*
 * Minimum time between frames. 0 disables pacing.
 *
 * #if, not #ifdef: a Kconfig int is always defined, and the project has already been bitten once by
 * an #ifdef on a `default n` bool that the build never defined.
 */
#if CONFIG_STREAM_MAX_FPS > 0
#define STREAM_MIN_INTERVAL_US (1000000 / CONFIG_STREAM_MAX_FPS)
#else
#define STREAM_MIN_INTERVAL_US 0
#endif

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
/* When the last frame *started*, for the rate limit. s_last_frame_us is when the last frame
 * finished sending, and pacing against that double-counted: an 83 ms wait after an 80 ms send is
 * 163 ms per frame, measured as src=5.5 fps against a 12 fps cap. */
static int64_t s_last_grab_us = 0;
static volatile int s_clients = 0;

/*
 * Diagnostics only - nothing here changes what the stream does.
 *
 * The question is whether N clients share one frame source or each drive their own capture. The
 * driver's esp_camera_fb_get() pops from cam_obj->frame_buffer_queue (cam_hal.c:709), which is one
 * queue for the whole camera, so the theory is that the produced frames are divided. This counter
 * proves it either way: if camera_get/s stays at the camera's rate as clients are added, the frames
 * are being shared; if it multiplies with the client count, each handler is provoking its own grab.
 */
static volatile unsigned s_fb_get_total = 0;
static volatile float s_get_rate = 0.0f;
static int64_t s_rate_start_us = 0;
static unsigned s_rate_count = 0;
static volatile int s_client_seq = 0;

static void note_fb_get(void)
{
    s_fb_get_total++;
    s_rate_count++;
    int64_t now = esp_timer_get_time();
    if (s_rate_start_us == 0) {
        s_rate_start_us = now;
        return;
    }
    int64_t elapsed = now - s_rate_start_us;
    if (elapsed >= 2000000) {                  /* a two second window: get/s over recent frames */
        s_get_rate = (float)s_rate_count * 1000000.0f / (float)elapsed;
        s_rate_count = 0;
        s_rate_start_us = now;
    }
}

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
        "<tr><td>Clients</td><td>%d (%d get/s)</td></tr>\n"
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
        (int)(s_get_rate + 0.5f),
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

/*
 * Camera-only benchmark: the same camera, the same configuration, no HTTP video, no Wi-Fi stream.
 *
 * It exists to answer one question and nothing else: what frame rate does the OV5640 actually
 * produce through esp32-camera into the framebuffer queue? If that number is ~11 fps, the ceiling is
 * upstream of HTTP and the stream path is innocent; if it is ~30, the loss is in the HTTP/TCP/Wi-Fi
 * path. Nothing here changes the camera - it calls the same esp_camera_fb_get() the stream handler
 * does, and it refuses to run while a /stream client is attached, because two consumers would split
 * the frames and make the number meaningless.
 *
 *   curl -X POST http://<board>/api/bench
 */
/*
 * Read-only snapshot of the registers that decide the frame rate. Nothing is written.
 *
 * Parsed alongside the raw hex because the interesting numbers are Hz-adjacent: HTS and VTS set the
 * frame length, exposure is what the AEC has actually chosen (in lines), and the AEC max-exposure
 * registers say whether the sensor is allowed to stretch the exposure past VTS - which is the
 * difference between a fixed frame rate and one that drops in dim light.
 */
static void read_regs(char *out, size_t n)
{
    uint8_t r3034 = camera_peek_reg(0x3034), r3035 = camera_peek_reg(0x3035);
    uint8_t r3036 = camera_peek_reg(0x3036), r3037 = camera_peek_reg(0x3037);
    uint8_t r3108 = camera_peek_reg(0x3108), r3824 = camera_peek_reg(0x3824);
    uint8_t r460c = camera_peek_reg(0x460c);
    uint8_t hc = camera_peek_reg(0x380c), hl = camera_peek_reg(0x380d);
    uint8_t vc = camera_peek_reg(0x380e), vl = camera_peek_reg(0x380f);
    uint8_t e0 = camera_peek_reg(0x3500), e1 = camera_peek_reg(0x3501), e2 = camera_peek_reg(0x3502);
    uint8_t man = camera_peek_reg(0x3503);
    uint8_t g0 = camera_peek_reg(0x350a), g1 = camera_peek_reg(0x350b);
    uint8_t a0 = camera_peek_reg(0x3a00);
    uint8_t m0 = camera_peek_reg(0x3a02), m1 = camera_peek_reg(0x3a03);
    uint8_t n0 = camera_peek_reg(0x3a14), n1 = camera_peek_reg(0x3a15);
    uint8_t c0 = camera_peek_reg(0x3a18), c1 = camera_peek_reg(0x3a19);

    int hts = (hc << 8) | hl;
    int vts = (vc << 8) | vl;
    int exposure = ((e0 & 0x0F) << 12) | (e1 << 4) | ((e2 >> 4) & 0x0F);
    int gain = (g0 << 8) | g1;
    int aec_max = (m0 << 8) | m1;
    int aec_max_alt = (n0 << 8) | n1;
    int gain_ceil = ((c0 & 0x03) << 8) | c1;

    snprintf(out, n,
        "{\"3034\":\"0x%02x\",\"3035\":\"0x%02x\",\"3036\":\"0x%02x\",\"3037\":\"0x%02x\","
        "\"3108\":\"0x%02x\",\"3824\":\"0x%02x\",\"460c\":\"0x%02x\","
        "\"380c_d\":\"0x%02x%02x\",\"380e_f\":\"0x%02x%02x\","
        "\"3500_02\":\"0x%02x%02x%02x\",\"3503\":\"0x%02x\",\"350a_b\":\"0x%02x%02x\","
        "\"3a00\":\"0x%02x\",\"3a02_03\":\"0x%02x%02x\",\"3a14_15\":\"0x%02x%02x\","
        "\"3a18_19\":\"0x%02x%02x\","
        "\"hts\":%d,\"vts\":%d,\"exposure\":%d,\"gain\":%d,"
        "\"aec_max\":%d,\"aec_max_alt\":%d,\"gain_ceiling\":%d}",
        r3034, r3035, r3036, r3037, r3108, r3824, r460c,
        hc, hl, vc, vl, e0, e1, e2, man, g0, g1,
        a0, m0, m1, n0, n1, c0, c1,
        hts, vts, exposure, gain, aec_max, aec_max_alt, gain_ceil);
}

static esp_err_t bench_handler(httpd_req_t *req)
{
    if (s_clients > 0) {
        return httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST,
                                   "a /stream client is attached; disconnect it first");
    }

    const int seconds = 20;
    unsigned frames = 0;
    unsigned long long bytes = 0, min_b = 0, max_b = 0;
    long long sum_get = 0, min_get = 0, max_get = 0;

    static char regs_before[640], regs_during[640], regs_after[640];
    read_regs(regs_before, sizeof(regs_before));
    bool sampled = false;

    int64_t t0 = esp_timer_get_time();
    while (esp_timer_get_time() - t0 < (int64_t)seconds * 1000000) {
        if (!sampled && esp_timer_get_time() - t0 >= (int64_t)seconds * 500000) {
            read_regs(regs_during, sizeof(regs_during));
            sampled = true;
        }
        int64_t g0 = esp_timer_get_time();
        camera_fb_t *fb = esp_camera_fb_get();
        int64_t g1 = esp_timer_get_time();
        if (fb == NULL) {
            continue;
        }
        frames++;
        bytes += fb->len;
        if (min_b == 0 || fb->len < min_b) min_b = fb->len;
        if (fb->len > max_b) max_b = fb->len;
        long long d = g1 - g0;
        sum_get += d;
        if (min_get == 0 || d < min_get) min_get = d;
        if (d > max_get) max_get = d;
        esp_camera_fb_return(fb);
    }
    int64_t elapsed = esp_timer_get_time() - t0;

    read_regs(regs_after, sizeof(regs_after));

    char body[2048];
    int n = snprintf(body, sizeof(body),
        "{\"mode\":\"camera_only\",\"seconds\":%d,\"elapsed_ms\":%lld,\"frames\":%u,"
        "\"fps\":%.2f,\"avg_frame_bytes\":%llu,\"min_frame_bytes\":%llu,"
        "\"max_frame_bytes\":%llu,\"avg_fb_get_us\":%.0f,"
        "\"min_fb_get_us\":%lld,\"max_fb_get_us\":%lld,\"clients\":%d,\"uptime_s\":%u,"
        "\"regs_before\":%s,\"regs_during\":%s,\"regs_after\":%s}\n",
        seconds, (long long)(elapsed / 1000), frames,
        frames > 0 ? (double)frames * 1000000.0 / (double)elapsed : 0.0,
        frames ? (unsigned long long)(bytes / frames) : 0ULL, min_b, max_b,
        frames ? (double)sum_get / (double)frames : 0.0,
        (long long)min_get, (long long)max_get, (int)s_clients,
        (unsigned)(esp_timer_get_time() / 1000000),
        regs_before[0] ? regs_before : "null",
        regs_during[0] ? regs_during : "null",
        regs_after[0] ? regs_after : "null");
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    return httpd_resp_send(req, body, n);
}

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

    /*
     * Nagle off.
     *
     * esp_http_server does not touch TCP_NODELAY, so Nagle is on by default for this
     * socket, and each frame is written as several pieces (chunk header, prefix, JPEG,
     * terminator). Nagle holds a small piece back until earlier data is acknowledged,
     * and the peer's delayed ACK adds its own wait on top. Measured on this board that
     * cost about 281 ms per frame - an 11.7 KB frame taking 281 ms when the link's round
     * trip was 66 ms - and once the frames were big enough to fill the send buffer the
     * connection wedged outright: "httpd_sock_err: error in send : 11" (EAGAIN) after the
     * 10 s send timeout, with the board writing 105 KB while the client received 35 KB.
     *
     * A stream is the one case where batching writes is never worth a round trip.
     */
    int sock = httpd_req_to_sockfd(req);
    if (sock >= 0) {
        int one = 1;
        if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) != 0) {
            ESP_LOGW(TAG, "could not disable Nagle on the stream socket: errno %d", errno);
        } else {
            ESP_LOGI(TAG, "stream socket: TCP_NODELAY on (Nagle disabled)");
        }
    } else {
        ESP_LOGW(TAG, "could not resolve the stream socket, Nagle stays on");
    }

    /*
     * Scratch buffer, so a camera frame is never held across network I/O.
     *
     * The driver only has fb_count frame buffers, and a streaming task used to hold one
     * of them for the whole duration of a send - which can be seconds when the network
     * is unhappy. Two clients then held both of them and the camera pipeline froze: the
     * board logged 9 frames in 100 s and never recovered. Copying the JPEG to PSRAM and
     * handing the camera buffer straight back decouples the two, and a slow client can
     * now only hurt itself.
     */
    const size_t scratch_size = 96 * 1024;   /* SVGA JPEGs are ~12 KB, but be generous */
    uint8_t *scratch = heap_caps_malloc(scratch_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (scratch == NULL) {
        ESP_LOGW(TAG, "no scratch buffer; frames will be held across the send instead");
    }

    uint32_t stat_frames = 0;
    uint32_t stat_fails = 0;
    int64_t grab_us = 0, send_us = 0, grab_max_us = 0, send_max_us = 0;
    size_t jpeg_sum = 0, jpeg_min = 0, jpeg_max = 0;
    int64_t window_start_us = esp_timer_get_time();

    while (res == ESP_OK) {
        /*
         * Pacing: the uplink is the scarce resource here, not the camera. Measured on this
         * rig, 800x600 at 24-27 fps is about 1 MB/s - far more than the vision service uses
         * (it works at around 10) and more transmit airtime than this supply should carry.
         * Waiting so frames start at most STREAM_MAX_FPS apart halves the airtime and the
         * average transmit current, and unlike lowering the transmit power it costs nothing
         * in link quality: 6 dBm made the uplink marginal and Wi-Fi took three minutes to
         * settle. Resolution stays 800x600 because the bay calibration is in those pixels.
         */
        if (STREAM_MIN_INTERVAL_US > 0 && s_last_grab_us != 0) {
            int64_t since = esp_timer_get_time() - s_last_grab_us;
            if (since < STREAM_MIN_INTERVAL_US) {
                vTaskDelay(pdMS_TO_TICKS((STREAM_MIN_INTERVAL_US - since + 999) / 1000));
            }
        }
        s_last_grab_us = esp_timer_get_time();
        int64_t t_grab0 = esp_timer_get_time();
        camera_fb_t *fb = esp_camera_fb_get();
        int64_t t_grab1 = esp_timer_get_time();
        note_fb_get();
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
        const uint8_t *body = fb->buf;

        /* Better to pay a 12 KB memcpy than to keep a frame buffer from the camera. */
        if (scratch != NULL && len <= scratch_size) {
            memcpy(scratch, fb->buf, len);
            body = scratch;
            esp_camera_fb_return(fb);
            fb = NULL;
        }

        char prefix[128];
        int plen = snprintf(prefix, sizeof(prefix), STREAM_PREFIX_FMT, (unsigned)len);

        res = httpd_resp_send_chunk(req, prefix, plen);
        if (res == ESP_OK) {
            res = httpd_resp_send_chunk(req, (const char *)body, len);
        }

        if (fb != NULL) {
            esp_camera_fb_return(fb);
        }
        if (res != ESP_OK) {
            stat_fails++;
        }
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
                          "%.0f KB/s | send failures %u",
                     (unsigned)stat_frames, window_s, (double)stat_frames / window_s,
                     (double)grab_us / 1000.0 / stat_frames, (double)grab_max_us / 1000.0,
                     (double)send_us / 1000.0 / stat_frames, (double)send_max_us / 1000.0,
                     (double)jpeg_sum / 1024.0 / stat_frames,
                     (double)jpeg_min / 1024.0, (double)jpeg_max / 1024.0,
                     (double)jpeg_sum / 1024.0 / window_s, (unsigned)stat_fails);
            stat_frames = 0;
            stat_fails = 0;
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

    if (scratch != NULL) {
        free(scratch);
    }

    s_clients--;
    s_last_frame_us = 0;
    s_last_grab_us = 0;
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
    /* A stalled client must not tie up a streaming task for long. */
    config.send_wait_timeout = 3;

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

    httpd_uri_t bench_uri = {
        .uri = STREAM_BENCH_PATH,
        .method = HTTP_POST,
        .handler = bench_handler,
        .user_ctx = NULL,
    };
    ESP_ERROR_CHECK(httpd_register_uri_handler(s_server, &bench_uri));

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
    s_last_grab_us = 0;
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
