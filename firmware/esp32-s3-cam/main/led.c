#include "led.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include <cJSON.h>
#include <driver/rmt_tx.h>
#include <esp_heap_caps.h>
#include <esp_http_client.h>
#include <esp_log.h>
#include <esp_rom_sys.h>
#include <esp_timer.h>
#include <sdkconfig.h>

#include "discovery.h"
#include "wifi.h"

/*
 * Bay status lights - implementation. Read led.h first: it holds the colour
 * semantics and the two rules (UNKNOWN is dark; a failed poll blanks).
 *
 * Layout of the strip:
 *
 *   pixel 0            pixel 1            ...   pixel N-1
 *   bay LED_ORDER[0]   bay LED_ORDER[1]         bay LED_ORDER[N-1]
 *
 * LED_ORDER is the installation, not a derivation: which pixel ended up next to
 * which bay is a fact about where the strip was glued down, and nothing on the
 * vision side can know it.
 *
 * The waveform is generated locally with the RMT byte encoder. RMT is free here,
 * and that was checked rather than assumed: the camera XCLK owns
 * LEDC_TIMER_0/CHANNEL_0 and the TFT backlight owns LEDC_TIMER_1/CHANNEL_1
 * (README 12.1), so there is no resource to share.
 */

static const char *TAG = "led";

/* --- timing ------------------------------------------------------------- */

/* 10 MHz: one RMT tick is 0.1 us, so the datasheet's timings are integers. */
#define RMT_RESOLUTION_HZ 10000000

/*
 * WS2812B bit timing in us. Datasheet: bit 0 is T0H 0.4 / T0L 0.85, bit 1 is
 * T1H 0.8 / T1L 0.45. These sit slightly inside the window (0.3/0.9 and 0.6/0.6).
 */
#define T0H_TICKS 3
#define T0L_TICKS 9
#define T1H_TICKS 6
#define T1L_TICKS 6

/* Latch: the strip needs >280 us of idle between frames. */
#define RESET_US 400

/* --- sizes -------------------------------------------------------------- */

/* Bay ids in this lot are "9", "37", "B1" - 8 bytes is generous, not tight. */
#define BAY_ID_MAX 8

/*
 * The state document is ~3.7 KB for 48 bays (~78 bytes each). This cap is
 * deliberately loose; a body that overflows it is treated as a failed poll
 * rather than a truncated read, because a truncated JSON document would parse
 * as "no bays" and turn the whole strip dark for the wrong reason.
 */
#define STATE_BODY_MAX 8192

/* route.json is about 400 bytes; loose for the same reason as above. */
#define ROUTE_BODY_MAX 2048

/* --- colours -------------------------------------------------------------
 *
 * Full-scale RGB, scaled by LED_BRIGHTNESS when the frame is built. One table,
 * so the meaning of a colour has exactly one home.
 */
typedef struct {
    uint8_t r, g, b;
} rgb_t;

typedef enum {
    BAY_EMPTY = 0,      /* green  */
    BAY_OCCUPIED,       /* red    */
    BAY_NAVIGATION,     /* yellow - reserved, see below */
    BAY_UNKNOWN,        /* dark   */
    BAY_STATE_COUNT
} bay_state_t;

static const rgb_t s_colour[BAY_STATE_COUNT] = {
    [BAY_EMPTY]      = {   0, 255,   0 },
    [BAY_OCCUPIED]   = { 255,   0,   0 },
    /*
     * Reserved for navigation: the bay the current route targets, which is
     * P0003's route document (route.json, "target_spot", when "ok" is true).
     * Nothing sets BAY_NAVIGATION yet - the colour exists here so that when it
     * is wired up there is one place to look, and so yellow is not double-booked
     * by something else in the meantime.
     */
    [BAY_NAVIGATION] = { 255, 255,   0 },
    [BAY_UNKNOWN]    = {   0,   0,   0 },
};

static const char STATE_CHAR[BAY_STATE_COUNT] = { 'E', 'O', 'N', 'U' };

/* --- module state ------------------------------------------------------- */

static uint8_t s_frame[CONFIG_LED_COUNT * 3];
static bay_state_t s_state[CONFIG_LED_COUNT];

/* Bay driving each pixel; an empty string means "this pixel is not used". */
static char s_bay[CONFIG_LED_COUNT][BAY_ID_MAX];

/* What each pixel showed at the last frame that went out, as a STATE_CHAR. */
static char s_shown[CONFIG_LED_COUNT];

/*
 * Freshness of the state document, without a clock.
 *
 * The device has no reliable wall clock, but it does not need one: main.py rewrites
 * the document every frame and its timestamp is whole seconds, so the timestamp
 * advancing at all means someone is writing, and standing still means nobody is.
 *
 * Why this exists: the first version only checked that the fetch succeeded, so a
 * document that had been frozen for five minutes was displayed as the current state
 * - a strip confidently showing occupied bays that nobody had looked at since the
 * vision loop died. That is the same lie as reporting UNKNOWN as EMPTY, just slower,
 * and the README claimed the firmware did not do it. Now it does not.
 */
static int64_t s_doc_ts = -1;              /* last timestamp seen in the document */
static int64_t s_doc_ts_changed_us;        /* when it last changed */
static bool s_reported_stale;

static rmt_channel_handle_t s_rmt = NULL;
static rmt_encoder_handle_t s_encoder = NULL;

typedef enum {
    LED_DISABLED = 0,   /* never started */
    LED_NO_DATA,        /* started, nothing read yet */
    LED_STALE,          /* was working, the last poll failed */
    LED_OK,
} led_status_t;

static volatile led_status_t s_status = LED_DISABLED;

/* ------------------------------------------------------------------------- */
/* Hardware                                                                   */
/* ------------------------------------------------------------------------- */

static void led_init_hw(void)
{
    rmt_tx_channel_config_t tx_cfg = {
        .clk_src = RMT_CLK_SRC_DEFAULT,
        .gpio_num = CONFIG_LED_GPIO,
        .mem_block_symbols = 64,
        .resolution_hz = RMT_RESOLUTION_HZ,
        .trans_queue_depth = 4,
    };
    ESP_ERROR_CHECK(rmt_new_tx_channel(&tx_cfg, &s_rmt));

    rmt_bytes_encoder_config_t enc_cfg = {
        .bit0 = { .level0 = 1, .duration0 = T0H_TICKS,
                  .level1 = 0, .duration1 = T0L_TICKS },
        .bit1 = { .level0 = 1, .duration0 = T1H_TICKS,
                  .level1 = 0, .duration1 = T1L_TICKS },
        .flags = { .msb_first = 1 },
    };
    ESP_ERROR_CHECK(rmt_new_bytes_encoder(&enc_cfg, &s_encoder));
    ESP_ERROR_CHECK(rmt_enable(s_rmt));
}

static void led_show(void)
{
    rmt_transmit_config_t cfg = { .loop_count = 0 };
    ESP_ERROR_CHECK(rmt_transmit(s_rmt, s_encoder, s_frame, sizeof(s_frame), &cfg));
    ESP_ERROR_CHECK(rmt_tx_wait_all_done(s_rmt, -1));
    esp_rom_delay_us(RESET_US);

    for (int i = 0; i < CONFIG_LED_COUNT; i++) {
        s_shown[i] = STATE_CHAR[s_state[i]];
    }
}

/* Build the wire bytes for one pixel. GRB order, scaled by the duty limit. */
static void frame_set(int pixel, bay_state_t state)
{
    const rgb_t c = s_colour[state];

    /*
     * Brightness is a duty limit, not a taste setting: 48 pixels at full white
     * would be ~2.9 A and this board runs off USB or a small battery. It is also
     * the electrical margin on the data line - see README 13.
     */
    s_frame[pixel * 3 + 0] = (uint8_t)((c.g * CONFIG_LED_BRIGHTNESS) / 255);
    s_frame[pixel * 3 + 1] = (uint8_t)((c.r * CONFIG_LED_BRIGHTNESS) / 255);
    s_frame[pixel * 3 + 2] = (uint8_t)((c.b * CONFIG_LED_BRIGHTNESS) / 255);

    s_state[pixel] = state;
}

/* Everything dark. Used at boot and whenever the state is not fresh. */
static void frame_all_dark(void)
{
    for (int i = 0; i < CONFIG_LED_COUNT; i++) {
        frame_set(i, BAY_UNKNOWN);
    }
}

/* ------------------------------------------------------------------------- */
/* Configuration                                                              */
/* ------------------------------------------------------------------------- */

/*
 * Parse LED_ORDER ("9,37,B1,...") into pixel -> bay. Position in the string is
 * the pixel index, so the string is literally the strip seen from DIN.
 */
static int parse_order(void)
{
    const char *p = CONFIG_LED_ORDER;
    int pixel = 0;

    while (*p != '\0' && pixel < CONFIG_LED_COUNT) {
        while (*p == ' ' || *p == ',') {
            p++;
        }
        if (*p == '\0') {
            break;
        }

        size_t n = 0;
        while (p[n] != '\0' && p[n] != ',' && p[n] != ' ' && n < BAY_ID_MAX - 1) {
            n++;
        }
        memcpy(s_bay[pixel], p, n);
        s_bay[pixel][n] = '\0';
        pixel++;
        p += n;
    }

    return pixel;
}

/* ------------------------------------------------------------------------- */
/* Polling                                                                    */
/* ------------------------------------------------------------------------- */

static bool fetch_document(const char *url, char *body, size_t cap, size_t *out_len)
{
    esp_http_client_config_t cfg = {};
    cfg.url = url;
    cfg.timeout_ms = 5000;
    cfg.buffer_size = 1024;

    esp_http_client_handle_t client = esp_http_client_init(&cfg);
    if (client == NULL) {
        return false;
    }

    *out_len = 0;
    bool ok = false;

    if (esp_http_client_open(client, 0) == ESP_OK) {
        /*
         * Headers before status. get_status_code() reads 0 until the response
         * headers have been read, so asking first reports "HTTP 0" for a request
         * the server answered with 200 - and then this would bail out without
         * draining the body, which the server logs as "Connection reset by
         * peer". Same trap the OTA client fell into; see ota.c.
         */
        int content_len = esp_http_client_fetch_headers(client);
        int status = esp_http_client_get_status_code(client);

        if (status != 200) {
            ESP_LOGW(TAG, "GET %s -> HTTP %d", url, status);
        } else if (content_len > (int)cap - 1) {
            /* Refuse rather than truncate: half a document reads as "no bays". */
            ESP_LOGW(TAG, "%s is %d bytes, larger than the %u byte buffer",
                     url, content_len, (unsigned)cap);
        } else {
            int total = 0;
            while (total < (int)cap - 1) {
                int n = esp_http_client_read(client, body + total, cap - 1 - (size_t)total);
                if (n <= 0) {
                    break;
                }
                total += n;
                if (content_len > 0 && total >= content_len) {
                    break;
                }
            }
            body[total] = '\0';
            *out_len = (size_t)total;
            ok = total > 0;
        }
    } else {
        ESP_LOGW(TAG, "could not open %s", url);
    }

    esp_http_client_close(client);
    esp_http_client_cleanup(client);
    return ok;
}

/*
 * The navigation target, if there is one right now.
 *
 * A failed read here is deliberately NOT a failed frame: the occupancy colours are
 * still fresh and still true, so they stay up and only the yellow goes away. Throwing
 * away a true colour because a second document was unreachable would be worse than
 * losing the target hint.
 */
static bool fetch_target(char *route_buf, char *out, size_t out_len)
{
    static int failures = 0;

    char route_url[192];
    discovery_url(CONFIG_LED_ROUTE_URL, route_url, sizeof(route_url));

    size_t len = 0;
    if (!fetch_document(route_url, route_buf, ROUTE_BODY_MAX, &len)) {
        if (failures++ == 0 || failures % 20 == 0) {
            ESP_LOGW(TAG, "no route document (%d in a row) - showing no target", failures);
        }
        return false;
    }
    failures = 0;

    cJSON *root = cJSON_ParseWithLength(route_buf, len);
    if (root == NULL) {
        ESP_LOGW(TAG, "route document is not valid JSON (%u bytes)", (unsigned)len);
        return false;
    }

    bool ok = false;
    const cJSON *jok = cJSON_GetObjectItemCaseSensitive(root, "ok");
    const cJSON *jt = cJSON_GetObjectItemCaseSensitive(root, "target_spot");

    /*
     * "ok" before "target_spot", always. A refusal names the bay that was asked for
     * with ok == false (spot-has-no-access, hero-not-visible, ...), and the page's
     * "navigation ended" clear writes ok == false with an empty target. Reading
     * target_spot without checking ok is exactly how a refusal would light up as if
     * it were a plan - and a refusal means "you cannot go there".
     */
    if (cJSON_IsTrue(jok) && cJSON_IsString(jt) && jt->valuestring[0] != '\0') {
        snprintf(out, out_len, "%s", jt->valuestring);
        ok = true;
    }

    cJSON_Delete(root);
    return ok;
}

/*
 * Turn the state document's "spots" object into a frame.
 *
 * Only "state" is read. "changed_ratio" is deliberately ignored: the contract
 * says it is diagnostic information and must not be used to make a decision, and
 * this is a decision.
 *
 * Returns how many of the mapped bays the document actually contained, which the
 * caller uses to say so when the answer is "none of them".
 */
static int build_frame(const cJSON *spots, const char *target)
{
    int found = 0;

    for (int i = 0; i < CONFIG_LED_COUNT; i++) {
        bay_state_t state = BAY_UNKNOWN;

        if (s_bay[i][0] != '\0') {
            const cJSON *spot = cJSON_GetObjectItemCaseSensitive(spots, s_bay[i]);
            const cJSON *js = spot ? cJSON_GetObjectItemCaseSensitive(spot, "state") : NULL;

            if (js != NULL) {
                found++;
            }
            if (cJSON_IsString(js)) {
                if (strcmp(js->valuestring, "EMPTY") == 0) {
                    state = BAY_EMPTY;
                } else if (strcmp(js->valuestring, "OCCUPIED") == 0) {
                    state = BAY_OCCUPIED;
                }
                /* Anything else, including "UNKNOWN", stays dark. */
            }
            /*
             * A bay named in LED_ORDER that the state document does not contain
             * is left dark rather than guessed at - a mis-typed id must not look
             * like an empty bay.
             */
        }

        frame_set(i, state);
    }

    /* Second pass: the navigation target paints over green, never over red. */
    if (target == NULL || target[0] == '\0') {
        return found;
    }

    for (int i = 0; i < CONFIG_LED_COUNT; i++) {
        if (strcmp(s_bay[i], target) != 0) {
            continue;
        }

        /*
         * Yellow means "come here", so it only ever replaces an empty bay. If the
         * target has been taken since it was planned, red has to win: pointing a
         * driver at a bay that is already gone is the one thing the lights must not
         * do. This also makes the pair self-correcting - the target bay fills up and
         * the yellow turns red on its own.
         */
        if (s_state[i] == BAY_EMPTY) {
            frame_set(i, BAY_NAVIGATION);
        }
        break;
    }

    return found;
}

/* Log the pixels whose colour changed, so a transition can be traced to a bay. */
static void log_changes(void)
{
    char line[200];
    int used = 0;
    int drawn = 0;
    int dark = 0;

    line[0] = '\0';

    for (int i = 0; i < CONFIG_LED_COUNT; i++) {
        if (s_state[i] == BAY_UNKNOWN) {
            dark++;
        } else {
            drawn++;
        }

        if (STATE_CHAR[s_state[i]] != s_shown[i]) {
            int n = snprintf(line + used, sizeof(line) - (size_t)used, "%s%c>%c ",
                             s_bay[i][0] ? s_bay[i] : "?", s_shown[i], STATE_CHAR[s_state[i]]);
            if (n > 0) {
                used += n;
                if (used >= (int)sizeof(line)) {
                    used = (int)sizeof(line) - 1;
                }
            }
        }
    }

    if (used > 0) {
        ESP_LOGI(TAG, "%d lit, %d dark | %s", drawn, dark, line);
    }
}

/*
 * Say so when the state document and LED_ORDER do not describe the same bays.
 *
 * Without this the mismatch is silent: every pixel is dark, and a transition log
 * has nothing to report because "dark" is what it started as. Seen in practice -
 * a vision process still running against an older calibration published bay ids of
 * 105..207 while the strip was mapped to 9..50, and the strip simply sat there
 * dark with a clean log. The count is reported when it changes, so a pipeline that
 * comes back with the right ids says that too.
 */
static void log_bays_found(int found)
{
    static int last = -1;

    if (found == last) {
        return;
    }
    last = found;

    if (found == 0) {
        ESP_LOGE(TAG, "none of the %d bays in LED_ORDER are in the state document - "
                      "the ids do not match the running calibration (check that the vision "
                      "process was restarted after the map was edited)", CONFIG_LED_COUNT);
    } else if (found < CONFIG_LED_COUNT) {
        ESP_LOGW(TAG, "only %d of %d bays in LED_ORDER are in the state document - "
                      "the rest stay dark", found, CONFIG_LED_COUNT);
    } else {
        ESP_LOGI(TAG, "all %d bays matched", found);
    }
}

/* ------------------------------------------------------------------------- */
/* Bring-up test                                                              */
/* ------------------------------------------------------------------------- */

#if CONFIG_LED_SELFTEST

/*
 * The point of this is to answer "is the strip wired and working" without a
 * meter, a logic analyser or a second firmware project: it runs once, in the led
 * task, before the module does anything else.
 *
 * What each phase proves:
 *
 *   walk    one pixel at a time from DIN outwards. Counting how many light up
 *           gives LED_COUNT, and whether it starts at pixel 0 says the strip is
 *           fed from its INPUT end rather than its output.
 *   colours green / red / yellow, the three sematics this module will use. Wrong
 *           colour out ("red" that looks green) means the strip is RGB, not GRB.
 *   ramp    brightness up and down, the one phase that fails electrically: a
 *           strip on 5 V wants 3.5 V for a logic high while the ESP32 drives
 *           3.3 V, and a board-fed strip can also brown the rail out. Watch for
 *           a flickering or wrong-coloured first pixel, or the board rebooting.
 *
 * The ramp deliberately stops at 64/255 (about 0.7 A for 48 pixels) instead of
 * 255 (2.9 A). Raising it tests the supply harder, and is the intended way to
 * reproduce a rail problem - but it is also how a marginal USB feed gets
 * brown-out reset in the middle of a demo, so it is not the default.
 */
#define SELFTEST_RAMP_MAX 64
#define SELFTEST_RAMP_STEP 8
#define SELFTEST_WALK_MS 120
#define SELFTEST_COLOUR_MS 700

static void selftest(void)
{
    ESP_LOGW(TAG, "bring-up test: count the pixels that light up, then watch for flicker");

    for (int i = 0; i < CONFIG_LED_COUNT; i++) {
        frame_all_dark();
        led_show();
        frame_set(i, BAY_EMPTY);
        led_show();
        ESP_LOGI(TAG, "  pixel %d lit", i);
        vTaskDelay(pdMS_TO_TICKS(SELFTEST_WALK_MS));
    }

    /* The three colours this module will actually use, including the reserved one. */
    static const bay_state_t shown[] = { BAY_EMPTY, BAY_OCCUPIED, BAY_NAVIGATION };
    for (int i = 0; i < (int)(sizeof(shown) / sizeof(shown[0])); i++) {
        bool all = false;
        for (int p = 0; p < CONFIG_LED_COUNT; p++) {
            frame_set(p, shown[i]);
            all = true;
        }
        if (all) {
            led_show();
            vTaskDelay(pdMS_TO_TICKS(SELFTEST_COLOUR_MS));
        }
    }

    ESP_LOGI(TAG, "  brightness ramp 0 -> %d -> 0 (flicker? wrong colour? board reset?)",
             SELFTEST_RAMP_MAX);
    for (int duty = SELFTEST_RAMP_STEP; duty <= SELFTEST_RAMP_MAX; duty += SELFTEST_RAMP_STEP) {
        for (int p = 0; p < CONFIG_LED_COUNT; p++) {
            s_frame[p * 3 + 0] = (uint8_t)duty;
            s_frame[p * 3 + 1] = (uint8_t)duty;
            s_frame[p * 3 + 2] = (uint8_t)duty;
            s_state[p] = BAY_EMPTY;
        }
        led_show();
        vTaskDelay(pdMS_TO_TICKS(SELFTEST_WALK_MS));
    }
    for (int duty = SELFTEST_RAMP_MAX - SELFTEST_RAMP_STEP; duty > 0; duty -= SELFTEST_RAMP_STEP) {
        for (int p = 0; p < CONFIG_LED_COUNT; p++) {
            s_frame[p * 3 + 0] = (uint8_t)duty;
            s_frame[p * 3 + 1] = (uint8_t)duty;
            s_frame[p * 3 + 2] = (uint8_t)duty;
        }
        led_show();
        vTaskDelay(pdMS_TO_TICKS(SELFTEST_WALK_MS));
    }

    frame_all_dark();
    led_show();
    ESP_LOGW(TAG, "bring-up test done");
}

#endif /* CONFIG_LED_SELFTEST */

/* ------------------------------------------------------------------------- */
/* Task                                                                       */
/* ------------------------------------------------------------------------- */

static void led_task(void *arg)
{
    (void)arg;

#if CONFIG_LED_SELFTEST
    selftest();
#endif

    if (CONFIG_LED_ORDER[0] == '\0') {
        /*
         * No mapping, no lights. It is deliberately not inferred: an unmapped
         * pixel cannot show anything true, and a strip that lights up "anyway"
         * is how a wrong colour gets trusted. Refusing loudly costs one log line.
         */
        ESP_LOGE(TAG, "LED_ORDER is empty: %d pixels have no bay, strip stays dark "
                      "(set it in menuconfig to show bay state)", CONFIG_LED_COUNT);
        s_status = LED_DISABLED;
        while (true) {
            vTaskDelay(pdMS_TO_TICKS(10000));
        }
    }

    char *body = heap_caps_malloc(STATE_BODY_MAX, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (body == NULL) {
        /* 8 KB from internal RAM is affordable but not free; PSRAM is the
         * intended home for it, as with the OTA download buffer. */
        body = malloc(STATE_BODY_MAX);
    }
    if (body == NULL) {
        ESP_LOGE(TAG, "no buffer for the state document - lights stay off");
        return;
    }

    char *route = heap_caps_malloc(ROUTE_BODY_MAX, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (route == NULL) {
        route = malloc(ROUTE_BODY_MAX);
    }
    if (route == NULL) {
        ESP_LOGE(TAG, "no buffer for the route document - occupancy only, no target");
    }

    int failures = 0;
    bool is_dark = false;
    const TickType_t period = pdMS_TO_TICKS(CONFIG_LED_POLL_MS);

    while (true) {
        if (!wifi_is_connected()) {
            if (!is_dark) {
                frame_all_dark();
                led_show();
                is_dark = true;
            }
            s_status = LED_NO_DATA;
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        /*
         * Resolved per poll rather than once: the server can move (the sandbox
         * changes Wi-Fi), and the cost is one strcmp against the last beacon. If
         * nothing has been heard this is simply the configured URL.
         */
        char state_url[192];
        discovery_url(CONFIG_LED_STATE_URL, state_url, sizeof(state_url));

        size_t len = 0;
        bool ok = fetch_document(state_url, body, STATE_BODY_MAX, &len);

        cJSON *root = NULL;
        const cJSON *spots = NULL;

        if (ok) {
            root = cJSON_ParseWithLength(body, len);
            if (root == NULL) {
                ESP_LOGW(TAG, "state document is not valid JSON (%u bytes)", (unsigned)len);
                ok = false;
            }
        }
        if (ok) {
            spots = cJSON_GetObjectItemCaseSensitive(root, "spots");
            if (!cJSON_IsObject(spots)) {
                ESP_LOGW(TAG, "state document has no \"spots\" object");
                ok = false;
            }
        }

        /*
         * Reached, but is anyone still writing it? A document that arrives fine and
         * never changes is a dead vision loop, and its contents describe a moment
         * that has passed.
         */
        if (ok) {
            const cJSON *jts = cJSON_GetObjectItemCaseSensitive(root, "timestamp");
            int64_t now = esp_timer_get_time();

            if (cJSON_IsNumber(jts) && (int64_t)jts->valuedouble != s_doc_ts) {
                s_doc_ts = (int64_t)jts->valuedouble;
                s_doc_ts_changed_us = now;
            } else if (s_doc_ts >= 0 &&
                       (now - s_doc_ts_changed_us) / 1000 > CONFIG_LED_STATE_STALE_MS) {
                if (!s_reported_stale) {
                    s_reported_stale = true;
                    ESP_LOGE(TAG, "state document has not changed for %d ms - blanking the "
                                  "strip (is main.py still running?)",
                             CONFIG_LED_STATE_STALE_MS);
                }
                ok = false;
            }
        }

        if (!ok) {
            /*
             * Blank rather than hold the last frame. A stale green on a bay that
             * just filled up is the failure this module exists to avoid, and it
             * is the same lie as reporting UNKNOWN as EMPTY.
             */
            failures++;
            if (!is_dark) {
                frame_all_dark();
                led_show();
                is_dark = true;
                ESP_LOGW(TAG, "no fresh state - strip blanked");
            }
            /*
             * Rate limited: un-limited this would print two lines a second
             * forever and bury everything else in the log.
             */
            if (failures == 1 || failures % 20 == 0) {
                ESP_LOGW(TAG, "%d consecutive state fetch failure(s), retrying", failures);
            }
            s_status = LED_STALE;
        } else {
            if (failures > 0) {
                ESP_LOGI(TAG, "state link recovered after %d failure(s)", failures);
                failures = 0;
            }
            if (s_reported_stale) {
                s_reported_stale = false;
                ESP_LOGI(TAG, "state document is moving again");
            }

            /*
             * The target is applied on top of the occupancy colours. route == NULL
             * (allocation failed) means occupancy only, which is still a true display.
             */
            char target[BAY_ID_MAX] = "";
            bool have_target = (route != NULL) && fetch_target(route, target, sizeof(target));

            int found = build_frame(spots, have_target ? target : NULL);
            led_show();
            is_dark = false;
            log_changes();
            log_bays_found(found);
            s_status = LED_OK;
        }

        if (root != NULL) {
            cJSON_Delete(root);
        }

        vTaskDelay(period);
    }
}

/* ------------------------------------------------------------------------- */
/* Public                                                                     */
/* ------------------------------------------------------------------------- */

bool led_start(void)
{
    int mapped = parse_order();

    ESP_LOGI(TAG, "DIN=GPIO%d, %d pixels, %d mapped, brightness %d/255",
             CONFIG_LED_GPIO, CONFIG_LED_COUNT, mapped, CONFIG_LED_BRIGHTNESS);
    if (mapped < CONFIG_LED_COUNT) {
        ESP_LOGW(TAG, "pixels %d..%d have no bay in LED_ORDER and stay dark",
                 mapped, CONFIG_LED_COUNT - 1);
    }
    if (CONFIG_LED_ORDER[0] != '\0') {
        char url[192];
        discovery_url(CONFIG_LED_STATE_URL, url, sizeof(url));
        ESP_LOGI(TAG, "state source = %s (every %d ms)", url, CONFIG_LED_POLL_MS);
        discovery_url(CONFIG_LED_ROUTE_URL, url, sizeof(url));
        ESP_LOGI(TAG, "target source = %s (green=EMPTY red=OCCUPIED yellow=target dark=UNKNOWN)",
                 url);
    }

    led_init_hw();
    frame_all_dark();
    led_show();                     /* start dark, not undefined */
    s_status = LED_NO_DATA;

    if (xTaskCreate(led_task, "led", 5120, NULL, 3, NULL) != pdPASS) {
        ESP_LOGE(TAG, "could not start the led task");
        s_status = LED_DISABLED;
        return false;
    }

    return true;
}

const char *led_status(void)
{
    switch (s_status) {
    case LED_OK:       return "ok";
    case LED_STALE:    return "stale";
    case LED_NO_DATA:  return "no-data";
    case LED_DISABLED: return "disabled";
    default:           return "?";
    }
}
