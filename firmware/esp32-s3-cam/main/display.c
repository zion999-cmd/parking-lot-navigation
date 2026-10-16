#include "display.h"

#include <stdio.h>
#include <string.h>

#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>
#include <freertos/task.h>

#include <driver/ledc.h>
#include <driver/spi_common.h>
#include <esp_heap_caps.h>
#include <esp_heap_caps.h>
#include <esp_lcd_panel_io.h>
#include <esp_lcd_panel_ops.h>
#include <esp_lcd_panel_vendor.h>
#include <esp_timer.h>
#include <esp_log.h>

#include "font8x16.h"
#include "qr_code.h"
#include <sdkconfig.h>

/*
 * On-device status panel for the ESP32-S3-CAM.
 *
 * Every hardware value below is taken from the reference project's working
 * implementation, cross-checked against the ESP32S3-CAM-KZB-V2_0 schematic:
 *
 *   reference: /Users/bx/Esp32/xiaozhi-esp32-224
 *              main/boards/bread-compact-wifi-s3cam/{config.h,compact_wifi_board_s3cam.cc}
 *              plus its sdkconfig CONFIG_LCD_ST7789_240X240=y
 *
 *   controller      ST7789 via esp_lcd_new_panel_st7789 (esp_lcd, not LVGL)
 *   panel           CONFIG_LCD_ST7789_240X240 -> 240x240, invert on,
 *                   no mirror, no swap, offset 0/0, SPI mode 0, RGB order
 *   bus             SPI3_HOST, 40 MHz, 16 bpp, queue depth 10
 *   pins            CS 45, DC 47, RST 21, SCLK 19, MOSI 20, backlight 38
 *   backlight       LEDC PWM, output not inverted (active high)
 *
 * The schematic lists exactly the same six signals (GPIO19 CLK, GPIO20 MOSI,
 * GPIO21 RST, GPIO47 DC, GPIO38 BL, GPIO45 CS), so nothing here is guessed.
 *
 * Note on LEDC: camera.c already uses LEDC_TIMER_0 / LEDC_CHANNEL_0 for the
 * camera XCLK, so the backlight deliberately uses timer/channel 1.
 */

static const char *TAG = "display";

#define LCD_HOST        SPI3_HOST
#define LCD_PIN_CS      45
#define LCD_PIN_DC      47
#define LCD_PIN_RST     21
#define LCD_PIN_SCLK    19
#define LCD_PIN_MOSI    20
#define LCD_PIN_BL      38

#define LCD_W           240
#define LCD_H           240
#define LCD_PCLK_HZ     (40 * 1000 * 1000)

#define BL_LEDC_TIMER   LEDC_TIMER_1
#define BL_LEDC_CHANNEL LEDC_CHANNEL_1
#define BL_DUTY_PERCENT 75

/*
 * The panel is flushed in horizontal bands, not in one call.
 *
 * The frame buffer lives in PSRAM, and the SPI DMA engine cannot read from
 * PSRAM, so the driver has to bounce every transfer through an internal
 * DMA-capable buffer of the *transfer* size. Asking for the whole 240x240 frame
 * at once therefore wants ~115 KB of internal RAM, and on this board the internal
 * heap only has ~38 KB free - less than 20 KB while an OTA check is running. The
 * allocation failed, the panel reported "Failed to allocate priv TX buffer" and
 * the draw call hung waiting for a completion callback that never came.
 *
 * BAND_LINES = 16 keeps the bounce buffer at 240*16*2 = 7.5 KB.
 */
#define BAND_LINES 16
#define BAND_BYTES (LCD_W * BAND_LINES * sizeof(uint16_t))

#define COL_BG      0x0000  /* black  */
#define COL_FG      0xFFFF  /* white  */
#define COL_DIM     0x8410  /* grey   */
#define COL_ACCENT  0x07FF  /* cyan   */
#define COL_WARN    0xFD20  /* orange */

/* Pure black on pure white: a QR code read by a phone camera is one of the few
 * places where "use the exact values" matters more than "match the theme". */
#define COL_QR_DARK  0x0000
#define COL_QR_LIGHT 0xFFFF

#define TEXT_X      8
#define TEXT_Y      16
#define LINE_H      20
#define MAX_LINES   11      /* (240 - TEXT_Y) / LINE_H */

static esp_lcd_panel_handle_t s_panel = NULL;
static uint16_t *s_fb = NULL;
static bool s_ready = false;
static int s_flush_fail_streak = 0;
static int64_t s_flush_paused_until_us = 0;
static uint32_t s_flush_count = 0;

/* ------------------------------------------------------------------------- */
/* Message queue                                                              */
/* ------------------------------------------------------------------------- */

typedef enum {
    MSG_BOOT,
    MSG_PROVISIONING,
    MSG_WIFI_CONNECTING,
    MSG_READY,
    MSG_CAMERA_ERROR,
    MSG_WIFI_ERROR,
    MSG_OTA_CHECKING,
    MSG_OTA_PROGRESS,
    MSG_OTA_OK,
    MSG_OTA_FAIL,
    MSG_ERROR,
    MSG_QR,
} msg_kind_t;

/*
 * OTA screens *and the QR screen* are transient overlays: they cover whatever the
 * device was showing and are expected to be undone. Everything else is a "base"
 * screen - the steady state a human should see.
 *
 * This exists because every way an OTA check can end (fetch failure, invalid
 * manifest, already up to date, download or SHA256 failure) has to put the base
 * screen back. Otherwise the panel stays stuck on "Checking update..." forever
 * and the one thing it must always show - the device IP - disappears.
 *
 * The QR is in the same category for the same reason: it is shown because
 * something is happening (a car at the gate, or the boot self test), and when
 * that stops the panel has to go back to the IP without anyone having to
 * remember what it was.
 */
static bool msg_is_transient(msg_kind_t kind)
{
    return kind == MSG_OTA_CHECKING || kind == MSG_OTA_PROGRESS ||
           kind == MSG_OTA_OK || kind == MSG_OTA_FAIL || kind == MSG_QR;
}

typedef struct {
    msg_kind_t kind;
    char a[40];
    char b[40];
    int num;
} disp_msg_t;

static QueueHandle_t s_queue = NULL;
static disp_msg_t s_base;
static bool s_base_valid = false;

static void enqueue(const disp_msg_t *m)
{
    if (s_queue == NULL) {
        return;
    }
    if (xQueueSend(s_queue, m, 0) != pdTRUE) {
        /* Display is best effort: drop the oldest message rather than block or
         * stall the caller (which may be the Wi-Fi event task). */
        disp_msg_t dropped;
        xQueueReceive(s_queue, &dropped, 0);
        xQueueSend(s_queue, m, 0);
    }
}

static void post(const disp_msg_t *m)
{
    if (!msg_is_transient(m->kind)) {
        s_base = *m;
        s_base_valid = true;
    }
    enqueue(m);
}

/* Put the last steady-state screen back after a transient overlay. */
void display_restore(void)
{
    if (s_base_valid) {
        enqueue(&s_base);
    }
}

static disp_msg_t make_msg(msg_kind_t kind)
{
    disp_msg_t m = {};
    m.kind = kind;
    return m;
}

/* ------------------------------------------------------------------------- */
/* Drawing primitives                                                         */
/* ------------------------------------------------------------------------- */

static void fill(uint16_t colour)
{
    for (int i = 0; i < LCD_W * LCD_H; i++) {
        s_fb[i] = colour;
    }
}

static void fill_rect(int x, int y, int w, int h, uint16_t colour)
{
    for (int ry = y; ry < y + h && ry < LCD_H; ry++) {
        if (ry < 0) {
            continue;
        }
        for (int rx = x; rx < x + w && rx < LCD_W; rx++) {
            if (rx < 0) {
                continue;
            }
            s_fb[ry * LCD_W + rx] = colour;
        }
    }
}

static void draw_char(int x, int y, char c, uint16_t colour, int scale)
{
    if (c < DISPLAY_FONT_FIRST || c > DISPLAY_FONT_LAST) {
        c = '?';
    }
    const uint8_t *glyph = display_font[(int)c - DISPLAY_FONT_FIRST];

    for (int row = 0; row < DISPLAY_FONT_H; row++) {
        uint8_t bits = glyph[row];
        if (bits == 0) {
            continue;
        }
        for (int col = 0; col < DISPLAY_FONT_W; col++) {
            if ((bits & (1 << (7 - col))) == 0) {
                continue;
            }
            int px = x + col * scale;
            int py = y + row * scale;
            fill_rect(px, py, scale, scale, colour);
        }
    }
}

static void draw_text(int x, int y, const char *text, uint16_t colour, int scale)
{
    if (text == NULL) {
        return;
    }
    int cursor = x;
    for (const char *p = text; *p != '\0'; p++) {
        if (cursor + DISPLAY_FONT_W * scale > LCD_W) {
            break;              /* no wrapping: screens are laid out by hand */
        }
        draw_char(cursor, y, *p, colour, scale);
        cursor += DISPLAY_FONT_W * scale;
    }
}

static void flush(void)
{
    if (s_panel == NULL || s_fb == NULL) {
        return;
    }
    /*
     * Back off instead of hammering. A draw failure means the SPI driver could
     * not get its bounce buffer (usually because internal RAM is tight, for
     * example during an OTA check). Retrying immediately makes it worse and
     * floods the log, so pause for a while and keep the panel usable afterwards.
     */
    int64_t now = esp_timer_get_time();
    if (now < s_flush_paused_until_us) {
        return;
    }

    for (int y = 0; y < LCD_H; y += BAND_LINES) {
        int y_end = y + BAND_LINES;
        if (y_end > LCD_H) {
            y_end = LCD_H;
        }
        esp_err_t err = esp_lcd_panel_draw_bitmap(s_panel, 0, y, LCD_W, y_end,
                                                 s_fb + (size_t)y * LCD_W);
        if (err != ESP_OK) {
            s_flush_fail_streak++;
            if (s_flush_fail_streak == 1) {
                ESP_LOGW(TAG, "panel draw failed at y=%d: %s - internal RAM is "
                              "tight (free internal heap was %u bytes). Pausing "
                              "the status panel; the camera keeps running.",
                         y, esp_err_to_name(err),
                         (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
            }
            if (s_flush_fail_streak >= 3) {
                s_flush_paused_until_us = now + 10 * 1000 * 1000;
                s_flush_fail_streak = 0;
            }
            return;
        }
    }
    s_flush_fail_streak = 0;
    s_flush_count++;
}

/* ------------------------------------------------------------------------- */
/* QR screen                                                                  */
/* ------------------------------------------------------------------------- */

/*
 * The quiet zone is part of the code, not decoration: a scanner uses the white
 * border to find where the code starts. Four modules is the spec's minimum, and
 * the scale is chosen so a full four module border fits - which is also what
 * makes the modules as large as they can be:
 *
 *   (25 + 2*4) modules * 7 px = 231 px of 240, leaving 32 px each side.
 *
 * The payload decides the version and therefore the module size, which is why the
 * QR points at /go on the Mac's .local name (38 characters, version 3, 6 px per
 * module) rather than straight at /web/navigation.html (55, version 4, 5 px).
 *
 * A name instead of an address is what makes the code survive a change of Wi-Fi:
 * no IP in it means nothing to go stale, and macOS already answers for
 * <hostname>.local on any network it joins. The price is one version - an address
 * would fit in version 2 and give 7 px per module - which is the trade taken
 * because a QR that has to be regenerated after every move is worse than one that
 * is 14% smaller.
 *
 * Anything bigger than 8 px per module would eat into the quiet zone, and a code
 * that is 12% larger but unreadable is not larger at all. The scale is computed
 * rather than written down so that a different payload (a real URL, later) with a
 * different version still comes out scaled to the screen.
 */
#define QR_QUIET_MODULES 4

static void draw_qr(void)
{
    const int total_modules = QR_SIZE + 2 * QR_QUIET_MODULES;
    int scale = LCD_W / total_modules;
    if (scale < 1) {
        scale = 1;                  /* pathological payload: draw it anyway */
    }
    const int side = QR_SIZE * scale;
    const int origin_x = (LCD_W - side) / 2;
    const int origin_y = (LCD_H - side) / 2;

    /* White first: this is the quiet zone as well as the light modules, and
     * drawing dark-on-dark would be the one thing a scanner cannot fix. */
    fill(COL_QR_LIGHT);

    for (int row = 0; row < QR_SIZE; row++) {
        for (int col = 0; col < QR_SIZE; col++) {
            if ((qr_rows[row][col / 8] >> (col % 8)) & 1) {
                /* Integer-aligned squares at the module size: no scaling, no
                 * interpolation, sharp edges. */
                fill_rect(origin_x + col * scale, origin_y + row * scale,
                          scale, scale, COL_QR_DARK);
            }
        }
    }
}

/* ------------------------------------------------------------------------- */
/* Screens                                                                    */
/* ------------------------------------------------------------------------- */

/* Left-aligned block of lines, each with its own colour. */
static void draw_block(const char *const *lines, const uint16_t *colours, int count)
{
    int y = TEXT_Y;
    for (int i = 0; i < count && i < MAX_LINES; i++) {
        draw_text(TEXT_X, y, lines[i], colours[i], 1);
        y += LINE_H;
    }
}

static void render(const disp_msg_t *m)
{
    /* 96 bytes so that concatenating two 40 byte fields cannot truncate. */
    char line_a[96];
    char line_b[96];

    switch (m->kind) {
    case MSG_BOOT: {
        const char *lines[] = { "Parking Camera", "Booting..." };
        snprintf(line_a, sizeof(line_a), "v%s", m->a);
        const char *all[] = { lines[0], lines[1], line_a };
        const uint16_t cols[] = { COL_FG, COL_DIM, COL_ACCENT };
        fill(COL_BG);
        draw_block(all, cols, 3);
        break;
    }

    case MSG_PROVISIONING: {
        snprintf(line_a, sizeof(line_a), "Connect to:");
        snprintf(line_b, sizeof(line_b), "%s", m->a);
        const char *all[] = { "Wi-Fi Setup", "", line_a, line_b, "", "Then open:", m->b };
        const uint16_t cols[] = { COL_ACCENT, COL_BG, COL_FG, COL_WARN, COL_BG, COL_FG, COL_WARN };
        fill(COL_BG);
        draw_block(all, cols, 7);
        break;
    }

    case MSG_WIFI_CONNECTING: {
        snprintf(line_a, sizeof(line_a), "%s", m->a);
        const char *all[] = { "Connecting Wi-Fi...", "", line_a };
        const uint16_t cols[] = { COL_FG, COL_BG, COL_ACCENT };
        fill(COL_BG);
        draw_block(all, cols, 3);
        break;
    }

    case MSG_READY: {
        snprintf(line_a, sizeof(line_a), "IP: %s", m->a);
        const char *all[] = { "Camera Ready", "", "Wi-Fi: Connected", line_a, "", "/stream" };
        const uint16_t cols[] = { COL_FG, COL_BG, COL_FG, COL_ACCENT, COL_BG, COL_DIM };
        fill(COL_BG);
        draw_block(all, cols, 6);
        break;
    }

    case MSG_CAMERA_ERROR: {
        const char *all[] = { "Camera ERROR", "", "Check camera cable" };
        const uint16_t cols[] = { COL_WARN, COL_BG, COL_FG };
        fill(COL_BG);
        draw_block(all, cols, 3);
        break;
    }

    case MSG_WIFI_ERROR: {
        const char *all[] = { "Wi-Fi unavailable", "", "Starting setup mode..." };
        const uint16_t cols[] = { COL_WARN, COL_BG, COL_FG };
        fill(COL_BG);
        draw_block(all, cols, 3);
        break;
    }

    case MSG_OTA_CHECKING: {
        const char *all[] = { "Checking update..." };
        const uint16_t cols[] = { COL_FG };
        fill(COL_BG);
        draw_block(all, cols, 1);
        break;
    }

    case MSG_OTA_PROGRESS: {
        snprintf(line_a, sizeof(line_a), "%s -> %s", m->a, m->b);
        snprintf(line_b, sizeof(line_b), "%d%%", m->num);
        fill(COL_BG);
        draw_text(TEXT_X, TEXT_Y, "Updating...", COL_FG, 1);
        draw_text(TEXT_X, TEXT_Y + LINE_H, line_a, COL_ACCENT, 1);
        draw_text(TEXT_X, TEXT_Y + LINE_H * 3, line_b, COL_FG, 1);

        /* simple progress bar */
        int bar_x = TEXT_X + 48;
        int bar_y = TEXT_Y + LINE_H * 3 + 3;
        int bar_w = LCD_W - bar_x - TEXT_X;
        int filled = bar_w * (m->num < 0 ? 0 : (m->num > 100 ? 100 : m->num)) / 100;
        fill_rect(bar_x, bar_y, bar_w, 10, COL_DIM);
        fill_rect(bar_x, bar_y, filled, 10, COL_ACCENT);
        break;
    }

    case MSG_OTA_OK: {
        const char *all[] = { "Update complete", "Rebooting..." };
        const uint16_t cols[] = { COL_ACCENT, COL_FG };
        fill(COL_BG);
        draw_block(all, cols, 2);
        break;
    }

    case MSG_OTA_FAIL: {
        const char *all[] = { "Update failed", "", "Current firmware safe" };
        const uint16_t cols[] = { COL_WARN, COL_BG, COL_FG };
        fill(COL_BG);
        draw_block(all, cols, 3);
        break;
    }

    case MSG_ERROR: {
        snprintf(line_a, sizeof(line_a), "%s", m->a);
        snprintf(line_b, sizeof(line_b), "%s", m->b);
        const char *all[] = { line_a, "", line_b };
        const uint16_t cols[] = { COL_WARN, COL_BG, COL_FG };
        fill(COL_BG);
        draw_block(all, cols, 3);
        break;
    }

    case MSG_QR:
        /* Nothing but the code: at 240 px there is no room for a code that can be
         * scanned and a label that can be read, and the code is the one that has
         * a job to do. m->a carries the payload for the log, not for the panel. */
        draw_qr();
        break;
    }

    flush();
}

static void display_task(void *arg)
{
    (void)arg;
    disp_msg_t m;

    while (true) {
        if (xQueueReceive(s_queue, &m, portMAX_DELAY) == pdTRUE) {
            render(&m);
        }
    }
}

/* ------------------------------------------------------------------------- */
/* Hardware                                                                   */
/* ------------------------------------------------------------------------- */

static void backlight_init(void)
{
    ledc_timer_config_t timer = {};
    timer.speed_mode = LEDC_LOW_SPEED_MODE;
    timer.duty_resolution = LEDC_TIMER_10_BIT;
    timer.timer_num = BL_LEDC_TIMER;
    timer.freq_hz = 5000;
    timer.clk_cfg = LEDC_AUTO_CLK;
    ESP_ERROR_CHECK(ledc_timer_config(&timer));

    ledc_channel_config_t ch = {};
    ch.gpio_num = LCD_PIN_BL;
    ch.speed_mode = LEDC_LOW_SPEED_MODE;
    ch.channel = BL_LEDC_CHANNEL;
    ch.timer_sel = BL_LEDC_TIMER;
    ch.duty = 0;
    ch.hpoint = 0;
    ESP_ERROR_CHECK(ledc_channel_config(&ch));

    /* DISPLAY_BACKLIGHT_OUTPUT_INVERT is false in the reference config, so a
     * higher duty means brighter. */
    ESP_ERROR_CHECK(ledc_set_duty(LEDC_LOW_SPEED_MODE, BL_LEDC_CHANNEL,
                                  (1023 * BL_DUTY_PERCENT) / 100));
    ESP_ERROR_CHECK(ledc_update_duty(LEDC_LOW_SPEED_MODE, BL_LEDC_CHANNEL));
}

bool display_init(void)
{
#if defined(CONFIG_DISPLAY_ENABLE) && !CONFIG_DISPLAY_ENABLE
    ESP_LOGI(TAG, "Status display disabled at build time (CONFIG_DISPLAY_ENABLE=n)");
    return false;
#else
    if (s_ready) {
        return true;
    }

    ESP_LOGI(TAG, "Status display: ST7789 240x240 on SPI3 (cs=%d dc=%d rst=%d sclk=%d mosi=%d bl=%d)",
             LCD_PIN_CS, LCD_PIN_DC, LCD_PIN_RST, LCD_PIN_SCLK, LCD_PIN_MOSI, LCD_PIN_BL);

    spi_bus_config_t bus = {};
    bus.mosi_io_num = LCD_PIN_MOSI;
    bus.miso_io_num = -1;
    bus.sclk_io_num = LCD_PIN_SCLK;
    bus.quadwp_io_num = -1;
    bus.quadhd_io_num = -1;
    bus.max_transfer_sz = BAND_BYTES;

    esp_err_t err = spi_bus_initialize(LCD_HOST, &bus, SPI_DMA_CH_AUTO);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "spi_bus_initialize failed: %s", esp_err_to_name(err));
        return false;
    }

    esp_lcd_panel_io_handle_t io = NULL;
    esp_lcd_panel_io_spi_config_t io_cfg = {};
    io_cfg.cs_gpio_num = LCD_PIN_CS;
    io_cfg.dc_gpio_num = LCD_PIN_DC;
    io_cfg.spi_mode = 0;
    io_cfg.pclk_hz = LCD_PCLK_HZ;
    /*
     * Two transactions in flight, not ten.
     *
     * The SPI driver cannot read the frame buffer from PSRAM, so every band is copied
     * through an internal DMA-capable bounce buffer of 7.5 KB. A depth of 10 let a
     * single redraw hold up to 75 KB of that memory - the same internal RAM the Wi-Fi
     * transmit path draws from on a board that only has ~123 KB free while streaming.
     * Depth 2 caps the panel's appetite at 15 KB.
     */
    io_cfg.trans_queue_depth = 2;
    io_cfg.lcd_cmd_bits = 8;
    io_cfg.lcd_param_bits = 8;
    err = esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)LCD_HOST, &io_cfg, &io);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_lcd_new_panel_io_spi failed: %s", esp_err_to_name(err));
        return false;
    }

    esp_lcd_panel_dev_config_t panel_cfg = {};
    panel_cfg.reset_gpio_num = LCD_PIN_RST;
    panel_cfg.rgb_ele_order = LCD_RGB_ELEMENT_ORDER_RGB;
    panel_cfg.bits_per_pixel = 16;
    err = esp_lcd_new_panel_st7789(io, &panel_cfg, &s_panel);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_lcd_new_panel_st7789 failed: %s", esp_err_to_name(err));
        return false;
    }

    /* Same sequence as the proven reference implementation. */
    ESP_ERROR_CHECK(esp_lcd_panel_reset(s_panel));
    ESP_ERROR_CHECK(esp_lcd_panel_init(s_panel));
    ESP_ERROR_CHECK(esp_lcd_panel_invert_color(s_panel, true));   /* 240x240 variant */
    ESP_ERROR_CHECK(esp_lcd_panel_swap_xy(s_panel, false));
    ESP_ERROR_CHECK(esp_lcd_panel_mirror(s_panel, false, false));
    err = esp_lcd_panel_disp_on_off(s_panel, true);
    if (err == ESP_ERR_NOT_SUPPORTED) {
        ESP_LOGW(TAG, "panel does not support disp_on_off; assuming on");
    } else {
        ESP_ERROR_CHECK(err);
    }

    /* Frame buffer in PSRAM: 240*240*2 = 112.5 KB. */
    s_fb = heap_caps_malloc(LCD_W * LCD_H * sizeof(uint16_t),
                            MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (s_fb == NULL) {
        ESP_LOGE(TAG, "could not allocate the %u byte frame buffer",
                 (unsigned)(LCD_W * LCD_H * sizeof(uint16_t)));
        return false;
    }

    s_queue = xQueueCreate(8, sizeof(disp_msg_t));
    if (s_queue == NULL) {
        ESP_LOGE(TAG, "could not create the display queue");
        return false;
    }

    fill(COL_BG);
    flush();

    backlight_init();

    if (xTaskCreate(display_task, "display", 4096, NULL, 3, NULL) != pdPASS) {
        ESP_LOGE(TAG, "could not start the display task");
        return false;
    }

    s_ready = true;
    ESP_LOGI(TAG, "Status display ready");
    return true;
#endif  /* CONFIG_DISPLAY_ENABLE */
}

bool display_ready(void)
{
    return s_ready;
}

uint32_t display_flush_count(void)
{
    return s_flush_count;
}

/* ------------------------------------------------------------------------- */
/* Reporting API                                                              */
/* ------------------------------------------------------------------------- */

void display_show_boot(const char *version)
{
    disp_msg_t m = make_msg(MSG_BOOT);
    snprintf(m.a, sizeof(m.a), "%s", version ? version : "?");
    post(&m);
}

void display_show_provisioning(const char *ap_ssid, const char *ap_url)
{
    disp_msg_t m = make_msg(MSG_PROVISIONING);
    snprintf(m.a, sizeof(m.a), "%s", ap_ssid ? ap_ssid : "?");
    snprintf(m.b, sizeof(m.b), "%s", ap_url ? ap_url : "192.168.4.1");
    post(&m);
}

void display_show_wifi_connecting(const char *ssid)
{
    disp_msg_t m = make_msg(MSG_WIFI_CONNECTING);
    snprintf(m.a, sizeof(m.a), "%s", ssid ? ssid : "?");
    post(&m);
}

void display_show_ready(const char *ip)
{
    disp_msg_t m = make_msg(MSG_READY);
    snprintf(m.a, sizeof(m.a), "%s", ip ? ip : "0.0.0.0");
    post(&m);
}

void display_show_camera_error(void)
{
    disp_msg_t m = make_msg(MSG_CAMERA_ERROR);
    post(&m);
}

void display_show_wifi_error(void)
{
    disp_msg_t m = make_msg(MSG_WIFI_ERROR);
    post(&m);
}

void display_show_ota_checking(void)
{
    disp_msg_t m = make_msg(MSG_OTA_CHECKING);
    post(&m);
}

void display_show_ota_progress(const char *version_from, const char *version_to, int percent)
{
    disp_msg_t m = make_msg(MSG_OTA_PROGRESS);
    snprintf(m.a, sizeof(m.a), "%s", version_from ? version_from : "?");
    snprintf(m.b, sizeof(m.b), "%s", version_to ? version_to : "?");
    m.num = percent;
    post(&m);
}

void display_show_ota_result(bool ok)
{
    disp_msg_t m = make_msg(ok ? MSG_OTA_OK : MSG_OTA_FAIL);
    post(&m);
}

void display_show_error(const char *title, const char *hint)
{
    disp_msg_t m = make_msg(MSG_ERROR);
    snprintf(m.a, sizeof(m.a), "%s", title ? title : "ERROR");
    snprintf(m.b, sizeof(m.b), "%s", hint ? hint : "");
    post(&m);
}

void display_show_qr(const char *payload)
{
    disp_msg_t m = make_msg(MSG_QR);
    /*
     * The payload does not reach the panel - the modules are a baked constant, so
     * there is nothing to encode at runtime and nothing to allocate. It is carried
     * in the message anyway so that the log says which code is on the glass, which
     * is the thing that a "cannot scan it" report needs to be checked against.
     */
    snprintf(m.a, sizeof(m.a), "%s", payload ? payload : QR_PAYLOAD);
    ESP_LOGI(TAG, "showing QR code: \"%s\" (%d x %d modules)",
             m.a, QR_SIZE, QR_SIZE);
    post(&m);
}
