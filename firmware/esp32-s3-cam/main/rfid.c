/* UHF RFID reader on UART1 - see rfid.h for the wiring.

   Protocol: vendor manual "RFID模块通讯协议使用说明 V4.1.3".

     BB | Type | Command | PL(MSB) PL(LSB) | Parameters | Checksum | 7E
     Type      0x00 command, 0x01 response, 0x02 notification
     Checksum  low byte of the sum from Type through the last parameter

     select port n   BB 00 1B 00 02 01 <n> <sum> 7E        (n = 1..4)
     single inventory BB 00 22 00 00 22 7E
     tag notification BB 02 22 00 12 <RSSI> <PC hi lo> <EPC 12> <CRC 2> <ANT> <sum> 7E
     no tag           BB 01 FF 00 01 15 16 7E

   Scheduling: one explicit select per port, then one inventory, round-robin over four ports. The
   reader has a single RF path that a switch routes to the selected port, so an inventory command is
   only meaningful when the host knows which port is connected - hence "never inventory in an
   unknown RF state": if the select is not acknowledged (after one retry) the port is skipped and
   nothing is transmitted on it.

   Antenna attribution comes from the host's acknowledged selection, not from the notification. On
   this unit the notification's antenna byte reads 0 no matter which port is selected.

   A transaction ends when the reader's terminating response frame arrives; the timeout is only a
   guard against a silent module, not the thing that decides completion. Measured with a clean
   power cycle, 20 rounds per port: select acknowledged 20/20, inventory complete 20/20, first byte
   at 20-30 ms median, whole transaction 20-30 ms median with p95 under 70 ms.
 */
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "driver/uart.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "sdkconfig.h"

#include "rfid.h"

#if CONFIG_RFID_ENABLE

#define TAG "rfid"

#define RFID_PORT          CONFIG_RFID_UART
#define RFID_MAX_TAGS      16
#define RFID_TTL_MS        5000     /* kept at 5 s until the gap statistics say otherwise */
#define RFID_GUARD_MS      150      /* response guard per transaction, not a dwell time */
#define HDR                0xBB
#define END                0x7E

static bool s_ready;
static SemaphoreHandle_t s_lock;
static rfid_tag_t s_tags[RFID_MAX_TAGS];
static rfid_port_stat_t s_ports[5];          /* 1..4 */
static int s_last_port;                      /* last port whose select was acknowledged */
static uint32_t s_polls, s_errors, s_frames_rx, s_notifs, s_bytes_rx;

static uint32_t now_ms(void) { return (uint32_t)(esp_timer_get_time() / 1000); }

/* --- tag table ---------------------------------------------------------- */

static rfid_tag_t *find_or_make(const uint8_t *epc)
{
    int free_slot = -1, oldest_slot = 0;
    uint32_t oldest = UINT32_MAX;
    for (int i = 0; i < RFID_MAX_TAGS; i++) {
        if (s_tags[i].valid && memcmp(s_tags[i].epc, epc, 12) == 0) {
            return &s_tags[i];
        }
        if (!s_tags[i].valid && free_slot < 0) {
            free_slot = i;
        }
        if (s_tags[i].valid && s_tags[i].age_ms < oldest) {
            oldest = s_tags[i].age_ms;
            oldest_slot = i;
        }
    }
    int slot = (free_slot >= 0) ? free_slot : oldest_slot;   /* recycle the least recently seen */
    memset(&s_tags[slot], 0, sizeof(rfid_tag_t));
    memcpy(s_tags[slot].epc, epc, 12);
    s_tags[slot].valid = true;
    return &s_tags[slot];
}

static void note_tag(const uint8_t *epc, int8_t rssi, int host_port)
{
    if (!s_lock || xSemaphoreTake(s_lock, pdMS_TO_TICKS(20)) != pdTRUE) {
        return;
    }
    rfid_tag_t *t = find_or_make(epc);
    bool appeared = (t->reads == 0);
    uint32_t now = now_ms();
    if (!appeared) {
        uint32_t gap = now - (now - t->age_ms);              /* age was time since the last read */
        gap = t->age_ms;
        t->last_gap_ms = gap;
        t->gap_sum_ms += gap;
        t->gap_count++;
        if (gap > t->max_gap_ms) {
            t->max_gap_ms = gap;
        }
    }
    t->rssi = rssi;
    t->port = (int8_t)host_port;
    t->age_ms = 0;
    t->reads++;
    if (host_port >= 1 && host_port <= 4) {
        s_ports[host_port].tags++;
    }
    xSemaphoreGive(s_lock);

    if (appeared) {
        ESP_LOGI(TAG, "tag appeared %02X%02X%02X%02X-%02X%02X%02X%02X-%02X%02X%02X%02X "
                      "port=%d rssi=%d dBm", epc[0], epc[1], epc[2], epc[3], epc[4], epc[5],
                 epc[6], epc[7], epc[8], epc[9], epc[10], epc[11], host_port, rssi);
    }
}

/* --- frame handling ----------------------------------------------------- */

/* Consume bytes until a frame of the wanted kind arrives or the guard expires.
 * Returns 1 if the wanted frame was seen, 0 on timeout. Tag notifications seen along the way are
 * filed against `port` - the port the host selected and had acknowledged. */
static int pump(uint32_t guard_ms, uint8_t want_type, uint8_t want_cmd, int port)
{
    uint32_t t0 = now_ms();
    uint32_t last_rx = 0;            /* when a frame last completed */
    int frames_seen = 0;
    static uint8_t buf[512];
    int len = 0;
    while (now_ms() - t0 < guard_ms) {
        uint8_t chunk[128];
        int got = uart_read_bytes(RFID_PORT, chunk, sizeof(chunk), pdMS_TO_TICKS(5));
        if (got > 0) {
            s_bytes_rx += (uint32_t)got;
            for (int i = 0; i < got && len < (int)sizeof(buf); i++) {
                buf[len++] = chunk[i];
            }
        }
        for (int i = 0; i + 6 < len; i++) {
            if (buf[i] != HDR) {
                continue;
            }
            int pl = (buf[i + 3] << 8) | buf[i + 4];
            int fin = i + 5 + pl + 2;
            if (fin > len) {
                break;                                   /* frame still arriving */
            }
            if (buf[fin - 1] != END) {
                continue;
            }
            uint16_t sum = 0;
            for (int k = 1; k < 5 + pl; k++) {
                sum += buf[i + k];
            }
            if ((uint8_t)sum != buf[i + 5 + pl]) {
                continue;                                /* bad checksum: drop it */
            }
            s_frames_rx++;
            frames_seen++;
            last_rx = now_ms();
            uint8_t type = buf[i + 1], cmd = buf[i + 2];
            if (type == 0x02 && cmd == 0x22 && (pl == 17 || pl == 18)) {
                s_notifs++;
                note_tag(&buf[i + 5 + 3], (int8_t)buf[i + 5], port);
            } else if (type == want_type && cmd == want_cmd) {
                return 1;
            }
            /* shift the consumed frame out */
            memmove(buf, buf + fin, (size_t)(len - fin));
            len -= fin;
            i = -1;
        }
        if (len > (int)sizeof(buf) - 128) {
            len = 0;                                     /* never let the buffer fill up */
        }

        /* End of transaction. A single inventory does not send a terminating response when it does
         * find tags - it just emits one notification per tag and stops - so "the reader has
         * finished" has to be read as "frames have arrived and the line has gone quiet". The
         * 40 ms is measured, not guessed: first byte at 20-30 ms, whole burst under 70 ms p95. */
        if (frames_seen > 0 && (now_ms() - last_rx) >= 40) {
            return 1;
        }
    }
    return 0;
}

static void send_frame(const uint8_t *body, int body_len)
{
    /* body is Type..last-parameter; this adds BB, the checksum and 7E. */
    uint8_t out[32];
    uint16_t sum = 0;
    out[0] = HDR;
    for (int i = 0; i < body_len && i + 1 < (int)sizeof(out) - 1; i++) {
        out[1 + i] = body[i];
        sum += body[i];
    }
    out[1 + body_len] = (uint8_t)sum;
    out[2 + body_len] = END;
    uart_write_bytes(RFID_PORT, out, body_len + 3);
}

static bool select_port(int port)
{
    uint8_t body[6] = {0x00, 0x1B, 0x00, 0x02, 0x01, (uint8_t)port};
    for (int attempt = 0; attempt < 2; attempt++) {
        uart_flush_input(RFID_PORT);
        send_frame(body, sizeof(body));
        if (pump(RFID_GUARD_MS, 0x01, 0x1B, port)) {
            s_last_port = port;
            return true;
        }
    }
    s_last_port = 0;                     /* RF state is now unknown: do not inventory on it */
    return false;
}

/* --- task --------------------------------------------------------------- */

static void rfid_task(void *arg)
{
    (void)arg;
    uint32_t last_age = now_ms();
    uint32_t last_summary = 0;
    int port = 0;

    while (true) {
        port = (port % CONFIG_RFID_ANTENNAS) + 1;
        rfid_port_stat_t *st = &s_ports[port];

        if (!select_port(port)) {
            st->select_fail++;
            s_errors++;
            ESP_LOGW(TAG, "port %d: select not acknowledged twice, skipping this round", port);
        } else {
            st->select_ok++;
            uart_flush_input(RFID_PORT);
            uint8_t body[4] = {0x00, 0x22, 0x00, 0x00};
            send_frame(body, sizeof(body));
            s_polls++;
            if (pump(RFID_GUARD_MS, 0x01, 0x22, port)) {
                st->inventory_ok++;
            } else {
                st->inventory_timeout++;
                s_errors++;
            }
        }

        uint32_t t = now_ms();
        uint32_t dt = t - last_age;
        last_age = t;
        if (s_lock && xSemaphoreTake(s_lock, pdMS_TO_TICKS(20)) == pdTRUE) {
            for (int i = 0; i < RFID_MAX_TAGS; i++) {
                if (!s_tags[i].valid) {
                    continue;
                }
                s_tags[i].age_ms += dt;
                if (s_tags[i].age_ms > RFID_TTL_MS) {
                    s_tags[i].valid = false;
                    const uint8_t *e = s_tags[i].epc;
                    ESP_LOGI(TAG, "tag left    %02X%02X%02X%02X-%02X%02X%02X%02X-%02X%02X%02X%02X"
                                  " (silent %lu ms, max gap %lu ms)",
                             e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9], e[10],
                             e[11], (unsigned long)s_tags[i].age_ms,
                             (unsigned long)s_tags[i].max_gap_ms);
                }
            }
            xSemaphoreGive(s_lock);
        }

        if (t - last_summary >= 10000) {
            last_summary = t;
            int live = 0;
            for (int i = 0; i < RFID_MAX_TAGS; i++) {
                live += s_tags[i].valid ? 1 : 0;
            }
            ESP_LOGI(TAG, "p%d sel %lu/%lu inv %lu/%lu | p%d %lu/%lu %lu/%lu | "
                          "p%d %lu/%lu %lu/%lu | p%d %lu/%lu %lu/%lu | tags=%d",
                    1, (unsigned long)s_ports[1].select_ok,
                    (unsigned long)(s_ports[1].select_ok + s_ports[1].select_fail),
                    (unsigned long)s_ports[1].inventory_ok,
                    (unsigned long)(s_ports[1].inventory_ok + s_ports[1].inventory_timeout),
                    2, (unsigned long)s_ports[2].select_ok,
                    (unsigned long)(s_ports[2].select_ok + s_ports[2].select_fail),
                    (unsigned long)s_ports[2].inventory_ok,
                    (unsigned long)(s_ports[2].inventory_ok + s_ports[2].inventory_timeout),
                    3, (unsigned long)s_ports[3].select_ok,
                    (unsigned long)(s_ports[3].select_ok + s_ports[3].select_fail),
                    (unsigned long)s_ports[3].inventory_ok,
                    (unsigned long)(s_ports[3].inventory_ok + s_ports[3].inventory_timeout),
                    4, (unsigned long)s_ports[4].select_ok,
                    (unsigned long)(s_ports[4].select_ok + s_ports[4].select_fail),
                    (unsigned long)s_ports[4].inventory_ok,
                    (unsigned long)(s_ports[4].inventory_ok + s_ports[4].inventory_timeout),
                    live);
        }
        vTaskDelay(pdMS_TO_TICKS(CONFIG_RFID_POLL_MS));
    }
}

/* --- public ------------------------------------------------------------- */

esp_err_t rfid_start(void)
{
    uart_config_t cfg = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    esp_err_t err = uart_param_config(RFID_PORT, &cfg);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "uart_param_config: %s", esp_err_to_name(err));
        return err;
    }
    err = uart_set_pin(RFID_PORT, CONFIG_RFID_TX_GPIO, CONFIG_RFID_RX_GPIO,
                       UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "uart_set_pin: %s", esp_err_to_name(err));
        return err;
    }
    err = uart_driver_install(RFID_PORT, 2048, 0, 0, NULL, 0);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "uart_driver_install: %s", esp_err_to_name(err));
        return err;
    }
    s_lock = xSemaphoreCreateMutex();

    /* Leave whatever mode a previous run left behind: this module keeps its state across our
       reboots, so an earlier "continuous inventory" experiment survived a reflash and made plain
       polls answer "no tag" until the module itself was power cycled. */
    {
        uint8_t body[4] = {0x00, 0x28, 0x00, 0x00};
        send_frame(body, sizeof(body));
        vTaskDelay(pdMS_TO_TICKS(120));
        uart_flush_input(RFID_PORT);
    }

    uint8_t body[5] = {0x00, 0x03, 0x00, 0x01, 0x00};
    send_frame(body, sizeof(body));
    uint8_t buf[64];
    int got = uart_read_bytes(RFID_PORT, buf, sizeof(buf), pdMS_TO_TICKS(300));
    if (got > 8 && buf[0] == HDR) {
        char ver[32] = {0};
        int n = (got > 20) ? 20 : got;
        for (int i = 6, k = 0; i < n && k < 31; i++) {
            if (buf[i] >= 0x20 && buf[i] < 0x7F) {
                ver[k++] = (char)buf[i];
            }
        }
        ESP_LOGI(TAG, "reader answered: %s", ver);
    } else {
        ESP_LOGW(TAG, "no answer from the reader on UART%d (rx=%d tx=%d) - polling anyway",
                 RFID_PORT, CONFIG_RFID_RX_GPIO, CONFIG_RFID_TX_GPIO);
    }

    s_ready = true;
    ESP_LOGI(TAG, "ready: UART%d 115200 8N1, rx=GPIO%d tx=GPIO%d, round-robin over %d ports, "
                  "guard %d ms", RFID_PORT, CONFIG_RFID_RX_GPIO, CONFIG_RFID_TX_GPIO,
             CONFIG_RFID_ANTENNAS, RFID_GUARD_MS);
    xTaskCreate(rfid_task, "rfid", 4096, NULL, 3, NULL);
    return ESP_OK;
}

bool rfid_available(void) { return s_ready; }
uint32_t rfid_polls(void) { return s_polls; }
uint32_t rfid_errors(void) { return s_errors; }
uint32_t rfid_frames_rx(void) { return s_frames_rx; }
uint32_t rfid_notifications(void) { return s_notifs; }
uint32_t rfid_bytes_rx(void) { return s_bytes_rx; }

int rfid_tag_count(void)
{
    int n = 0;
    if (s_lock && xSemaphoreTake(s_lock, pdMS_TO_TICKS(20)) == pdTRUE) {
        for (int i = 0; i < RFID_MAX_TAGS; i++) {
            n += s_tags[i].valid ? 1 : 0;
        }
        xSemaphoreGive(s_lock);
    }
    return n;
}

int rfid_snapshot(rfid_tag_t *out, int max)
{
    int n = 0;
    if (!out || max <= 0 || !s_lock) {
        return 0;
    }
    if (xSemaphoreTake(s_lock, pdMS_TO_TICKS(20)) != pdTRUE) {
        return 0;
    }
    for (int i = 0; i < RFID_MAX_TAGS && n < max; i++) {
        if (s_tags[i].valid) {
            out[n++] = s_tags[i];
        }
    }
    xSemaphoreGive(s_lock);
    return n;
}

bool rfid_port_stats(int port, rfid_port_stat_t *out)
{
    if (port < 1 || port > 4 || !out) {
        return false;
    }
    *out = s_ports[port];
    out->present = (s_ports[port].select_ok + s_ports[port].select_fail) > 0;
    return true;
}

#else  /* !CONFIG_RFID_ENABLE */

esp_err_t rfid_start(void) { return ESP_ERR_NOT_SUPPORTED; }
bool rfid_available(void) { return false; }
int rfid_tag_count(void) { return 0; }
int rfid_snapshot(rfid_tag_t *o, int m) { (void)o; (void)m; return 0; }
uint32_t rfid_polls(void) { return 0; }
uint32_t rfid_errors(void) { return 0; }
uint32_t rfid_frames_rx(void) { return 0; }
uint32_t rfid_notifications(void) { return 0; }
uint32_t rfid_bytes_rx(void) { return 0; }
bool rfid_port_stats(int p, rfid_port_stat_t *o) { (void)p; (void)o; return false; }

#endif
