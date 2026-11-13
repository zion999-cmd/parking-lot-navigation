/* Four magnetometers behind a TCA9548A, on I2C port 1.
 *
 * Why a mux: the HMC5883L answers at 0x1E and that address is not configurable, so four of them
 * on one bus is four devices shouting the same name. The TCA9548A is a switch that connects one
 * channel at a time, so each sensor keeps 0x1E and only the selected one is on the bus.
 *
 * Why port 1: port 0 belongs to the camera's SCCB bus (GPIO4/5, wired in camera.c). Sharing it
 * would put magnetometer traffic in the middle of frame capture.
 *
 * Two chips answer to "HMC5883L" in the shops - the real Honeywell part at 0x1E, and the QMC5883L
 * at 0x0D with a completely different register map. Both are detected here rather than assumed,
 * because guessing wrong produces plausible-looking nonsense.
 *
 * Everything is best effort. A sensor that stops answering must not take the camera, the lights or
 * the gate with it, so no ESP_ERROR_CHECK in this file and every transfer has a timeout.
 */
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c_master.h"
#include "esp_log.h"
#include "sdkconfig.h"

#include "mag.h"

#if CONFIG_MAG_ENABLE

#define TAG "mag"

#define MAG_PORT          CONFIG_MAG_I2C_PORT
#define MAG_TIMEOUT_MS    50
#define MAG_HUB_ADDR      CONFIG_MAG_HUB_ADDR
#define MAG_MAX_SENSORS   8

#define HMC_ADDR          0x1E
#define QMCP_ADDR        0x2C   /* QMC5883P: another chip sold as "HMC5883L" */
#define QMC_ADDR          0x0D

/* LSB sizes, from the datasheets. The field values are useless without them and they differ by
 * nearly 3x between the two chips, which is exactly how a wrong guess would go unnoticed. */
#define HMC_UT_PER_LSB    (100.0f / 1090.0f)   /* gain 1090 LSB/Gauss */
#define QMC_UT_PER_LSB    (100.0f / 3000.0f)   /* range 8 G, 3000 LSB/Gauss */

static bool s_ready;
static uint8_t s_count;
static mag_reading_t s_readings[MAG_MAX_SENSORS];
static float s_qmcp_scale[MAG_MAX_SENSORS];

/* The scan results travel over HTTP, not only over the console. A board mounted on a wall has no
 * serial port, and these lines are the difference between "no sensor" and "the wrong chip". */
static char s_diag[1200];
static size_t s_diag_len;

static void diag_add(const char *fmt, ...)
{
    if (s_diag_len >= sizeof(s_diag) - 2) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(s_diag + s_diag_len, sizeof(s_diag) - s_diag_len, fmt, ap);
    va_end(ap);
    if (n > 0) {
        s_diag_len += (size_t)n;
        if (s_diag_len > sizeof(s_diag) - 2) {
            s_diag_len = sizeof(s_diag) - 2;
        }
        s_diag[s_diag_len++] = ';';
        s_diag[s_diag_len] = 0;
    }
}

const char *mag_diag(void) { return s_diag; }

static i2c_master_bus_handle_t s_bus;
static i2c_master_dev_handle_t s_hub, s_hmc, s_qmc, s_qmcp;

static esp_err_t add_device(uint8_t addr, i2c_master_dev_handle_t *out)
{
    i2c_device_config_t cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = addr,
        .scl_speed_hz = CONFIG_MAG_CLK_HZ,
    };
    return i2c_master_bus_add_device(s_bus, &cfg, out);
}

static esp_err_t bus_read(uint8_t addr, uint8_t reg, uint8_t *buf, size_t len)
{
    i2c_master_dev_handle_t dev = (addr == HMC_ADDR) ? s_hmc
                                : (addr == QMC_ADDR) ? s_qmc
                                : (addr == QMCP_ADDR) ? s_qmcp : s_hub;
    if (dev == NULL) {
        return ESP_ERR_INVALID_STATE;
    }
    return i2c_master_transmit_receive(dev, &reg, 1, buf, len, MAG_TIMEOUT_MS);
}

static esp_err_t bus_write(uint8_t addr, uint8_t reg, uint8_t value)
{
    i2c_master_dev_handle_t dev = (addr == HMC_ADDR) ? s_hmc
                                : (addr == QMC_ADDR) ? s_qmc
                                : (addr == QMCP_ADDR) ? s_qmcp : s_hub;
    if (dev == NULL) {
        return ESP_ERR_INVALID_STATE;
    }
    uint8_t b[2] = {reg, value};
    return i2c_master_transmit(dev, b, 2, MAG_TIMEOUT_MS);
}

static esp_err_t mux_select(uint8_t channel)
{
    if (s_hub == NULL) {
        return ESP_ERR_INVALID_STATE;
    }
    uint8_t mask = (uint8_t)(1u << channel);
    return i2c_master_transmit(s_hub, &mask, 1, MAG_TIMEOUT_MS);
}

/* Diagnostics that tell the truth.
 *
 * The first version scanned with i2c_master_receive, which reported a device at 0x70 while every
 * write to 0x70 NACKed - the two cannot both be right, and i2c_master_probe is the API that
 * actually checks the address phase. The second half sweeps all eight TCA9548A addresses, because
 * A0/A1/A2 decide that and a board wired differently answers at 0x71..0x77, not 0x70.
 */
static void bus_scan(void)
{
    for (uint8_t addr = 0x08; addr < 0x78; addr++) {
        if (i2c_master_probe(s_bus, addr, 50) == ESP_OK) {
            ESP_LOGI(TAG, "  probe ACK at 0x%02x", addr);
            diag_add("bus:0x%02x", addr);
        }
    }
}

static void mux_sweep(void)
{
    for (uint8_t addr = 0x70; addr <= 0x77; addr++) {
        i2c_device_config_t dcfg = {
            .dev_addr_length = I2C_ADDR_BIT_LEN_7,
            .device_address = addr,
            .scl_speed_hz = CONFIG_MAG_CLK_HZ,
        };
        i2c_master_dev_handle_t dev = NULL;
        if (i2c_master_bus_add_device(s_bus, &dcfg, &dev) != ESP_OK || dev == NULL) {
            continue;
        }
        uint8_t mask = 0x00;
        esp_err_t wr = i2c_master_transmit(dev, &mask, 1, 50);
        esp_err_t pr = i2c_master_probe(s_bus, addr, 50);
        ESP_LOGI(TAG, "  mux addr 0x%02x: probe=%s write=%s", addr,
                 esp_err_to_name(pr), esp_err_to_name(wr));
        if (pr == ESP_OK) {
            diag_add("mux0x%02x:probe=%s,write=%s", addr, esp_err_to_name(pr),
                     esp_err_to_name(wr));
        }
        i2c_master_bus_rm_device(dev);
    }
}

/* Scan the bus behind each channel, and every address on it.
 
   Only probing 0x1E and 0x0D assumes the modules are HMC5883L or QMC5883L. A great many boards
   sold as "HMC5883L" carry something else - QMC5883P at 0x2C, IST8310 at 0x0E, LIS3MDL at 0x1C -
   and a channel that answers at none of the addresses we guess looks exactly like a channel with
   no sensor on it. This scan turns "nothing answered" into either "nothing responded at all"
   (electrical) or "something responded at 0xNN" (the wrong chip, or the right one at an address
   we did not try). */
static void channel_scan(void)
{
    for (uint8_t ch = 0; ch < 8; ch++) {
        if (mux_select(ch) != ESP_OK) {
            ESP_LOGW(TAG, "  ch%u: channel select failed", ch);
            continue;
        }
        int found = 0;
        for (uint8_t addr = 0x08; addr < 0x78; addr++) {
            if (i2c_master_probe(s_bus, addr, 20) == ESP_OK) {
                ESP_LOGI(TAG, "  ch%u: device ACK at 0x%02x", ch, addr);
                diag_add("ch%u:0x%02x", ch, addr);
                found++;
            }
        }
        if (found == 0) {
            ESP_LOGW(TAG, "  ch%u: nothing acked anywhere (0x08-0x77)", ch);
            diag_add("ch%u:empty", ch);
        }
    }
    mux_select(0);
}

/* The register map of the QMC5883P is not the map of either chip this file already knew, and the
 * published variants disagree about which register is control and which is status. Rather than
 * guess and then wonder at a strange magnitude, this dumps what is actually there and reads the
 * data registers twice, 150 ms apart, so "moving, plausible" and "static, zero" can be told apart.
 */
/* QMC5883P: CONTROL1 is 0x0A and CONTROL2 is 0x0B - the opposite of what the first version of this
 * file assumed, which is why the sensor sat in suspend and every data register read zero.
 *
 * Writing 0x03 to 0x0A puts it in continuous measurement (the reset value keeps it suspended), and
 * the data lands in 0x01..0x06 as X/Y/Z little-endian pairs.
 *
 * The full-scale range is a two-bit field whose encoding the datasheets disagree on, and the four
 * possible sensitivities differ by a factor of fifteen (2 G -> 15000 LSB/G, 8 G -> 3750, 12 G ->
 * 2500, 30 G -> 1000). Getting that wrong produces a beautifully stable number that is 15x too
 * small, which is exactly the kind of error that survives review. So the range is inferred from the
 * first reading: the earth's field is 20-80 uT, and only one of the four scales puts the measured
 * magnitude in that window.
 */
static const float QMCP_SCALES[] = {15000.0f, 3750.0f, 2500.0f, 1000.0f};   /* 2, 8, 12, 30 G */

static float qmcp_infer_scale(int32_t counts)
{
    float best = 0.0f;
    float best_err = 1e9f;
    for (size_t i = 0; i < sizeof(QMCP_SCALES) / sizeof(QMCP_SCALES[0]); i++) {
        float ut = (float)counts / QMCP_SCALES[i] * 100.0f;      /* Gauss -> uT */
        if (ut < 20.0f || ut > 80.0f) {
            continue;
        }
        float err = fabsf(ut - 50.0f);                            /* 50 uT is typical */
        if (err < best_err) {
            best_err = err;
            best = QMCP_SCALES[i];
        }
    }
    return best > 0.0f ? best : 3750.0f;      /* 8 G is the middle option if nothing fits */
}

/* The full configuration search, done once at boot.
 *
 * The first attempt swept CONTROL1 (0x0A) alone with a 40 ms dwell and found nothing. That test was
 * flawed: the conversion rate can be as slow as 10 Hz, so 40 ms may be shorter than one sample, and
 * the range and set/reset bits live in CONTROL2 (0x0B) where a wrong value keeps the part from
 * converting at all. This walks a small cube - CONTROL2 x CONTROL1 x the sign/axis register 0x29 -
 * with a 150 ms dwell, and reports the largest field it ever saw, so "everything is zero" and "one
 * combination works and the rest do not" are distinguishable afterwards.
 */
static uint8_t s_qmcp_ctrl1, s_qmcp_ctrl2, s_qmcp_sign;
static int32_t s_qmcp_best_counts;

static int32_t qmcp_read_counts(int16_t *x, int16_t *y, int16_t *z)
{
    uint8_t b[6] = {0};
    if (bus_read(QMCP_ADDR, 0x01, b, 6) != ESP_OK) {
        *x = *y = *z = 0;
        return 0;
    }
    *x = (int16_t)((b[1] << 8) | b[0]);
    *y = (int16_t)((b[3] << 8) | b[2]);
    *z = (int16_t)((b[5] << 8) | b[4]);
    return (int32_t)sqrtf((float)((int32_t)*x * *x + (int32_t)*y * *y + (int32_t)*z * *z));
}

static void qmcp_find_config(uint8_t channel)
{
    static const uint8_t CTRL2[] = {0x00, 0x04, 0x0C, 0x18, 0x20, 0x30};
    static const uint8_t CTRL1[] = {0x01, 0x03, 0x1D, 0x2D, 0x3D};
    static const uint8_t SIGN[] = {0x00, 0x06};

    s_qmcp_ctrl1 = s_qmcp_ctrl2 = s_qmcp_sign = 0;
    s_qmcp_best_counts = 0;
    int16_t x, y, z;

    for (size_t a = 0; a < sizeof(CTRL2); a++) {
        for (size_t b = 0; b < sizeof(CTRL1); b++) {
            for (size_t c = 0; c < sizeof(SIGN); c++) {
                bus_write(QMCP_ADDR, 0x0B, 0x80);            /* soft reset */
                vTaskDelay(pdMS_TO_TICKS(20));
                bus_write(QMCP_ADDR, 0x29, SIGN[c]);         /* axis signs */
                bus_write(QMCP_ADDR, 0x0B, CTRL2[a]);        /* range + set/reset */
                bus_write(QMCP_ADDR, 0x0A, CTRL1[b]);        /* mode + rate */
                vTaskDelay(pdMS_TO_TICKS(150));
                int32_t counts = qmcp_read_counts(&x, &y, &z);
                if (counts > s_qmcp_best_counts) {
                    s_qmcp_best_counts = counts;
                    s_qmcp_ctrl1 = CTRL1[b];
                    s_qmcp_ctrl2 = CTRL2[a];
                    s_qmcp_sign = SIGN[c];
                    diag_add("ch%u c2=0x%02x c1=0x%02x s=0x%02x |B|=%ld xyz=%d,%d,%d",
                             channel, CTRL2[a], CTRL1[b], SIGN[c], (long)counts, x, y, z);
                }
            }
        }
    }
    ESP_LOGI(TAG, "ch%u: QMC5883P best |B|=%ld counts with ctrl2=0x%02x ctrl1=0x%02x sign=0x%02x",
             channel, (long)s_qmcp_best_counts, s_qmcp_ctrl2, s_qmcp_ctrl1, s_qmcp_sign);
    if (s_qmcp_best_counts == 0) {
        diag_add("ch%u all %u combinations read zero",
                 channel, (unsigned)(sizeof(CTRL2) * sizeof(CTRL1) * sizeof(SIGN)));
    }
}

/* The configuration that actually produces data, found by the sweep above on 2026-10-31 and now
 * used directly, so a normal boot does not spend thirteen seconds searching.
 *
 *   write 0x0B = 0x80   soft reset
 *   write 0x29 = 0x00   axis signs, no inversion
 *   write 0x0B = 0x0C   CONTROL2: range + set/reset period
 *   write 0x0A = 0x3D   CONTROL1: measurement mode + rate + oversampling
 *
 * Two mistakes made this take an afternoon. The first sweep only touched CONTROL1, and CONTROL2
 * holds the range and set/reset bits - with a wrong value there the part never converts, so every
 * data register stays zero and looks like a dead chip. The second was the dwell: 40 ms is shorter
 * than one conversion on the slower output rates, so even a correct write looked like a zero.
 *
 * The range is 2 G, 15000 LSB/G, i.e. 0.006667 uT per count. That is not guessed either: earth's
 * field is 25-65 uT and only the 2 G scale puts the measured magnitude in that window.
 */
#define QMCP_CTRL2    0x0C
#define QMCP_CTRL1    0x3D
#define QMCP_SIGN     0x00
#define QMCP_UT_LSB   (100.0f / 15000.0f)

/* Which of the four ranges is in effect? No two sources agreed, and the options differ by 15x, so
 * the external fact decides: the earth's field is 25-65 uT. */
static float qmcp_pick_scale(int32_t counts, const char **name)
{
    static const float CANDS[] = {15000.0f, 3750.0f, 2500.0f, 1000.0f};   /* 2, 8, 12, 30 G */
    static const char *NAMES[] = {"2G", "8G", "12G", "30G"};
    int best = 0;
    float best_err = 1e9f;
    for (int i = 0; i < 4; i++) {
        float err = fabsf((float)counts / CANDS[i] * 100.0f - 45.0f);
        if (err < best_err) {
            best_err = err;
            best = i;
        }
    }
    if (name) {
        *name = NAMES[best];
    }
    return 100.0f / CANDS[best];
}

static bool qmcp_apply_config(uint8_t channel)
{
    bus_write(QMCP_ADDR, 0x0B, 0x80);
    vTaskDelay(pdMS_TO_TICKS(20));
    bus_write(QMCP_ADDR, 0x29, QMCP_SIGN);
    bus_write(QMCP_ADDR, 0x0B, QMCP_CTRL2);
    bus_write(QMCP_ADDR, 0x0A, QMCP_CTRL1);
    vTaskDelay(pdMS_TO_TICKS(220));

    int16_t x, y, z;
    int32_t counts = qmcp_read_counts(&x, &y, &z);
    if (counts == 0) {
        diag_add("ch%u fixed config produced zero, sweeping", channel);
        return false;
    }
    const char *range_name = "?";
    float scale = qmcp_pick_scale(counts, &range_name);
    diag_add("ch%u range=%s (%d counts -> %.1f uT)", channel, range_name, (int)counts,
             (double)((float)counts * scale));

    /* Ten samples a second apart: a magnitude that wanders outside 25-65 uT means the scale or the
     * range is wrong even though the numbers look plausible. */
    float lo = 1e9f, hi = -1e9f, sum = 0.0f;
    int n = 0;
    for (int i = 0; i < 10; i++) {
        int32_t c = qmcp_read_counts(&x, &y, &z);
        float ut = (float)c * scale;
        if (c > 0) {
            lo = ut < lo ? ut : lo;
            hi = ut > hi ? ut : hi;
            sum += ut;
            n++;
        }
        vTaskDelay(pdMS_TO_TICKS(100));
    }
    diag_add("ch%u |B| %.1f..%.1f uT avg %.1f over %d (2G scale)", channel, (double)lo, (double)hi,
             (double)(n ? sum / n : 0.0f), n);
    ESP_LOGI(TAG, "ch%u: QMC5883P |B| %.1f..%.1f uT (avg %.1f) - earth field is 25-65 uT",
             channel, (double)lo, (double)hi, (double)(n ? sum / n : 0.0f));
    return true;
}

static void configure_qmcp(uint8_t channel, float *ut_per_lsb)
{
    if (!qmcp_apply_config(channel)) {
        qmcp_find_config(channel);          /* only when the known good values do not work here */
        if (s_qmcp_ctrl1 != 0) {
            bus_write(QMCP_ADDR, 0x29, s_qmcp_sign);
            bus_write(QMCP_ADDR, 0x0B, s_qmcp_ctrl2);
            bus_write(QMCP_ADDR, 0x0A, s_qmcp_ctrl1);
            vTaskDelay(pdMS_TO_TICKS(150));
        }
    }

    uint8_t b[6] = {0};
    if (bus_read(QMCP_ADDR, 0x01, b, 6) != ESP_OK) {
        diag_add("ch%u qmcp read fail", channel);
        return;
    }
    int16_t x = (int16_t)((b[1] << 8) | b[0]);
    int16_t y = (int16_t)((b[3] << 8) | b[2]);
    int16_t z = (int16_t)((b[5] << 8) | b[4]);
    int32_t counts = (int32_t)sqrtf((float)(x * x + y * y + z * z));
    *ut_per_lsb = qmcp_pick_scale(counts, NULL);
    diag_add("ch%u qmcp xyz=%d,%d,%d |B|=%ld counts scale=%.4fuT/lsb", channel, x, y, z,
             (long)counts, (double)*ut_per_lsb);
    ESP_LOGI(TAG, "ch%u: QMC5883P first sample %d,%d,%d -> |B|=%ld counts, %.2f uT (scale %.4f)",
             channel, x, y, z, (long)counts, (double)(counts * *ut_per_lsb), (double)*ut_per_lsb);
}

static void qmcp_probe_report(uint8_t channel)
{
    uint8_t regs[16];
    for (uint8_t base = 0x00; base < 0x10; base += 8) {
        if (bus_read(QMCP_ADDR, base, regs, 8) != ESP_OK) {
            diag_add("ch%u dump%02x=fail", channel, base);
            continue;
        }
        char hex[3 * 8 + 1];
        int n = 0;
        for (int i = 0; i < 8; i++) {
            n += snprintf(hex + n, sizeof(hex) - n, "%02x", regs[i]);
        }
        diag_add("ch%u r%02x=%s", channel, base, hex);
    }
    /* CONTROL2 on the QMC5883P is documented at 0x29 in every variant that agrees on anything:
     * enabling set/reset there is what turns continuous measurement on. */
    bus_write(QMCP_ADDR, 0x29, 0x06);
    bus_write(QMCP_ADDR, 0x0B, 0x40);      /* continuous, mid range - corrected by measurement */
    vTaskDelay(pdMS_TO_TICKS(60));

    uint8_t a[6] = {0}, b[6] = {0};
    bus_read(QMCP_ADDR, 0x01, a, 6);
    vTaskDelay(pdMS_TO_TICKS(150));
    bus_read(QMCP_ADDR, 0x01, b, 6);
    int16_t ax = (int16_t)((a[1] << 8) | a[0]), ay = (int16_t)((a[3] << 8) | a[2]),
            az = (int16_t)((a[5] << 8) | a[4]);
    int16_t bx = (int16_t)((b[1] << 8) | b[0]), by = (int16_t)((b[3] << 8) | b[2]),
            bz = (int16_t)((b[5] << 8) | b[4]);
    diag_add("ch%u a=%d,%d,%d b=%d,%d,%d", channel, ax, ay, az, bx, by, bz);
}

static mag_chip_t probe(uint8_t channel)
{
    if (mux_select(channel) != ESP_OK) {
        return MAG_CHIP_NONE;
    }
    /* ACK first, registers second: "nothing on the channel" and "something on the channel that
     * does not answer register reads" are different faults with different fixes. */
    bool ack_hmc = i2c_master_probe(s_bus, HMC_ADDR, 50) == ESP_OK;
    bool ack_qmc = i2c_master_probe(s_bus, QMC_ADDR, 50) == ESP_OK;
    bool ack_qmcp = i2c_master_probe(s_bus, QMCP_ADDR, 50) == ESP_OK;
    if (ack_hmc || ack_qmc || ack_qmcp) {
        ESP_LOGI(TAG, "  ch%u: ACK 0x1E=%d 0x0D=%d 0x2C=%d", channel,
                 (int)ack_hmc, (int)ack_qmc, (int)ack_qmcp);
        diag_add("probe ch%u 1E=%d 0D=%d 2C=%d", channel, (int)ack_hmc, (int)ack_qmc,
                 (int)ack_qmcp);
    }
    /* The 0x2C device is a QMC5883P in most of these modules. Its chip id lives in register 0x00
     * and reads 0x80; logging it settles the identity instead of guessing from the address. */
    if (ack_qmcp) {
        uint8_t reg = 0x00, id = 0;
        i2c_device_config_t dcfg = {
            .dev_addr_length = I2C_ADDR_BIT_LEN_7,
            .device_address = QMCP_ADDR,
            .scl_speed_hz = CONFIG_MAG_CLK_HZ,
        };
        i2c_master_dev_handle_t dev = NULL;
        if (i2c_master_bus_add_device(s_bus, &dcfg, &dev) == ESP_OK && dev != NULL) {
            if (i2c_master_transmit_receive(dev, &reg, 1, &id, 1, 50) == ESP_OK) {
                ESP_LOGI(TAG, "  ch%u: 0x2C chip id 0x%02x", channel, id);
                diag_add("ch%u id2C=0x%02x", channel, id);
            }
            i2c_master_bus_rm_device(dev);
            qmcp_probe_report(channel);
        }
    }
    if (ack_qmcp) {
        uint8_t reg = 0x00, chip = 0;
        if (bus_read(QMCP_ADDR, reg, &chip, 1) == ESP_OK && chip == 0x80) {
            return MAG_CHIP_QMC5883P;
        }
    }
    uint8_t id[3] = {0};
    if (ack_hmc && bus_read(HMC_ADDR, 0x0A, id, 3) == ESP_OK && id[0] == 'H') {
        return MAG_CHIP_HMC5883L;         /* "H43" in 0x0A..0x0C */
    }
    uint8_t chip = 0;
    if (ack_qmc && bus_read(QMC_ADDR, 0x0D, &chip, 1) == ESP_OK && chip == 0xFF) {
        return MAG_CHIP_QMC5883L;
    }
    return MAG_CHIP_NONE;
}

static void configure(mag_chip_t chip)
{
    if (chip == MAG_CHIP_HMC5883L) {
        bus_write(HMC_ADDR, 0x00, 0x70);  /* 8 sample average, 15 Hz, normal measurement */
        bus_write(HMC_ADDR, 0x01, 0x20);  /* gain 1090 LSB/Gauss */
        bus_write(HMC_ADDR, 0x02, 0x00);  /* continuous mode */
    } else if (chip == MAG_CHIP_QMC5883L) {
        bus_write(QMC_ADDR, 0x0B, 0x01);  /* set/reset period */
        bus_write(QMC_ADDR, 0x09, 0x1D);  /* 512x oversample, 8 G, 200 Hz, continuous */
        bus_write(QMC_ADDR, 0x0A, 0x00);  /* no interrupts, no pointer roll */
    }
}

static bool read_sample(mag_reading_t *r)
{
    uint8_t b[6] = {0};
    if (r->chip == MAG_CHIP_HMC5883L) {
        if (bus_read(HMC_ADDR, 0x03, b, 6) != ESP_OK) {
            return false;
        }
        /* The HMC5883L data order is X, Z, Y - not X, Y, Z. Getting this wrong swaps two axes and
         * still produces plausible numbers. */
        r->x = (int16_t)((b[0] << 8) | b[1]);
        r->z = (int16_t)((b[2] << 8) | b[3]);
        r->y = (int16_t)((b[4] << 8) | b[5]);
        r->field_ut = (float)(int32_t)r->x * r->x + (int32_t)r->y * r->y + (int32_t)r->z * r->z;
        r->field_ut = (r->field_ut > 0.0f) ? HMC_UT_PER_LSB * sqrtf(r->field_ut) : 0.0f;
        return true;
    }
    if (r->chip == MAG_CHIP_QMC5883P) {
        if (bus_read(QMCP_ADDR, 0x01, b, 6) != ESP_OK) {
            return false;
        }
        r->x = (int16_t)((b[1] << 8) | b[0]);
        r->y = (int16_t)((b[3] << 8) | b[2]);
        r->z = (int16_t)((b[5] << 8) | b[4]);
        float counts = sqrtf((float)((int32_t)r->x * r->x + (int32_t)r->y * r->y
                                     + (int32_t)r->z * r->z));
        r->ut_per_lsb = r->ut_per_lsb > 0.0f ? r->ut_per_lsb : 100.0f / 3750.0f;
        r->field_ut = counts * r->ut_per_lsb;
        return true;
    }
    if (r->chip == MAG_CHIP_QMC5883L) {
        if (bus_read(QMC_ADDR, 0x00, b, 6) != ESP_OK) {
            return false;
        }
        r->x = (int16_t)((b[1] << 8) | b[0]);
        r->y = (int16_t)((b[3] << 8) | b[2]);
        r->z = (int16_t)((b[5] << 8) | b[4]);
        r->field_ut = (float)(int32_t)r->x * r->x + (int32_t)r->y * r->y + (int32_t)r->z * r->z;
        r->field_ut = (r->field_ut > 0.0f) ? QMC_UT_PER_LSB * sqrtf(r->field_ut) : 0.0f;
        return true;
    }
    return false;
}

static void mag_task(void *arg)
{
    (void)arg;
    const uint32_t period_ms = 1000u / (CONFIG_MAG_SAMPLE_HZ > 0 ? CONFIG_MAG_SAMPLE_HZ : 1);
    uint32_t tick = 0;
    while (true) {
        for (uint8_t i = 0; i < s_count; i++) {
            mag_reading_t *r = &s_readings[i];
            if (mux_select(r->channel) != ESP_OK) {
                r->ok = false;
                r->errors++;
                continue;
            }
            r->ut_per_lsb = s_qmcp_scale[i];
            if (read_sample(r)) {
                r->ok = true;
                r->reads++;
            } else {
                r->ok = false;
                r->errors++;
            }
        }
        /* One block every ten seconds. Every second was four lines of numbers nobody was reading,
         * and it buried the RFID lines that were the reason the console was open. */
        if (tick % (10 * (CONFIG_MAG_SAMPLE_HZ > 0 ? CONFIG_MAG_SAMPLE_HZ : 1)) == 0) {
            for (uint8_t i = 0; i < s_count; i++) {
                const mag_reading_t *r = &s_readings[i];
                ESP_LOGI(TAG, "ch%u %s %s x=%d y=%d z=%d B=%.1fuT reads=%lu err=%lu",
                         r->channel,
                         r->chip == MAG_CHIP_HMC5883L ? "HMC5883L" : "QMC5883L",
                         r->ok ? "ok  " : "FAIL",
                         r->x, r->y, r->z, (double)r->field_ut,
                         (unsigned long)r->reads, (unsigned long)r->errors);
            }
        }
        tick++;
        vTaskDelay(pdMS_TO_TICKS(period_ms));
    }
}

esp_err_t mag_start(void)
{
    /* The new driver, not the legacy one. The legacy driver's conflict check calls abort() when
     * it believes the port belongs to someone else, which turns a wiring mistake into a boot loop -
     * the board became unreachable after the first flash of this file. The new driver returns an
     * error instead, so a failed install is a log line and nothing more. */
    i2c_master_bus_config_t cfg = {
        .i2c_port = MAG_PORT,
        .sda_io_num = CONFIG_MAG_SDA_GPIO,
        .scl_io_num = CONFIG_MAG_SCL_GPIO,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    esp_err_t err = i2c_new_master_bus(&cfg, &s_bus);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "i2c bus on port %d failed: %s (camera SCCB may own it)",
                 MAG_PORT, esp_err_to_name(err));
        return err;
    }
    if (add_device(MAG_HUB_ADDR, &s_hub) != ESP_OK) {
        ESP_LOGW(TAG, "could not add mux 0x%02x", MAG_HUB_ADDR);
        return ESP_ERR_NOT_FOUND;
    }
    add_device(HMC_ADDR, &s_hmc);
    add_device(QMC_ADDR, &s_qmc);
    add_device(QMCP_ADDR, &s_qmcp);

    /* No mux-presence gate. An earlier version asked the TCA9548A for a register (it has none),
     * failed, and returned before probing a single sensor - so a working setup reported "no mux".
     * The honest test is the one that matters: select each channel and ask the sensor. */
    ESP_LOGI(TAG, "diagnostics on I2C%d (sda=%d scl=%d, %d Hz):",
             MAG_PORT, CONFIG_MAG_SDA_GPIO, CONFIG_MAG_SCL_GPIO, CONFIG_MAG_CLK_HZ);
    bus_scan();
    mux_sweep();
    channel_scan();
    if (mux_select(0) != ESP_OK) {
        ESP_LOGW(TAG, "mux 0x%02x did not ACK a channel write; probing channels anyway", MAG_HUB_ADDR);
    }

    for (uint8_t ch = 0; ch < CONFIG_MAG_CHANNELS && s_count < MAG_MAX_SENSORS; ch++) {
        mag_chip_t chip = probe(ch);
        if (chip == MAG_CHIP_NONE) {
            ESP_LOGW(TAG, "ch%u: nothing answered (0x1E or 0x0D) - check that channel's wiring", ch);
            continue;
        }
        s_qmcp_scale[s_count] = 0.0f;
        if (chip == MAG_CHIP_QMC5883P) {
            configure_qmcp(ch, &s_qmcp_scale[s_count]);
        } else {
            configure(chip);
            s_qmcp_scale[s_count] = 0.0f;
        }
        s_readings[s_count].channel = ch;
        s_readings[s_count].chip = chip;
        s_readings[s_count].ut_per_lsb = s_qmcp_scale[s_count];
        s_readings[s_count].ok = false;
        s_count++;
        ESP_LOGI(TAG, "ch%u: %s found", ch,
                 chip == MAG_CHIP_HMC5883L ? "HMC5883L (0x1E)"
                 : chip == MAG_CHIP_QMC5883P ? "QMC5883P (0x2C)"
                 : "QMC5883L (0x0D)");
    }

    ESP_LOGI(TAG, "hub 0x%02x ok, %u/%u sensors, %d Hz on I2C%d (sda=%d scl=%d)",
             MAG_HUB_ADDR, s_count, (unsigned)CONFIG_MAG_CHANNELS, CONFIG_MAG_SAMPLE_HZ,
             MAG_PORT, CONFIG_MAG_SDA_GPIO, CONFIG_MAG_SCL_GPIO);
    if (s_count == 0) {
        return ESP_ERR_NOT_FOUND;
    }
    s_ready = true;
    xTaskCreate(mag_task, "mag", 3072, NULL, 3, NULL);
    return ESP_OK;
}

bool mag_available(void) { return s_ready; }
uint8_t mag_count(void)   { return s_count; }

bool mag_get(uint8_t index, mag_reading_t *out)
{
    if (index >= s_count || out == NULL) {
        return false;
    }
    *out = s_readings[index];
    return true;
}

#else  /* !CONFIG_MAG_ENABLE */

esp_err_t mag_start(void) { return ESP_ERR_NOT_SUPPORTED; }
bool mag_available(void)  { return false; }
uint8_t mag_count(void)   { return 0; }
bool mag_get(uint8_t i, mag_reading_t *out) { (void)i; (void)out; return false; }

#endif
