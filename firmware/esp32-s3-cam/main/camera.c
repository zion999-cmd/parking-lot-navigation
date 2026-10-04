#include "camera.h"

#include <stdio.h>

#include <driver/gpio.h>
#include <esp_camera.h>
#include <esp_log.h>

static const char *TAG = "camera";

/*
 * ---------------------------------------------------------------------------
 * Camera pin map - ESP32-S3-CAM core board on the KZB V2.0 baseboard.
 *
 * Provenance of every line below. Do not "fix" these by guessing.
 *
 * 1. They are a copy of the values that are known to work on this exact board:
 *      /Users/bx/Esp32/xiaozhi-esp32-224/main/boards/bread-compact-wifi-s3cam/config.h
 *
 * 2. They were cross-checked against the vendor schematic
 *    ESP32S3-CAM-KZB-V2_0.pdf. In the PDF the camera signal label and the GPIO
 *    label sit on the same wire (identical y coordinate, adjacent x), which
 *    makes all 14 signals pair unambiguously. Result: 14/14 match.
 *
 * 3. They are confirmed at runtime by the serial log captured from this board:
 *      camera: Camera PID=0x3660 VER=0x00 MIDL=0x00 MIDH=0x00
 *      camera: Detected OV3660 camera
 *      camera: Detected camera at address=0x3c
 *      ov3660: Calculated VCO: 80000000 Hz, PLLCLK: 40000000 Hz, SYSCLK: 10 MHz
 *
 * WARNING - one reference file contradicts both of the above:
 *    /Users/bx/Esp32/xiaozhi-esp32-main223/docs/esp32s3-cam-kzb-v2_0-pins.md
 * It contains no VSYNC, no XCLK, no SIOD and no OV_D5 at all, and lists
 * GPIO16/17/18 as TFT signals. That document is AI-generated (it ends with
 * "如需详细功能…请补充需求"), not a schematic transcription, and it is wrong.
 * See README section "Hardware evidence chain".
 * ---------------------------------------------------------------------------
 */
#define CAM_PIN_D0    GPIO_NUM_11 /* schematic OV_D0 */
#define CAM_PIN_D1    GPIO_NUM_9  /* schematic OV_D1 */
#define CAM_PIN_D2    GPIO_NUM_8  /* schematic OV_D2 */
#define CAM_PIN_D3    GPIO_NUM_10 /* schematic OV_D3 */
#define CAM_PIN_D4    GPIO_NUM_12 /* schematic OV_D4 */
#define CAM_PIN_D5    GPIO_NUM_18 /* schematic OV_D5 */
#define CAM_PIN_D6    GPIO_NUM_17 /* schematic OV_D6 */
#define CAM_PIN_D7    GPIO_NUM_16 /* schematic OV_D7 */
#define CAM_PIN_XCLK  GPIO_NUM_15 /* schematic XCLK  */
#define CAM_PIN_PCLK  GPIO_NUM_13 /* schematic PCLK  */
#define CAM_PIN_VSYNC GPIO_NUM_6  /* schematic VSYNC */
#define CAM_PIN_HREF  GPIO_NUM_7  /* schematic HREF  */
#define CAM_PIN_SIOC  GPIO_NUM_5  /* schematic SIOC  */
#define CAM_PIN_SIOD  GPIO_NUM_4  /* schematic SIOD  */

/* The schematic leaves PWDN and RESET unconnected; the sensor is always powered. */
#define CAM_PIN_PWDN  (-1)
#define CAM_PIN_RESET (-1)

/* XCLK used by the proven configuration on this board. */
#define CAM_XCLK_FREQ_HZ 20000000

/*
 * Frame geometry. P0001 prioritises stability > latency > framerate >
 * resolution, so this starts at SVGA (800x600), which is inside the range
 * suggested for real-time vision work and leaves the sensor's native 4:3
 * aspect ratio untouched.
 */
#define CAM_FRAME_SIZE FRAMESIZE_SVGA

/*
 * PIXFORMAT_JPEG: let the sensor emit JPEG directly, so no software encoder and
 * no RGB565 byte swapping are needed on the ESP32-S3. The reference project
 * used PIXFORMAT_RGB565 because it had to feed LVGL preview and encode to JPEG
 * in software for a one-shot photo; a continuous MJPEG stream wants the
 * opposite trade-off. See README "Camera configuration".
 */
#define CAM_PIXEL_FORMAT PIXFORMAT_JPEG

/* Same JPEG quality the reference project used. */
#define CAM_JPEG_QUALITY 12

/*
 * 2 buffers + CAMERA_GRAB_LATEST keeps latency low while streaming; the
 * reference used 1 buffer because it only captured single photos.
 */
#define CAM_FB_COUNT 2

/*
 * Orientation, copied from the proven configuration: the reference set
 * hmirror = 0 and vflip = 1 for this sensor, so the image comes out upright.
 * If the camera ends up mounted upside down on the sandbox, flip this one line.
 */
#define CAM_HMIRROR 0
#define CAM_VFLIP 1

static bool s_ready = false;
static char s_sensor_name[16] = "unknown";
static const char *s_resolution = "unknown";
static const char *s_format = "unknown";

static const char *sensor_name_from_pid(uint16_t pid)
{
    switch (pid) {
    case OV2640_PID: return "OV2640";
    case OV3660_PID: return "OV3660";
    case OV5640_PID: return "OV5640";
    case GC0308_PID: return "GC0308";
    default: return "unknown";
    }
}

static const char *resolution_name(framesize_t fs)
{
    switch (fs) {
    case FRAMESIZE_QVGA: return "320x240 (QVGA)";
    case FRAMESIZE_VGA:  return "640x480 (VGA)";
    case FRAMESIZE_SVGA: return "800x600 (SVGA)";
    case FRAMESIZE_XGA:  return "1024x768 (XGA)";
    case FRAMESIZE_HD:   return "1280x720 (HD)";
    case FRAMESIZE_SXGA: return "1280x1024 (SXGA)";
    case FRAMESIZE_UXGA: return "1600x1200 (UXGA)";
    default:             return "unknown";
    }
}

static const char *pixel_format_name(pixformat_t fmt)
{
    switch (fmt) {
    case PIXFORMAT_JPEG:      return "JPEG";
    case PIXFORMAT_RGB565:    return "RGB565";
    case PIXFORMAT_YUV422:    return "YUV422";
    case PIXFORMAT_GRAYSCALE: return "GRAYSCALE";
    default:                  return "unknown";
    }
}

esp_err_t camera_init(void)
{
    camera_config_t config = {0};

    config.pin_pwdn = CAM_PIN_PWDN;
    config.pin_reset = CAM_PIN_RESET;
    config.pin_xclk = CAM_PIN_XCLK;
    config.pin_sccb_sda = CAM_PIN_SIOD;
    config.pin_sccb_scl = CAM_PIN_SIOC;
    config.pin_d0 = CAM_PIN_D0;
    config.pin_d1 = CAM_PIN_D1;
    config.pin_d2 = CAM_PIN_D2;
    config.pin_d3 = CAM_PIN_D3;
    config.pin_d4 = CAM_PIN_D4;
    config.pin_d5 = CAM_PIN_D5;
    config.pin_d6 = CAM_PIN_D6;
    config.pin_d7 = CAM_PIN_D7;
    config.pin_vsync = CAM_PIN_VSYNC;
    config.pin_href = CAM_PIN_HREF;
    config.pin_pclk = CAM_PIN_PCLK;

    config.xclk_freq_hz = CAM_XCLK_FREQ_HZ;
    config.ledc_timer = LEDC_TIMER_0;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.sccb_i2c_port = 0;

    config.pixel_format = CAM_PIXEL_FORMAT;
    config.frame_size = CAM_FRAME_SIZE;
    config.jpeg_quality = CAM_JPEG_QUALITY;
    config.fb_count = CAM_FB_COUNT;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_LATEST;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_camera_init failed: %s (0x%x)", esp_err_to_name(err), err);
        return err;
    }

    sensor_t *sensor = esp_camera_sensor_get();
    if (sensor == NULL) {
        ESP_LOGE(TAG, "esp_camera_sensor_get() returned NULL");
        esp_camera_deinit();
        return ESP_FAIL;
    }

    snprintf(s_sensor_name, sizeof(s_sensor_name), "%s",
             sensor_name_from_pid(sensor->id.PID));
    s_resolution = resolution_name(config.frame_size);
    s_format = pixel_format_name(config.pixel_format);

    sensor->set_hmirror(sensor, CAM_HMIRROR);
    sensor->set_vflip(sensor, CAM_VFLIP);

    s_ready = true;

    ESP_LOGI(TAG, "Camera sensor: %s (PID=0x%04x, SCCB addr=0x%02x)",
             s_sensor_name, sensor->id.PID, sensor->slv_addr);
    ESP_LOGI(TAG, "Camera initialized");
    ESP_LOGI(TAG, "Camera configuration: %s, %s, quality=%d, fb_count=%d, fb_location=PSRAM, xclk=%d Hz",
             s_resolution, s_format, CAM_JPEG_QUALITY, CAM_FB_COUNT, CAM_XCLK_FREQ_HZ);

    return ESP_OK;
}

bool camera_ready(void)
{
    return s_ready;
}

const char *camera_sensor_name(void)
{
    return s_sensor_name;
}

const char *camera_resolution_name(void)
{
    return s_resolution;
}

const char *camera_pixel_format_name(void)
{
    return s_format;
}

int camera_jpeg_quality(void)
{
    return CAM_JPEG_QUALITY;
}
