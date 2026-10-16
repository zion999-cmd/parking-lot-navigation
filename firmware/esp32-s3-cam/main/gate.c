#include "gate.h"

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include <driver/gpio.h>
#include <driver/ledc.h>
#include <esp_log.h>
#include <esp_timer.h>
#include <sdkconfig.h>

#include "display.h"

/*
 * Barrier gate - implementation. Read gate.h first for the resource audit.
 *
 * The gate is a small state machine driven by one debounced input:
 *
 *   CLOSED --press--> OPENING --ramp done--> OPEN --released + HOLD_MS--> CLOSING --ramp done--> CLOSED
 *
 * Two rules that came out of the requirements and are worth stating, because
 * both are safety properties rather than conveniences:
 *
 *   1. One press, one lift. A trigger is only accepted in CLOSED, and only when
 *      the input has been seen released since the last one. Holding the switch
 *      down cannot make the arm hunt.
 *   2. While the switch is still pressed the arm stays up, and the hold timer is
 *      restarted. That is the anti-pinch: "the car is still on the sensor" and
 *      "it is safe to lower the arm" are the same condition, so they are the same
 *      variable and cannot drift apart.
 *
 * Everything is non-blocking on a 20 ms tick, so the arm cannot stall the camera
 * or the strip, and the strip cannot stall the arm.
 */

static const char *TAG = "gate";

/* State machine tick. Also the debounce sampling period. */
#define GATE_TICK_MS 20

#define SERVO_PERIOD_US 20000                       /* 50 Hz */
#define SERVO_MAX_DUTY  ((1 << 14) - 1)             /* 14 bit timer, 16383 */

typedef enum {
    GATE_OFF = 0,
    GATE_CLOSED,
    GATE_OPENING,
    GATE_OPEN,
    GATE_CLOSING,
} gate_state_t;

static const char *const STATE_NAME[] = { "off", "closed", "opening", "open", "closing" };

static volatile gate_state_t s_state = GATE_OFF;

/* Debounce: the candidate level has to hold for GATE_DEBOUNCE_MS to become the
 * debounced level. A mechanical contact bounces for a few milliseconds; the
 * requirement is 100 ms, which also absorbs a hand brushing the switch. */
static bool s_debounced;        /* is the switch pressed, debounced */
static bool s_candidate;
static int s_candidate_ms;

/*
 * Re-arm. Starts false on purpose: a switch that is already closed at power-up
 * (shorted, jammed, or a car parked on it) must not swing the arm during boot.
 * Release it once and the gate is armed.
 */
static bool s_armed;

/* Servo position, in microseconds of pulse width. */
static int s_pulse_us;
static bool s_ramping;
static int s_ramp_from_us;
static int s_ramp_to_us;
static int64_t s_ramp_started_us;

/* Hold timer: -1 means "not counting", which is the case while the switch is
 * still pressed. */
static int64_t s_hold_since_us = -1;

/*
 * When to take the QR code down, or 0 while the gate is open and it is not
 * counting. Separate from the arm's own timing on purpose: how long the arm
 * stays up is a mechanical question (the car has to clear the gate), while how
 * long the code stays up is a human one (pick up the phone, open the camera, aim
 * it, focus - 5 to 15 seconds with a car in the other hand). Tying them together
 * means one of the two is always wrong.
 */
#if CONFIG_GATE_QR_ON_TRIGGER
static int64_t s_qr_off_at_us;
#endif

/* ------------------------------------------------------------------------- */
/* Servo                                                                      */
/* ------------------------------------------------------------------------- */

static uint32_t pulse_to_duty(int us)
{
    if (us < 100) {
        us = 100;
    }
    if (us > 3000) {
        us = 3000;
    }
    return (uint32_t)(((int64_t)us * SERVO_MAX_DUTY) / SERVO_PERIOD_US);
}

static void servo_write(int us)
{
    s_pulse_us = us;
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_2, pulse_to_duty(us));
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_2);
}

/*
 * Movement is spread over CONFIG_GATE_RAMP_MS instead of being applied in one
 * step. A servo commanded straight to a new position pulls its stall current for
 * the whole of the first movement, and this servo shares its 5 V rail with the
 * strip and (through the board's regulator) with the camera. Slower costs
 * nothing here and removes the current spike.
 */
static void servo_ramp_to(int target_us)
{
    s_ramp_from_us = s_pulse_us;
    s_ramp_to_us = target_us;
    s_ramp_started_us = esp_timer_get_time();
    s_ramping = true;
}

static void servo_ramp_step(void)
{
    if (!s_ramping) {
        return;
    }

    int64_t elapsed_ms = (esp_timer_get_time() - s_ramp_started_us) / 1000;
    if (elapsed_ms >= CONFIG_GATE_RAMP_MS) {
        servo_write(s_ramp_to_us);
        s_ramping = false;
        return;
    }

    int64_t span = s_ramp_to_us - s_ramp_from_us;
    servo_write(s_ramp_from_us + (int)((span * elapsed_ms) / CONFIG_GATE_RAMP_MS));
}

/* ------------------------------------------------------------------------- */
/* State machine                                                              */
/* ------------------------------------------------------------------------- */

static void enter(gate_state_t next)
{
    gate_state_t prev = s_state;
    if (prev == next) {
        return;
    }
    s_state = next;

    /*
     * Every transition is logged with the sensor level and the pulse width, so a
     * gate that behaves oddly can be read straight off the console without
     * guessing which transition happened when.
     */
    ESP_LOGI(TAG, "%s -> %s (sensor=%s pulse=%dus)", STATE_NAME[prev], STATE_NAME[next],
             s_debounced ? "pressed" : "released", s_pulse_us);
}

static void gate_step(void)
{
    int64_t now_us = esp_timer_get_time();

    /* --- debounce ------------------------------------------------------- */
    if (s_candidate != s_debounced) {
        s_candidate_ms += GATE_TICK_MS;
        if (s_candidate_ms >= CONFIG_GATE_DEBOUNCE_MS) {
            s_debounced = s_candidate;
            s_candidate_ms = 0;
        }
    } else {
        s_candidate_ms = 0;
    }

    /* --- transitions ---------------------------------------------------- */
    switch (s_state) {
    case GATE_CLOSED:
        if (s_debounced && s_armed) {
            s_armed = false;
            enter(GATE_OPENING);
            servo_ramp_to(CONFIG_GATE_PULSE_OPEN_US);
#if CONFIG_GATE_QR_ON_TRIGGER
            /*
             * A car has arrived, so this is the moment to put the code in front of
             * the person: they are standing there, holding the car, and the phone
             * is the next thing they will reach for. Showing it at boot instead (or
             * as well) is only for checking that it scans.
             *
             * A transient overlay, so display_restore() below puts the status
             * screen back without this file having to know what it was.
             */
            display_show_qr(NULL);
            s_qr_off_at_us = 0;      /* stays up while the car is on the sensor */
#endif
        }
        break;

    case GATE_OPENING:
        if (!s_ramping) {
            enter(GATE_OPEN);
            s_hold_since_us = -1;      /* the countdown starts at release */
        }
        break;

    case GATE_OPEN:
        if (s_debounced) {
            s_hold_since_us = -1;      /* still there: keep the arm up */
        } else if (s_hold_since_us < 0) {
            s_hold_since_us = now_us;
        } else if ((now_us - s_hold_since_us) / 1000 >= CONFIG_GATE_HOLD_MS) {
            enter(GATE_CLOSING);
            servo_ramp_to(CONFIG_GATE_PULSE_CLOSED_US);
        }
        break;

    case GATE_CLOSING:
        if (!s_ramping) {
            enter(GATE_CLOSED);
#if CONFIG_GATE_QR_ON_TRIGGER
            /* The arm is home, so start the code's own countdown. It is taken
             * down here rather than when the arm starts moving, which alone is
             * worth a ramp's worth of seconds. */
            s_qr_off_at_us = esp_timer_get_time() + (int64_t)CONFIG_GATE_QR_LINGER_MS * 1000;
#endif
        }
        break;

    case GATE_OFF:
    default:
        break;
    }

    /* --- QR countdown --------------------------------------------------- */
#if CONFIG_GATE_QR_ON_TRIGGER
    if (s_qr_off_at_us != 0 && now_us >= s_qr_off_at_us) {
        s_qr_off_at_us = 0;
        display_restore();
    }
#endif

    /* --- re-arm --------------------------------------------------------- */
    /* Only when the arm is home and the switch has actually been let go. */
    if (s_state == GATE_CLOSED && !s_debounced) {
        s_armed = true;
    }

    servo_ramp_step();
}

/* ------------------------------------------------------------------------- */
/* Calibration                                                                */
/* ------------------------------------------------------------------------- */

#if CONFIG_GATE_CALIBRATE

/*
 * A wide servo is not a promise: 185 degrees on the label usually means a
 * modified 180 degree unit whose ends are unreachable, and the pulse range that
 * actually maps to the arm's travel depends on how the horn is fitted. So the
 * two positions are measured, not assumed.
 *
 * This sweeps slowly through a deliberately narrow window (800..2200 us, well
 * inside the usual 500..2500) and prints the width it is holding, so the arm can
 * be watched while the numbers go by. Then it parks at the closed position and
 * the gate runs normally. Set the two widths in menuconfig afterwards and turn
 * this off.
 */
static void gate_calibrate(void)
{
    ESP_LOGW(TAG, "CALIBRATION: sweeping %d..%d us in %d us steps, %d ms each - "
                  "note the two widths you want for the arm",
             CONFIG_GATE_CALIBRATE_FROM_US, CONFIG_GATE_CALIBRATE_TO_US,
             CONFIG_GATE_CALIBRATE_STEP_US, CONFIG_GATE_CALIBRATE_DWELL_MS);

    for (int us = CONFIG_GATE_CALIBRATE_FROM_US; us <= CONFIG_GATE_CALIBRATE_TO_US;
         us += CONFIG_GATE_CALIBRATE_STEP_US) {
        servo_write(us);
        ESP_LOGI(TAG, "  pulse %4d us (duty %u)", us, (unsigned)pulse_to_duty(us));
        vTaskDelay(pdMS_TO_TICKS(CONFIG_GATE_CALIBRATE_DWELL_MS));
    }

    ESP_LOGW(TAG, "CALIBRATION done - set GATE_PULSE_CLOSED_US / GATE_PULSE_OPEN_US, "
                  "then turn GATE_CALIBRATE off");
    servo_write(CONFIG_GATE_PULSE_CLOSED_US);
    vTaskDelay(pdMS_TO_TICKS(300));
}

#endif /* CONFIG_GATE_CALIBRATE */

/* ------------------------------------------------------------------------- */
/* Task                                                                       */
/* ------------------------------------------------------------------------- */

static void gate_task(void *arg)
{
    (void)arg;

    /*
     * Park at the closed position, as a ramp rather than a step. With
     * GATE_CALIBRATE on this is skipped: the sweep below is the measurement, and
     * it starts at the low end of its own window.
     */
#if !CONFIG_GATE_CALIBRATE
    servo_ramp_to(CONFIG_GATE_PULSE_CLOSED_US);
    while (s_ramping) {
        vTaskDelay(pdMS_TO_TICKS(GATE_TICK_MS));
        servo_ramp_step();
    }
#endif

#if CONFIG_GATE_CALIBRATE
    gate_calibrate();
#endif

    s_state = GATE_CLOSED;
    ESP_LOGI(TAG, "ready: sensor=GPIO%d (pull-up, %s), servo=GPIO%d, %d Hz, "
                  "closed=%dus open=%dus hold=%dms ramp=%dms",
             CONFIG_GATE_SENSOR_GPIO,
             CONFIG_GATE_SENSOR_ACTIVE_LOW ? "active low" : "active high",
             CONFIG_GATE_SERVO_GPIO, 1000000 / SERVO_PERIOD_US,
             CONFIG_GATE_PULSE_CLOSED_US, CONFIG_GATE_PULSE_OPEN_US,
             CONFIG_GATE_HOLD_MS, CONFIG_GATE_RAMP_MS);

    const TickType_t tick = pdMS_TO_TICKS(GATE_TICK_MS);
    TickType_t last_wake = xTaskGetTickCount();

    while (true) {
        vTaskDelayUntil(&last_wake, tick);

        bool raw = gpio_get_level(CONFIG_GATE_SENSOR_GPIO) != 0;
        bool pressed = CONFIG_GATE_SENSOR_ACTIVE_LOW ? !raw : raw;
        s_candidate = pressed;

        gate_step();
    }
}

/* ------------------------------------------------------------------------- */
/* Public                                                                     */
/* ------------------------------------------------------------------------- */

bool gate_start(void)
{
    /* Input: switch to GND, internal pull-up. The board also has a 1 M resistor
     * in the speaker network this pin belongs to; 3.3 V through the internal
     * ~45 k pull-up against it still reads about 3.15 V, far above the 2.48 V
     * input threshold, so the level is solid either way. */
    gpio_config_t in = {
        .pin_bit_mask = 1ULL << CONFIG_GATE_SENSOR_GPIO,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = CONFIG_GATE_SENSOR_ACTIVE_LOW ? GPIO_PULLUP_ENABLE : GPIO_PULLUP_DISABLE,
        .pull_down_en = CONFIG_GATE_SENSOR_ACTIVE_LOW ? GPIO_PULLDOWN_DISABLE : GPIO_PULLDOWN_ENABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    if (gpio_config(&in) != ESP_OK) {
        ESP_LOGE(TAG, "could not configure GPIO%d", CONFIG_GATE_SENSOR_GPIO);
        s_state = GATE_OFF;
        return false;
    }

    /* Servo: its own timer, so its 50 Hz cannot disturb the camera's 20 MHz XCLK
     * (TIMER_0) or the panel backlight (TIMER_1). 14 bit at 50 Hz is 1.22 us per
     * step, which is finer than the servo can resolve. */
    ledc_timer_config_t timer = {};
    timer.speed_mode = LEDC_LOW_SPEED_MODE;
    timer.duty_resolution = LEDC_TIMER_14_BIT;
    timer.timer_num = LEDC_TIMER_2;
    timer.freq_hz = 1000000 / SERVO_PERIOD_US;
    timer.clk_cfg = LEDC_AUTO_CLK;
    if (ledc_timer_config(&timer) != ESP_OK) {
        ESP_LOGE(TAG, "could not configure servo timer");
        s_state = GATE_OFF;
        return false;
    }

    /*
     * The first PWM the servo sees is GATE_START_PULSE_US, not the closed
     * position, and that is deliberate. There is no position feedback: whatever
     * pulse arrives first is a command, and the servo travels to it. Until the
     * arm has been calibrated, the middle of the electrical range is the one
     * place least likely to be a mechanical stop, so that is where it goes first;
     * the move to the closed position is then a ramp, like every other move.
     *
     * Calibration overrides this and starts at the low end of its own window
     * instead - see gate_calibrate().
     */
#if CONFIG_GATE_CALIBRATE
    const int start_us = CONFIG_GATE_CALIBRATE_FROM_US;
#else
    const int start_us = CONFIG_GATE_START_PULSE_US;
#endif

    ledc_channel_config_t ch = {};
    ch.gpio_num = CONFIG_GATE_SERVO_GPIO;
    ch.speed_mode = LEDC_LOW_SPEED_MODE;
    ch.channel = LEDC_CHANNEL_2;
    ch.timer_sel = LEDC_TIMER_2;
    ch.duty = pulse_to_duty(start_us);
    ch.hpoint = 0;
    if (ledc_channel_config(&ch) != ESP_OK) {
        ESP_LOGE(TAG, "could not configure servo channel");
        s_state = GATE_OFF;
        return false;
    }

    /* Start from the position just commanded, so the first ramp has a defined
     * origin rather than a guess. */
    servo_write(start_us);

    if (xTaskCreate(gate_task, "gate", 4096, NULL, 4, NULL) != pdPASS) {
        ESP_LOGE(TAG, "could not start the gate task");
        s_state = GATE_OFF;
        return false;
    }

    return true;
}

const char *gate_status(void)
{
    gate_state_t s = s_state;
    return (s >= GATE_OFF && s <= GATE_CLOSING) ? STATE_NAME[s] : "?";
}
