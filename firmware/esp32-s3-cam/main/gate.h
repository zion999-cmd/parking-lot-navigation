#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Barrier gate: a contact switch says a car is here, a servo lifts the arm.
 *
 *   GPIO39  switch, one side to GND          (input, internal pull-up)
 *   GPIO40  servo signal                     (LEDC, TIMER_2 / CHANNEL_2, 50 Hz)
 *
 * Why this lives on the camera node rather than a second board: it is local
 * automation - sensor in, actuator out, no network in the loop. It has to work
 * when the PC is off, so gate_start() is called at boot and not from the network
 * task, and the gate never reads the parking state.
 *
 * Resources, all checked against the sources rather than the config files:
 *
 *   GPIO39/40   free. The only other users are the speaker I2S lines in the
 *               reference project (SPK_DOUT=39, BCLK=40) and this firmware never
 *               initialises audio. Cost of using them: they are the ESP32-S3's
 *               JTAG pins (io_mux_reg.h: IO_MUX_GPIO39_REG = MTCK, 40 = MTDO),
 *               so an external JTAG probe can no longer be attached. Crash
 *               backtraces are unaffected - CONFIG_ESP_GDBSTUB_ENABLED uses the
 *               console UART.
 *   LEDC        TIMER_0/CH0 is the camera XCLK (esp32-camera target/xclk.c) and
 *               TIMER_1/CH1 is the TFT backlight (display.c). TIMER_2/CH2 is
 *               unused, and a timer's frequency belongs to that timer alone, so
 *               the three cannot interfere.
 *   RMT         not used here. The WS2812 strip owns that (led.c).
 *
 * The switch is an input: it draws about 73 uA through the pull-up and drives
 * nothing. The current in this system is the servo's, and it shares the strip's
 * 5 V rail - see CONFIG_GATE_HOLD_MS and the README for what that costs.
 */

/**
 * Bring up the input and the servo, then start the gate task.
 *
 * The arm is driven to the closed position at startup, slowly, rather than being
 * left wherever it was. Returns false if the servo could not be configured; the
 * camera node is unaffected either way, like the panel and the strip.
 */
bool gate_start(void);

/** Current state for the diagnostics line: "closed" / "opening" / "open" / "closing" / "off". */
const char *gate_status(void);

#ifdef __cplusplus
}
#endif
