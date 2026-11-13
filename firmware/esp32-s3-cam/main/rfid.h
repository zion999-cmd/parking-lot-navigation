/* UHF RFID reader (4D four-antenna module) on UART1.

   Wiring, settled by the GPIO audit:
     module TX -> GPIO1  (UART1 RX)
     module RX <- GPIO2  (UART1 TX)
     module VCC -> 5V, GND common, EN -> 3.3V (the module needs EN > 1.1 V)

   3.3 V TTL on both sides, so no level shifter. UART0 on 43/44 stays the console: it is the only
   debugging channel on this board (external JTAG pins are taken by the IR switch, the magnetometer
   I2C and the servo; native USB D-/D+ are the LCD's SCLK/MOSI).

   This module reports presence, not position: no phase, and a ~30 cm read radius, so it answers
   "which tags are on the table" and, at best, "which antenna zone", never a bay coordinate.
*/
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"

typedef struct {
    bool valid;
    uint8_t epc[12];        /* EPC as reported, 96 bit */
    int8_t rssi;            /* dBm, signed */
    int8_t port;            /* 1..4, the HOST port that was selected when this was read */
    uint32_t age_ms;        /* since last read */
    uint32_t reads;         /* total reads since boot */
    /* Gap statistics: how long between two sightings of this EPC. These are what the TTL should
       be derived from, which is why they are measured rather than guessed. */
    uint32_t last_gap_ms;
    uint32_t max_gap_ms;
    uint32_t gap_sum_ms;
    uint32_t gap_count;
} rfid_tag_t;

typedef struct {
    bool present;
    uint32_t select_ok;
    uint32_t select_fail;
    uint32_t inventory_ok;
    uint32_t inventory_timeout;
    uint32_t tags;
} rfid_port_stat_t;

esp_err_t rfid_start(void);
bool rfid_available(void);
int rfid_tag_count(void);
/* Copies up to `max` live tags, newest first. Returns how many were copied. */
int rfid_snapshot(rfid_tag_t *out, int max);
uint32_t rfid_polls(void);
uint32_t rfid_errors(void);
uint32_t rfid_frames_rx(void);
uint32_t rfid_notifications(void);
uint32_t rfid_bytes_rx(void);
bool rfid_port_stats(int port, rfid_port_stat_t *out);   /* port = 1..4 */
