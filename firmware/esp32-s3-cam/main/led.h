#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Bay status lights: one WS2812B ("2812") pixel per parking bay, driven from the
 * bay states that PC Vision already publishes.
 *
 * The colour is the whole interface, and it is a three-state semantic, not a
 * decoration:
 *
 *   green   EMPTY      the bay can be taken
 *   red     OCCUPIED   the bay is taken
 *   yellow  NAVIGATION reserved: the bay the current route targets
 *   off     UNKNOWN / no data
 *
 * Two rules this module will not break, because the consequences are worse than
 * a dark pixel:
 *
 *   1. UNKNOWN is dark, never green. parking_state/1 defines UNKNOWN for frames
 *      that are missing, and contract.py is explicit that a bay with no fresh
 *      frame must not be reported as EMPTY. A strip that goes green when the
 *      vision side stops talking is worse than a strip that goes dark: it
 *      advertises free bays it has not looked at.
 *   2. A failed poll blanks the strip and says so. Stale colour is the same lie,
 *      just slower.
 *
 * Where the state comes from: GET CONFIG_LED_STATE_URL, which is what
 * src/serve.py already serves for the web pages. The device pulls; nothing
 * pushes into the camera node, so the one-way video boundary is widened into
 * "video out, state in" and not into an inbound control channel.
 *
 * Which pixel is which bay is an installation fact (where the strip was glued),
 * so it is configured in LED_ORDER rather than inferred from the map.
 */

/**
 * Start the light task. Safe to call once at boot, before Wi-Fi is up: the task
 * waits for a connection on its own and blanks the strip whenever it cannot
 * read a fresh state document.
 *
 * Returns false if the strip could not be set up. The camera node is unaffected
 * either way - like the status panel, the lights are best effort.
 */
bool led_start(void);

/** Short status for the diagnostics line: "ok" / "stale" / "off" / "disabled". */
const char *led_status(void);

#ifdef __cplusplus
}
#endif
