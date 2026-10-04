#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Boot-time connectivity state machine.
 *
 * The flow is the one the reference project uses in
 * main/boards/common/wifi_board.cc (TryWifiConnect / OnWifiConnectTimeout /
 * ConfigModeExit), trimmed to what this camera node needs:
 *
 *   BOOT
 *    |-- no credentials in NVS ----------> PROVISIONING
 *    `-- credentials in NVS -> STA_CONNECTING
 *                                |-- got IP ----------> CONNECTED
 *                                `-- timeout --------> PROVISIONING
 *   CONNECTED
 *    `-- link lost --------> STA_RECONNECTING
 *                              |-- got IP --------> CONNECTED
 *                              `-- timeout ------> PROVISIONING
 *   PROVISIONING
 *    `-- portal saved credentials -> STA_CONNECTING
 *
 * PROVISIONING is reached by failing to connect, never by staying in a retry
 * loop forever.
 */

/** Hook called when the node becomes reachable on the LAN. */
typedef void (*provisioning_hook_t)(void);

/** Start the state machine. Call once, after wifi_init(). */
void provisioning_start(void);

/**
 * Register application hooks.
 *
 * on_connected         - the station got an IPv4 address.
 * on_disconnected      - the station lost its link (also fires during failed
 *                        association attempts).
 * on_before_config_ap  - we are about to raise the SoftAP. The application must
 *                        release TCP port 80 here, because the captive portal
 *                        binds it too. (The MJPEG server binds 80 as well, and
 *                        the two never run at the same time.)
 *
 * Hooks run in the Wi-Fi event task, so they must be short: signal a task
 * rather than doing real work here.
 */
void provisioning_set_hooks(provisioning_hook_t on_connected,
                            provisioning_hook_t on_disconnected,
                            provisioning_hook_t on_before_config_ap);

/** Erase saved credentials and make sure the portal comes up. */
void provisioning_clear_credentials(void);

/** Human readable current state, e.g. "STA_CONNECTING". */
const char *provisioning_state_name(void);

/** True while the SoftAP captive portal is up. */
bool provisioning_in_config_mode(void);

/** Starts the serial console task that accepts the `clearwifi` command. */
void provisioning_start_serial_console(void);

#ifdef __cplusplus
}
#endif
