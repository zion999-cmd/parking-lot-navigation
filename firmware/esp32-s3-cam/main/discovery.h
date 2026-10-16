#pragma once

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Finding the PC without the PC's address being compiled in.
 *
 * The problem this solves: the firmware's three server URLs and the QR code all
 * name a PC whose address comes from DHCP. Carry the sandbox to another Wi-Fi and
 * every one of them is wrong, and the only way to fix it was to rebuild.
 *
 * How: the PC broadcasts a small UDP packet every couple of seconds (tools/beacon.py).
 * The device listens, and uses the *source address* of that packet as the server.
 *
 * Why UDP broadcast rather than mDNS:
 *
 *   - no extra component: mDNS is a real component with real flash cost, and this
 *     sandbox is a single subnet where a broadcast is enough
 *   - no router configuration, no hostname resolution, no service registration
 *   - nothing to configure at the venue: it works on whatever network it lands on
 *
 * What it costs: broadcasts do not cross a router, so the device and the PC must
 * share a subnet. On a guest network with client isolation neither this nor plain
 * HTTP works, so the limitation is not specific to discovery.
 *
 * What it deliberately does not change: ports. A service on another port is a
 * different service, and that is configuration; only the host is the environment.
 * See urlhost.c.
 *
 * Fallback: if no beacon has ever been heard, or the last one is stale, every URL
 * is used exactly as configured. The discovery is an improvement, never a
 * requirement.
 *
 * Ordering: start this after wifi_init(), never before. See discovery_start().
 */

/**
 * Start the listener task.
 *
 * MUST be called after wifi_init(). That is not a preference: wifi_init() is what
 * runs esp_netif_init() and therefore tcpip_init(), and socket() before it aborts
 * ("tcpip_send_msg_wait_sem: Invalid mbox"). This doc comment used to say the
 * opposite - "safe to call before Wi-Fi is up" - and that sentence is why the call
 * was placed too early, which panicked every boot of the new image before it could
 * confirm itself to the bootloader and produced an OTA/rollback loop. The order is
 * enforced by the note at the call site in main.c; do not move it back.
 *
 * Returns false only if the task could not be created; the caller carries on with
 * the configured addresses.
 */
bool discovery_start(void);

/**
 * Copy `template_url` to `out`, replacing its host with the discovered one.
 *
 * Falls back to a plain copy whenever there is nothing to replace it with: no
 * beacon heard, a stale one, or a template this cannot rewrite safely. So callers
 * do not need a fallback of their own, which is the point - one place decides.
 */
void discovery_url(const char *template_url, char *out, size_t size);

/**
 * Copy the discovered host into `out`, or "" if there is none (never heard, or
 * stale). Copies rather than returning a pointer on purpose: two tasks can ask at
 * once, and a shared static buffer would be overwritten mid-read.
 */
void discovery_host(char *out, size_t size);

#ifdef __cplusplus
}
#endif
