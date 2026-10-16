#pragma once

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Replacing the host in a URL, keeping scheme, port and path.
 *
 * This exists because the addresses in this firmware point at a PC whose IP comes
 * from DHCP, so the workspace moving to another Wi-Fi changes them. The rest of
 * that story lives in discovery.c (which finds the new host); this file is only
 * the string operation, and it deliberately has no ESP-IDF dependency so the
 * awkward cases can be tested on a PC - see tools/tests/urlhost_test.c.
 *
 *     http://192.168.3.102:8010/out/parking_state.json  +  "10.0.0.7"
 *  -> http://10.0.0.7:8010/out/parking_state.json
 *
 * What it deliberately does NOT do:
 *
 *   - change the port. Ports are configuration (a service on another port is a
 *     different service); hosts are the environment. Only the environment moves.
 *   - invent a scheme or a path. If the input has no "://", nothing happens: a
 *     template this cannot parse is a template that must keep working as written.
 *   - accept a host that would produce a malformed URL. A bad rewrite is worse
 *     than no rewrite, because the caller will use it without checking.
 */

/**
 * Rewrite `url` in place, replacing its authority (host, and any userinfo/port
 * already covered by the scheme separator) with `new_host`.
 *
 * Returns false and leaves `url` untouched when the input is not something this
 * can rewrite safely: no scheme, an empty host, a host containing characters that
 * cannot appear in one, or a result that would not fit.
 */
bool urlhost_rewrite(char *url, size_t size, const char *new_host);

#ifdef __cplusplus
}
#endif
