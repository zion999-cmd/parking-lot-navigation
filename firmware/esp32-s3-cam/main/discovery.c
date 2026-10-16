#include "discovery.h"

#include <errno.h>
#include <string.h>

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include <esp_log.h>
#include <esp_timer.h>
#include <lwip/inet.h>
#include <lwip/sockets.h>
#include <sdkconfig.h>

#include "urlhost.h"

/*
 * Server discovery - implementation. Read discovery.h for why this exists and
 * what it deliberately does not do.
 *
 * The packet's *source address* is the answer; its contents are only a check that
 * this is our packet at all. A field inside the packet claiming to be the host
 * would be a second source of truth for the same fact, and the one that arrived
 * on the wire is the one that demonstrably works - a host that can reach this
 * device is exactly what the device needs.
 */

static const char *TAG = "discovery";

/* Any packet that starts with this is ours. Trailing content is ignored, so the
 * beacon can carry readable details for a human without this having to parse them. */
static const char MAGIC[] = "PARKINGLOT/1";

/* A beacon is sent every couple of seconds. Treating one older than this as gone
 * means a PC that walks away stops being used, and the configured address comes
 * back - rather than the device spending the rest of the day talking to a machine
 * that is no longer there. */
#define DISCOVERY_STALE_MS 15000

static char s_host[64];
static int64_t s_seen_us;                 /* 0 = never heard a beacon */
static portMUX_TYPE s_lock = portMUX_INITIALIZER_UNLOCKED;

static volatile bool s_logged_fallback;

void discovery_url(const char *template_url, char *out, size_t size)
{
    char host[64];

    if (out == NULL || size == 0) {
        return;
    }

    discovery_host(host, sizeof(host));

    if (host[0] != '\0' && template_url != NULL) {
        snprintf(out, size, "%s", template_url);
        if (urlhost_rewrite(out, size, host)) {
            return;
        }
        /* urlhost refused (a template it cannot parse safely): fall through and
         * use the template as written rather than something half rewritten. */
    }

    snprintf(out, size, "%s", template_url ? template_url : "");
}

void discovery_host(char *out, size_t size)
{
    int64_t now = esp_timer_get_time();
    bool live;

    if (out == NULL || size == 0) {
        return;
    }

    /* The writer is another task, so the copy happens under the lock. Time is read
     * before taking it: a spinlock should not be held across a call that is not
     * part of the read-modify-write. */
    portENTER_CRITICAL(&s_lock);
    live = s_seen_us != 0 && (now - s_seen_us) / 1000 <= DISCOVERY_STALE_MS;
    snprintf(out, size, "%s", live ? s_host : "");
    portEXIT_CRITICAL(&s_lock);
}

static void note_beacon(const char *from_ip, const char *payload, int len)
{
    bool changed;

    portENTER_CRITICAL(&s_lock);
    changed = strcmp(s_host, from_ip) != 0;
    snprintf(s_host, sizeof(s_host), "%s", from_ip);
    s_seen_us = esp_timer_get_time();
    portEXIT_CRITICAL(&s_lock);

    if (changed) {
        /* Logged because a device that silently switched servers would be very
         * hard to explain from the other end: the strip would follow a PC that
         * nobody configured. */
        ESP_LOGI(TAG, "server is %s (beacon: %.*s)", from_ip, len, payload);
    }
    s_logged_fallback = false;
}

static void discovery_task(void *arg)
{
    (void)arg;

    for (;;) {
        int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (sock < 0) {
            ESP_LOGW(TAG, "no socket for discovery (%d), retrying", errno);
            vTaskDelay(pdMS_TO_TICKS(5000));
            continue;
        }

        struct sockaddr_in addr = {
            .sin_family = AF_INET,
            .sin_addr.s_addr = htonl(INADDR_ANY),
            .sin_port = htons(CONFIG_DISCOVERY_PORT),
        };
        if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            ESP_LOGW(TAG, "could not bind UDP %d (%d), retrying", CONFIG_DISCOVERY_PORT, errno);
            close(sock);
            vTaskDelay(pdMS_TO_TICKS(5000));
            continue;
        }

        /* A read timeout so the task can notice when the beacon stops, and say so
         * once, rather than sitting silently on a stale host forever. */
        struct timeval tv = { .tv_sec = 2, .tv_usec = 0 };
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        ESP_LOGI(TAG, "listening for a beacon on UDP %d", CONFIG_DISCOVERY_PORT);

        for (;;) {
            char buf[256];
            struct sockaddr_in from;
            socklen_t from_len = sizeof(from);

            int n = recvfrom(sock, buf, sizeof(buf) - 1, 0, (struct sockaddr *)&from, &from_len);
            if (n < 0) {
                /* Timeout: no beacon for a while. Say it once, and keep the socket
                 * open - the PC may simply be restarting. */
                char current[64];
                discovery_host(current, sizeof(current));
                if (current[0] == '\0' && !s_logged_fallback) {
                    s_logged_fallback = true;
                    ESP_LOGW(TAG, "no beacon heard; using the configured addresses");
                }
                continue;
            }
            buf[n] = '\0';

            if (n < (int)sizeof(MAGIC) - 1 || strncmp(buf, MAGIC, sizeof(MAGIC) - 1) != 0) {
                continue;                  /* someone else's broadcast */
            }

            char ip[INET_ADDRSTRLEN] = "";
            inet_ntoa_r(from.sin_addr, ip, sizeof(ip));
            note_beacon(ip, buf, n);
        }
    }
}

bool discovery_start(void)
{
    if (xTaskCreate(discovery_task, "discovery", 3584, NULL, 3, NULL) != pdPASS) {
        ESP_LOGE(TAG, "could not start the discovery task; addresses stay as configured");
        return false;
    }
    return true;
}
