#include "urlhost.h"

#include <stdio.h>
#include <string.h>

/* A host fits in a URL when it is made of these, and nothing else: letters,
 * digits, '.', '-' and '_'. Anything else (a '/' or a space, say) would silently
 * turn the rest of the URL into something else, which is worse than not rewriting
 * at all. Colons are excluded on purpose: this rewrites a host, not an authority
 * with a port, and a "host" containing a colon is almost certainly a mistake. */
static bool host_is_plausible(const char *host)
{
    size_t n = 0;

    if (host == NULL || *host == '\0') {
        return false;
    }
    for (const char *p = host; *p != '\0'; p++, n++) {
        if (n >= 255) {
            return false;               /* longer than a hostname can be */
        }
        bool ok = (*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
                  (*p >= '0' && *p <= '9') || *p == '.' || *p == '-' || *p == '_';
        if (!ok) {
            return false;
        }
    }
    return true;
}

bool urlhost_rewrite(char *url, size_t size, const char *new_host)
{
    if (url == NULL || size == 0 || !host_is_plausible(new_host)) {
        return false;
    }

    const char *scheme_end = strstr(url, "://");
    if (scheme_end == NULL) {
        return false;
    }
    const char *host_start = scheme_end + 3;

    /* The authority ends at the first '/' (path), '?' or '#'. A ':' ends the host
     * and starts a port, which is kept: see the header. */
    const char *authority_end = host_start;
    while (*authority_end != '\0' && *authority_end != '/' &&
           *authority_end != '?' && *authority_end != '#') {
        authority_end++;
    }

    const char *host_end = authority_end;
    for (const char *p = host_start; p < authority_end; p++) {
        if (*p == ':') {
            host_end = p;
            break;
        }
    }

    if (host_end == host_start) {
        return false;                   /* "http://:8010/..." has no host to replace */
    }

    size_t prefix = (size_t)(host_start - url);
    size_t suffix_len = strlen(host_end);
    size_t host_len = strlen(new_host);

    if (prefix + host_len + suffix_len + 1 > size) {
        return false;                   /* would not fit; leave the caller's copy alone */
    }

    char *tail = url + prefix + host_len;
    memmove(tail, host_end, suffix_len + 1);     /* +1 moves the terminator too */
    memcpy(url + prefix, new_host, host_len);
    return true;
}
