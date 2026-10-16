/*
 * Host tests for urlhost_rewrite(). Same idea as version_test.c: the function is
 * pure string handling, so the cases that are awkward to reach on the bench (a URL
 * with no scheme, a host that would not fit, a template that must be left alone)
 * are checked here instead.
 *
 *   tools/tests/run.sh
 */

#include <stdio.h>
#include <string.h>

#include "urlhost.h"

static int failures;
static int checks;

static void expect(const char *name, const char *url_in, const char *host,
                   const char *want, bool want_ok)
{
    char buf[128];
    snprintf(buf, sizeof(buf), "%s", url_in);     /* buffer the caller owns */

    bool ok = urlhost_rewrite(buf, sizeof(buf), host);
    checks++;

    if (ok != want_ok) {
        printf("FAIL %s: returned %s, wanted %s\n", name, ok ? "true" : "false",
               want_ok ? "true" : "false");
        failures++;
        return;
    }
    if (want_ok && strcmp(buf, want) != 0) {
        printf("FAIL %s:\n  got  %s\n  want %s\n", name, buf, want);
        failures++;
        return;
    }
    if (!want_ok && strcmp(buf, url_in) != 0) {
        printf("FAIL %s: refused but modified the buffer: %s\n", name, buf);
        failures++;
    }
}

int main(void)
{
    /* The real templates, with the host swapped. */
    expect("state url", "http://192.168.3.102:8010/out/parking_state.json", "10.0.0.7",
           "http://10.0.0.7:8010/out/parking_state.json", true);
    expect("manifest url", "http://192.168.3.102:8000/manifest.json", "10.0.0.7",
           "http://10.0.0.7:8000/manifest.json", true);
    expect("qr url", "http://192.168.3.102:8010/go", "parking-pc.local",
           "http://parking-pc.local:8010/go", true);

    /* The port survives; only the host was asked to move. */
    expect("port kept", "http://1.2.3.4:9999/x", "5.6.7.8", "http://5.6.7.8:9999/x", true);

    /* Shorter and longer hosts, both directions. */
    expect("longer host", "http://a.b:80/", "192.168.100.200", "http://192.168.100.200:80/", true);
    expect("shorter host", "http://192.168.100.200:80/", "a.b", "http://a.b:80/", true);

    /* No path, and path with a query. */
    expect("no path", "http://1.2.3.4:80", "9.9.9.9", "http://9.9.9.9:80", true);
    expect("query", "http://1.2.3.4:80/s?spot=38", "9.9.9.9", "http://9.9.9.9:80/s?spot=38", true);
    expect("fragment", "http://1.2.3.4:80/a#b", "9.9.9.9", "http://9.9.9.9:80/a#b", true);

    /* A host with no port at all. */
    expect("no port", "http://1.2.3.4/out/x.json", "9.9.9.9", "http://9.9.9.9/out/x.json", true);

    /* https is left https: the scheme is not this function's business. */
    expect("https", "https://1.2.3.4:443/m.json", "9.9.9.9", "https://9.9.9.9:443/m.json", true);

    /* --- refusals, each of which must leave the template alone --- */

    /* No scheme: not a URL this can rewrite, so it must stay as written. */
    expect("no scheme", "192.168.3.102:8010/x", "9.9.9.9", "", false);
    expect("plain path", "/out/parking_state.json", "9.9.9.9", "", false);
    expect("empty", "", "9.9.9.9", "", false);

    /* Bad new hosts: a rewrite that produces a broken URL is worse than none. */
    expect("host with slash", "http://1.2.3.4:80/x", "9.9/9.9", "", false);
    expect("host with space", "http://1.2.3.4:80/x", "9 9", "", false);
    expect("host with colon", "http://1.2.3.4:80/x", "9.9.9.9:80", "", false);
    expect("empty host", "http://1.2.3.4:80/x", "", "", false);
    expect("null host", "http://1.2.3.4:80/x", NULL, "", false);

    /* A template with no host at all. */
    expect("scheme only", "http://:80/x", "9.9.9.9", "", false);

    /* Does not fit: 8 chars into a URL already at the limit. */
    {
        char tight[32];
        snprintf(tight, sizeof(tight), "http://1.2.3.4:80/abcdefghij");
        bool ok = urlhost_rewrite(tight, sizeof(tight), "10.20.30.40");
        checks++;
        if (ok || strcmp(tight, "http://1.2.3.4:80/abcdefghij") != 0) {
            printf("FAIL too long: should refuse and leave the buffer alone (%s)\n", tight);
            failures++;
        }
    }

    /* Exactly fits: the same buffer, a host that lines up. */
    {
        char exact[32];
        snprintf(exact, sizeof(exact), "http://1.2.3.4:80/abcdefghij");
        bool ok = urlhost_rewrite(exact, sizeof(exact), "10.20.30.4");
        checks++;
        if (!ok || strcmp(exact, "http://10.20.30.4:80/abcdefghij") != 0) {
            printf("FAIL exact fit: got %s (ok=%d)\n", exact, ok);
            failures++;
        }
    }

    printf("urlhost: %d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
