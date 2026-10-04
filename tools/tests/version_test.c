/*
 * Host unit test for the firmware version rule.
 *
 * The requirement that matters here is that the comparison is numeric, not
 * lexicographic: "0.10.0" must be newer than "0.9.0". A plain strcmp() or a
 * string-ordering shortcut gets that wrong, which would either hide a real
 * update or, worse, offer a bogus downgrade.
 *
 * Run with: tools/tests/run.sh
 */

#include <stdio.h>
#include <string.h>

#include "version.h"

static int failures = 0;
static int checks = 0;

static void expect(const char *a, const char *b, const char *want)
{
    int got = version_cmp(a, b);
    const char *got_s = got < 0 ? "<" : (got > 0 ? ">" : "==");
    checks++;

    if (strcmp(got_s, want) != 0) {
        failures++;
        printf("  FAIL  cmp(%-8s, %-8s) = %s   expected %s\n", a, b, got_s, want);
    } else {
        printf("  ok    cmp(%-8s, %-8s) = %s\n", a, b, got_s);
    }
}

int main(void)
{
    printf("version_cmp() tests\n");

    /* equality and the plain ordering */
    expect("0.1.0", "0.1.0", "==");
    expect("0.1.0", "0.1.1", "<");
    expect("0.2.0", "0.1.9", ">");
    expect("1.0.0", "0.9.9", ">");

    /* the case a string comparison gets wrong */
    expect("0.10.0", "0.9.0", ">");
    expect("0.9.0", "0.10.0", "<");
    expect("0.10.0", "0.2.0", ">");

    /* patch level 10 vs 9 */
    expect("1.2.10", "1.2.9", ">");

    /* missing components are zero */
    expect("0.9", "0.9.0", "==");
    expect("0.9", "0.9.1", "<");
    expect("0.10", "0.9.9", ">");

    /* optional leading v */
    expect("v0.2.0", "0.2.0", "==");
    expect("V0.3.0", "v0.2.9", ">");

    /* the versions this project actually uses */
    expect("0.1.0", "0.2.0", "<");
    expect("0.2.0", "0.2.1", "<");
    expect("0.2.1", "0.2.0", ">");

    printf("\n%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
