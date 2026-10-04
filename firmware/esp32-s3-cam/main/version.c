#include "version.h"

#include <stdio.h>

/* Parse "major.minor.patch"; an optional leading 'v' is ignored. */
static void parse_version(const char *s, int out[3])
{
    out[0] = out[1] = out[2] = 0;
    if (s == NULL) {
        return;
    }
    if (*s == 'v' || *s == 'V') {
        s++;
    }
    /* Missing components stay 0, so "0.9" is read as 0.9.0. */
    sscanf(s, "%d.%d.%d", &out[0], &out[1], &out[2]);
}

int version_cmp(const char *a, const char *b)
{
    int va[3], vb[3];
    parse_version(a, va);
    parse_version(b, vb);

    for (int i = 0; i < 3; i++) {
        if (va[i] != vb[i]) {
            return va[i] < vb[i] ? -1 : 1;
        }
    }
    return 0;
}
