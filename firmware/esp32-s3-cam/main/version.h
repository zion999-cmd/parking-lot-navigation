#pragma once

/**
 * Compare two "major.minor.patch" version strings.
 *
 * An optional leading 'v' is ignored and missing components count as 0, so
 * "0.9" is read as 0.9.0.
 *
 * Returns <0 if a is older than b, 0 if equal, >0 if a is newer.
 *
 * The comparison is numeric, component by component, never lexicographic:
 * "0.10.0" is correctly newer than "0.9.0".
 *
 * This file deliberately has no ESP-IDF dependency so the rule can be unit
 * tested on a PC - see tools/tests/.
 */
int version_cmp(const char *a, const char *b);
