#pragma once

#include <stdbool.h>
#include <esp_err.h>

/**
 * Confirm the running image as good, cancelling a pending OTA rollback.
 *
 * Call this once the node has finished starting up. With
 * CONFIG_BOOTLOADER_APP_ROLLBACK_ENABLE an image installed by OTA boots as
 * ESP_OTA_IMG_PENDING_VERIFY and gets only one attempt: if it resets before
 * calling this, the bootloader rolls back to the previous working slot.
 */
void ota_mark_current_valid(void);

/** Log which slot booted and its OTA state (evidence for rollback tests). */
void ota_log_boot_info(void);

/** Start the OTA worker task. Checks once shortly after boot, then on demand. */
void ota_start(void);

/** Ask the worker to check for an update now. Returns false if one is running. */
bool ota_request_check(void);

/** Version of the running image, from esp_app_get_description(). */
const char *ota_running_version(void);

/** Version advertised by the last manifest fetch, or "-". */
const char *ota_available_version(void);

/** Short human readable outcome of the last check. */
const char *ota_last_result(void);

/** True while a check/download is in progress. */
bool ota_in_progress(void);
