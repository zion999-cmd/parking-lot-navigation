# ---------------------------------------------------------------------------
# Camera diagnostics patch
#
# camera_diag.c reads counter symbols that only exist after
# patches/camera-diagnostics.patch is applied to the downloaded camera component.
# managed_components/ is gitignored, so a clean clone has an unpatched driver and
# used to fail to compile. Apply the patch at configure time; if that is not possible,
# keep building without it (camera_diag.c reports zeros) - diagnostics must never be
# the reason the camera node cannot boot.
# ---------------------------------------------------------------------------
set(CAM_HAL_HEADER
    "${CMAKE_SOURCE_DIR}/managed_components/espressif__esp32-camera/driver/private_include/cam_hal.h")
set(CAM_DIAG_POS -1)

if(EXISTS "${CAM_HAL_HEADER}")
    file(READ "${CAM_HAL_HEADER}" CAM_HAL_TEXT)
    string(FIND "${CAM_HAL_TEXT}" "CAM_DIAG_COUNT" CAM_DIAG_POS)
    if(CAM_DIAG_POS EQUAL -1)
        find_program(CAM_PATCH_PYTHON NAMES python3 python)
        if(CAM_PATCH_PYTHON)
            execute_process(
                COMMAND "${CAM_PATCH_PYTHON}" "${CMAKE_CURRENT_LIST_DIR}/../patches/apply.py"
                WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
                RESULT_VARIABLE CAM_PATCH_RESULT
                OUTPUT_VARIABLE CAM_PATCH_OUT
                ERROR_VARIABLE CAM_PATCH_ERR)
            if(CAM_PATCH_RESULT EQUAL 0)
                message(STATUS "camera diagnostics: ${CAM_PATCH_OUT}")
                file(READ "${CAM_HAL_HEADER}" CAM_HAL_TEXT)
                string(FIND "${CAM_HAL_TEXT}" "CAM_DIAG_COUNT" CAM_DIAG_POS)
            else()
                message(WARNING "camera diagnostics patch not applied: ${CAM_PATCH_ERR}"
                                "counters will read zero")
            endif()
        else()
            message(WARNING "camera diagnostics: python3 not found; counters will read zero")
        endif()
    endif()
else()
    message(WARNING "camera diagnostics: ${CAM_HAL_HEADER} not found; counters will read zero")
endif()

idf_component_register(
    SRCS
        "main.c"
        "camera.c"
        "camera_diag.c"
        "stream.c"
        "ota.c"
        "led.c"
        "gate.c"
        "mag.c"
        "rfid.c"
        "discovery.c"
        "urlhost.c"
        "version.c"
        "display.c"
        "wifi.cpp"
        "provisioning.cpp"
    INCLUDE_DIRS "."
    # Read-only register peeks for the frame-rate investigation need the driver's SCCB header.
    # Diagnostic only; nothing in the product calls camera_peek_reg().
    PRIV_INCLUDE_DIRS "."
                      "${CMAKE_SOURCE_DIR}/managed_components/espressif__esp32-camera/driver/private_include"
    PRIV_REQUIRES
        esp_wifi
        esp_netif
        esp_http_server
        esp_http_client
        esp_timer
        esp_lcd
        esp_driver_spi
        esp_driver_ledc
        esp_driver_rmt
        esp_driver_gpio
        lwip
        nvs_flash
        app_update
        json
        mbedtls
        console
)

# The C file needs to know whether the driver patch (and therefore the counters) is present.
if(NOT CAM_DIAG_POS EQUAL -1)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE CAM_DIAG_PATCHED=1)
endif()
