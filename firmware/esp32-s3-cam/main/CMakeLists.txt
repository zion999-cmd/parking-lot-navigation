idf_component_register(
    SRCS
        "main.c"
        "camera.c"
        "camera_diag.c"
        "stream.c"
        "ota.c"
        "led.c"
        "gate.c"
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
