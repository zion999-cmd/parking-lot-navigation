cmake_minimum_required(VERSION 3.16)

# Reported at runtime as esp_app_get_description()->version and compared by the
# thin OTA client against the version in the server manifest. Keep it strictly
# in major.minor.patch form - ota.c compares the three components as integers.
set(PROJECT_VER "0.2.36")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(parking-lot-cam)
