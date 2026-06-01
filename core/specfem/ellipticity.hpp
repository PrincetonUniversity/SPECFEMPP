#pragma once

/**
 * @brief Umbrella header for reference ellipsoid models.
 *
 * Provides compile-time ellipsoid parameters (semi-major/minor axes) for
 * geodetic reference models (WGS-84, Clarke 1866). Include this header
 * rather than individual ellipticity sub-headers.
 */

#include "ellipticity/ellipticity.hpp"
#include "ellipticity/model.hpp"
