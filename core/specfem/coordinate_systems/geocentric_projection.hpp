#pragma once

#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/coordinate_systems/geocentric.hpp"
#include "specfem/coordinate_systems/geographic.hpp"

namespace specfem {
namespace coordinate_systems {

/**
 * @brief Configuration for geographic/geocentric conversion methods.
 *
 * The reference surface radius is supplied by the caller. For a perfect
 * sphere it is the planet radius; a globe resolver may supply a deformed
 * surface radius along the ray instead. Ellipticity and topography remain
 * resolution policies of the caller, not part of these spherical transforms.
 *
 * @code
 * const geocentric_projection_config config{ 6371000.0 };
 * const geographic_coordinates geographic{ 10.0, 45.0, 1000.0 };
 * const auto geocentric = geographic.to_geocentric(config);
 * const auto cartesian = geocentric.to_cartesian();
 * const auto recovered = cartesian.to_geocentric().to_geographic(config);
 * @endcode
 */
struct geocentric_projection_config {
  double r_planet; ///< Reference surface radius in meters.
};

} // namespace coordinate_systems
} // namespace specfem
