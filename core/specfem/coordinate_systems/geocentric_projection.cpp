#include "specfem/coordinate_systems/geocentric_projection.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace specfem {
namespace coordinate_systems {
namespace geocentric_impl {

constexpr double pi = std::numbers::pi;
constexpr double degrees_to_radians = pi / 180.0;
constexpr double radians_to_degrees = 180.0 / pi;

/// Wrap a longitude in radians to [0, 2*pi), matching globe's reduce().
double normalize_longitude(double radians) {
  radians = std::fmod(radians, 2.0 * pi);
  if (radians < 0.0)
    radians += 2.0 * pi;
  return radians;
}

} // namespace geocentric_impl
} // namespace coordinate_systems
} // namespace specfem

template <>
specfem::coordinate_systems::cartesian_coordinates<
    specfem::element::dimension_tag::dim3>
specfem::coordinate_systems::transform<
    specfem::coordinate_systems::cartesian_coordinates<
        specfem::element::dimension_tag::dim3>,
    specfem::coordinate_systems::geocentric_coordinates>(
    const specfem::coordinate_systems::geocentric_coordinates &geo) {

  const double sin_theta = std::sin(geo.theta);
  const double x = geo.r * sin_theta * std::cos(geo.phi);
  const double y = geo.r * sin_theta * std::sin(geo.phi);
  const double z = geo.r * std::cos(geo.theta);

  // Absolute coordinates: the planet center is the Cartesian origin.
  return { x, y, z };
}

template <>
specfem::coordinate_systems::geocentric_coordinates
specfem::coordinate_systems::transform<
    specfem::coordinate_systems::geocentric_coordinates,
    specfem::coordinate_systems::cartesian_coordinates<
        specfem::element::dimension_tag::dim3>>(
    const specfem::coordinate_systems::cartesian_coordinates<
        specfem::element::dimension_tag::dim3> &cart) {

  namespace geocentric_impl = specfem::coordinate_systems::geocentric_impl;

  const double r =
      std::sqrt(cart.x * cart.x + cart.y * cart.y + cart.z * cart.z);
  if (r == 0.0)
    return { 0.0, 0.0, 0.0 };

  // acos argument clamped against round-off at the poles.
  const double theta = std::acos(std::clamp(cart.z / r, -1.0, 1.0));
  const double phi =
      geocentric_impl::normalize_longitude(std::atan2(cart.y, cart.x));

  return { r, theta, phi };
}

template <>
specfem::coordinate_systems::geocentric_coordinates
specfem::coordinate_systems::transform<
    specfem::coordinate_systems::geocentric_coordinates,
    specfem::coordinate_systems::geographic_coordinates,
    specfem::coordinate_systems::geocentric_projection_config>(
    const specfem::coordinate_systems::geographic_coordinates &geographic,
    const specfem::coordinate_systems::geocentric_projection_config &config) {

  namespace geocentric_impl = specfem::coordinate_systems::geocentric_impl;

  // Perfect sphere: geographic latitude is the geocentric colatitude directly
  // (no (1-f)^2 flattening — that enters with the elliptical case).
  const double colatitude =
      geocentric_impl::pi / 2.0 -
      geographic.latitude * geocentric_impl::degrees_to_radians;
  const double longitude = geocentric_impl::normalize_longitude(
      geographic.longitude * geocentric_impl::degrees_to_radians);

  const double radius = config.r_planet - geographic.depth;

  return { radius, colatitude, longitude };
}

template <>
specfem::coordinate_systems::geographic_coordinates
specfem::coordinate_systems::transform<
    specfem::coordinate_systems::geographic_coordinates,
    specfem::coordinate_systems::geocentric_coordinates,
    specfem::coordinate_systems::geocentric_projection_config>(
    const specfem::coordinate_systems::geocentric_coordinates &geocentric,
    const specfem::coordinate_systems::geocentric_projection_config &config) {

  namespace geocentric_impl = specfem::coordinate_systems::geocentric_impl;

  const double latitude =
      90.0 - geocentric.theta * geocentric_impl::radians_to_degrees;
  const double longitude = geocentric.phi * geocentric_impl::radians_to_degrees;
  const double depth = config.r_planet - geocentric.r;

  return { longitude, latitude, depth };
}
