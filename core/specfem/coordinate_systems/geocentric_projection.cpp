#include "specfem/coordinate_systems/geocentric_projection.hpp"
#include "specfem/constants.hpp"

#include <cmath>

namespace specfem {
namespace coordinate_systems {
namespace geocentric_impl {

/// Port of specfem3d_globe's `reduce()`: bring colatitude @p theta into
/// @f$ [0,\pi] @f$ and longitude @p phi into @f$ [0,2\pi) @f$. Matches globe so
/// that a tiny-negative input maps toward 0 rather than wrapping up to
/// @f$ 2\pi @f$.
void reduce(double &theta, double &phi);

} // namespace geocentric_impl
} // namespace coordinate_systems
} // namespace specfem

void specfem::coordinate_systems::geocentric_impl::reduce(double &theta,
                                                          double &phi) {
  const double pi = specfem::constants::pi_double;
  const double two_pi = 2.0 * pi;
  constexpr double tiny = 1.0e-9;
  constexpr double nudge = 1.0e-7;

  // Nudge points off the exact polar axis to avoid roundoff ambiguity.
  if (std::abs(theta) < tiny)
    theta += nudge;
  if (std::abs(phi) < tiny)
    phi += nudge;

  double th = theta;
  double ph = phi;

  // Longitude into [0, 2*pi).
  if (ph < 0.0 || ph > two_pi) {
    const int i = std::abs(static_cast<int>(ph / two_pi));
    if (ph < 0.0)
      ph += (i + 1) * two_pi;
    else if (ph > two_pi)
      ph -= i * two_pi;
    phi = ph;
  }

  // Colatitude into [0, pi], switching hemisphere when it wraps.
  if (th < 0.0 || th > pi) {
    const int i = static_cast<int>(th / pi);
    if (th > 0.0) {
      if (i % 2 != 0) {
        th = (i + 1) * pi - th;
        ph = (ph < pi) ? ph + pi : ph - pi;
      } else {
        th -= i * pi;
      }
    } else {
      if (i % 2 == 0) {
        th = -th + i * pi;
        ph = (ph < pi) ? ph + pi : ph - pi;
      } else {
        th -= i * pi;
      }
    }
    theta = th;
    phi = ph;
  }
}

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

  // theta via atan2 (matches globe's xyz_2_rthetaphi) rather than acos(z/r),
  // which loses precision at the poles.
  double theta =
      std::atan2(std::sqrt(cart.x * cart.x + cart.y * cart.y), cart.z);
  double phi = std::atan2(cart.y, cart.x);
  geocentric_impl::reduce(theta, phi);

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
  double theta =
      specfem::constants::pi_double / 2.0 -
      geographic.latitude * specfem::constants::degrees_to_radians_double;
  double phi =
      geographic.longitude * specfem::constants::degrees_to_radians_double;
  geocentric_impl::reduce(theta, phi);

  const double radius = config.r_planet - geographic.depth;

  return { radius, theta, phi };
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

  double theta = geocentric.theta;
  double phi = geocentric.phi;
  geocentric_impl::reduce(theta, phi);

  const double latitude =
      90.0 - theta * specfem::constants::radians_to_degrees_double;
  const double longitude = phi * specfem::constants::radians_to_degrees_double;
  const double depth = config.r_planet - geocentric.r;

  return { longitude, latitude, depth };
}
