/// @brief UTM forward/inverse projection (Snyder PP 1395, eqs. 8-9..8-18).

#include "specfem/coordinate_systems/projections/utm.hpp"
#include "specfem/ellipticity/ellipticity.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numbers>

namespace specfem {
namespace coordinate_systems {

namespace {

// WGS-84 ellipsoid
constexpr auto ellipsoid =
    specfem::ellipticity::ellipsoid<specfem::ellipticity::model::wgs84>{};
constexpr double semi_major_axis = ellipsoid.semi_major_axis;
constexpr double semi_minor_axis = ellipsoid.semi_minor_axis;

// UTM constants
constexpr double utm_scaling_factor = 0.9996;
constexpr double false_easting = 500000.0;
constexpr double false_northing = 0.0;

constexpr double pi = std::numbers::pi;
constexpr double degrees_to_radians = pi / 180.0;
constexpr double radians_to_degrees = 180.0 / pi;

// First eccentricity squared and powers: @f$ e^2 = 1 - (b/a)^2 @f$
constexpr double e2 = 1.0 - (semi_minor_axis / semi_major_axis) *
                                (semi_minor_axis / semi_major_axis);
constexpr double e4 = e2 * e2;
constexpr double e6 = e2 * e4;
constexpr double ep2 = e2 / (1.0 - e2); ///< Second eccentricity squared

} // anonymous namespace

cartesian_coordinates to_cartesian(const geographic_coordinates &geo,
                                   const utm_projection_config &config) {
  if (config.suppress) {
    return { geo.longitude, geo.latitude, geo.depth };
  }

  const double dlat = geo.latitude;
  const double dlon = geo.longitude;

  // Zone parameters
  const int zone = std::abs(config.zone);
  const bool lsouth = (config.zone < 0);
  const double central_meridian = zone * 6.0 - 183.0;

  // Convert to radians
  const double rlat = degrees_to_radians * dlat;

  // Longitude difference from central meridian, wrapped to [-180, 180]
  double delam = dlon - central_meridian;
  if (delam < -180.0)
    delam = delam + 360.0;
  if (delam > 180.0)
    delam = delam - 360.0;
  delam = delam * degrees_to_radians;

  // Meridional arc length M (Snyder eq. 3-21)
  const double f1 =
      (1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0) * rlat;
  const double f2 = (3.0 * e2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0) *
                    std::sin(2.0 * rlat);
  const double f3 =
      (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0) * std::sin(4.0 * rlat);
  const double f4 = (35.0 * e6 / 3072.0) * std::sin(6.0 * rlat);
  const double rm = semi_major_axis * (f1 - f2 + f3 - f4);

  double xx, yy;

  if (dlat == 90.0 || dlat == -90.0) {
    // Special case: poles
    xx = 0.0;
    yy = utm_scaling_factor * rm;
  } else {
    // Radius of curvature in prime vertical (Snyder eq. 4-20)
    const double sin_rlat = std::sin(rlat);
    const double cos_rlat = std::cos(rlat);
    const double rn =
        semi_major_axis / std::sqrt(1.0 - e2 * sin_rlat * sin_rlat);

    // Snyder eq. 8-13, 8-14
    const double t = std::tan(rlat) * std::tan(rlat);
    const double c = ep2 * cos_rlat * cos_rlat;
    const double a = cos_rlat * delam;

    const double a2 = a * a;
    const double a3 = a2 * a;
    const double a4 = a3 * a;
    const double a5 = a4 * a;
    const double a6 = a5 * a;

    // Snyder eq. 8-9 for easting (x)
    const double x_f1 = (1.0 - t + c) * a3 / 6.0;
    const double x_f2 =
        (5.0 - 18.0 * t + t * t + 72.0 * c - 58.0 * ep2) * a5 / 120.0;
    xx = utm_scaling_factor * rn * (a + x_f1 + x_f2);

    // Snyder eq. 8-10 for northing (y)
    const double y_f1 = a2 / 2.0;
    const double y_f2 = (5.0 - t + 9.0 * c + 4.0 * c * c) * a4 / 24.0;
    const double y_f3 =
        (61.0 - 58.0 * t + t * t + 600.0 * c - 330.0 * ep2) * a6 / 720.0;
    yy = utm_scaling_factor * (rm + rn * std::tan(rlat) * (y_f1 + y_f2 + y_f3));
  }

  xx = xx + false_easting;
  yy = yy + false_northing;

  // Southern hemisphere: add 10,000 km offset
  if (lsouth)
    yy = yy + 1.0e7;

  return { xx, yy, geo.depth };
}

geographic_coordinates to_geographic(const cartesian_coordinates &cart,
                                     const utm_projection_config &config) {
  if (config.suppress) {
    return { cart.x, cart.y, cart.z };
  }

  // Zone parameters
  const int zone = std::abs(config.zone);
  const bool lsouth = (config.zone < 0);
  const double central_meridian = zone * 6.0 - 183.0;
  const double cmr = central_meridian * degrees_to_radians;

  // Remove false easting/northing and southern hemisphere offset
  double xx = cart.x - false_easting;
  double yy = cart.y - false_northing;
  if (lsouth)
    yy = yy - 1.0e7;

  // Snyder eq. 3-24 for e1
  const double sqrt_1_minus_e2 = std::sqrt(1.0 - e2);
  const double e1 = (1.0 - sqrt_1_minus_e2) / (1.0 + sqrt_1_minus_e2);

  // Snyder eq. 7-19 for footpoint latitude (mu)
  const double rm = yy / utm_scaling_factor;
  const double mu_denom = 1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0;
  const double mu = rm / (semi_major_axis * mu_denom);

  // Snyder eq. 3-26 for footpoint latitude (phi_1)
  const double e1_2 = e1 * e1;
  const double e1_3 = e1_2 * e1;
  const double e1_4 = e1_3 * e1;
  const double fp_f1 =
      (3.0 * e1 / 2.0 - 27.0 * e1_3 / 32.0) * std::sin(2.0 * mu);
  const double fp_f2 =
      (21.0 * e1_2 / 16.0 - 55.0 * e1_4 / 32.0) * std::sin(4.0 * mu);
  const double fp_f3 = (151.0 * e1_3 / 96.0) * std::sin(6.0 * mu);
  const double rlat1 = mu + fp_f1 + fp_f2 + fp_f3;
  double dlat1 = rlat1 * radians_to_degrees;

  double dlat, dlon;

  if (dlat1 >= 90.0 || dlat1 <= -90.0) {
    // Clamp to poles
    dlat1 = std::min(dlat1, 90.0);
    dlat1 = std::max(dlat1, -90.0);
    dlat = dlat1;
    dlon = central_meridian;
  } else {
    const double sin_rlat1 = std::sin(rlat1);
    const double cos_rlat1 = std::cos(rlat1);
    const double tan_rlat1 = std::tan(rlat1);

    const double c1 = ep2 * cos_rlat1 * cos_rlat1;
    const double t1 = tan_rlat1 * tan_rlat1;

    const double f1_denom = 1.0 - e2 * sin_rlat1 * sin_rlat1;
    // Radius of curvature in prime vertical (Snyder eq. 4-20)
    const double rn1 = semi_major_axis / std::sqrt(f1_denom);
    // Radius of curvature in meridional plane
    const double r1 = semi_major_axis * (1.0 - e2) /
                      std::sqrt(f1_denom * f1_denom * f1_denom);
    const double d = xx / (rn1 * utm_scaling_factor);

    const double d2 = d * d;
    const double d3 = d2 * d;
    const double d4 = d3 * d;
    const double d5 = d4 * d;
    const double d6 = d5 * d;

    // Snyder eq. 8-17 for latitude (phi)
    const double lat_f1 = rn1 * tan_rlat1 / r1;
    const double lat_f2 = d2 / 2.0;
    const double lat_f3 =
        (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1 * c1 - 9.0 * ep2) * d4 / 24.0;
    const double lat_f4 = (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1 * t1 -
                           252.0 * ep2 - 3.0 * c1 * c1) *
                          d6 / 720.0;
    const double rlat = rlat1 - lat_f1 * (lat_f2 - lat_f3 + lat_f4);
    dlat = rlat * radians_to_degrees;

    // Snyder eq. 8-18 for longitude (lambda)
    const double lon_f1 = (1.0 + 2.0 * t1 + c1) * d3 / 6.0;
    const double lon_f2 = (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1 * c1 +
                           8.0 * ep2 + 24.0 * t1 * t1) *
                          d5 / 120.0;
    const double rlon = cmr + (d - lon_f1 + lon_f2) / cos_rlat1;
    dlon = rlon * radians_to_degrees;

    // Wrap longitude to [-180, 180]
    if (dlon < -180.0)
      dlon = dlon + 360.0;
    if (dlon > 180.0)
      dlon = dlon - 360.0;
  }

  return { dlon, dlat, cart.z };
}

} // namespace coordinate_systems
} // namespace specfem
