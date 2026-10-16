#pragma once

#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/coordinate_systems/geographic.hpp"

namespace specfem {
namespace coordinate_systems {

/**
 * @brief Configuration for UTM projection methods on coordinate classes.
 *
 * Forward and inverse Transverse Mercator projection on the WGS-84 ellipsoid,
 * following Snyder, "Map Projections -- A Working Manual" (USGS PP 1395, 1987).
 * Ported from `fortran/meshfem3d/shared/utm_geo.f90` (CAMx v6.10 origin).
 * Computations use double precision.
 *
 * The zone sign encodes the hemisphere: positive for north, negative for south.
 * The central meridian is computed as \f$ 6z - 183 \f$ degrees, where
 * \f$ z = |\text{zone}| \f$.
 *
 * @code
 * const geographic_coordinates geographic{ 2.674, 51.561, 0.0 };
 * const utm_projection_config config{ 31 };
 * const auto cartesian = geographic.to_cartesian(config);
 * const auto recovered = cartesian.to_geographic(config);
 *
 * // Southern hemisphere: use a negative zone.
 * const geographic_coordinates sydney{ 151.209, -33.869, 0.0 };
 * const auto south = sydney.to_cartesian(utm_projection_config{ -56 });
 *
 * // Suppression passes through x=longitude, y=latitude, z=depth.
 * const auto passthrough =
 *     geographic.to_cartesian(utm_projection_config{ 31, true });
 * @endcode
 */
struct utm_projection_config {
  int zone;              ///< UTM zone: +1..+60 (north), -1..-60 (south).
  bool suppress = false; ///< If true, coordinates pass through unchanged.
};

} // namespace coordinate_systems
} // namespace specfem
