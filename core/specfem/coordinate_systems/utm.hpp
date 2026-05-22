#pragma once

/**
 * @brief UTM (Universal Transverse Mercator) projection conversions.
 *
 * Forward and inverse Transverse Mercator projection on the WGS-84 ellipsoid,
 * following Snyder, "Map Projections -- A Working Manual" (USGS PP 1395, 1987).
 * Ported from `fortran/meshfem3d/shared/utm_geo.f90` (CAMx v6.10 origin).
 *
 * All computations use double precision. The depth/z component is always
 * passed through unchanged.
 *
 * Example usage:
 * @code
 * #include "specfem/coordinate_systems.hpp"
 *
 * specfem::coordinate_systems::geographic_coordinates geo{ 2.674, 51.561,
 *                                                          0.0 };
 * specfem::coordinate_systems::utm_projection_config cfg{ 31 };
 *
 * // Forward: lon/lat -> UTM easting/northing
 * auto cart =
 *     specfem::coordinate_systems::transform<
 *         specfem::coordinate_systems::cartesian_coordinates>(geo, cfg);
 *
 * // Inverse: UTM -> lon/lat
 * auto recovered =
 *     specfem::coordinate_systems::transform<
 *         specfem::coordinate_systems::geographic_coordinates>(cart, cfg);
 *
 * // Southern hemisphere: use negative zone
 * auto cart_south =
 *     specfem::coordinate_systems::transform<
 *         specfem::coordinate_systems::cartesian_coordinates>(
 *         specfem::coordinate_systems::geographic_coordinates{
 *             151.209, -33.869, 0.0 },
 *         specfem::coordinate_systems::utm_projection_config{ -56 });
 *
 * // Suppress projection (pass-through: x=lon, y=lat, z=depth)
 * auto passthrough =
 *     specfem::coordinate_systems::transform<
 *         specfem::coordinate_systems::cartesian_coordinates>(
 *         geo, specfem::coordinate_systems::utm_projection_config{ 31, true });
 * @endcode
 */

#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/coordinate_systems/geographic.hpp"
#include "specfem/coordinate_systems/transform.hpp"

namespace specfem {
namespace coordinate_systems {

/**
 * @brief Configuration for a UTM projection.
 *
 * The zone sign encodes the hemisphere: positive for north, negative for south.
 * The central meridian is computed as @f$ 6z - 183 @f$ degrees, where
 * @f$ z = |\text{zone}| @f$.
 */
struct utm_projection_config {
  int zone;              ///< UTM zone: +1..+60 (north), -1..-60 (south)
  bool suppress = false; ///< If true, coordinates pass through unchanged
};

} // namespace coordinate_systems
} // namespace specfem

/**
 * @brief Geographic to cartesian via UTM forward projection.
 *
 * @param geo    Geographic coordinates (degrees, meters).
 * @param config Projection configuration (zone, suppress flag).
 * @return Cartesian coordinates (x=easting, y=northing, z=depth) in meters.
 */
template <>
specfem::coordinate_systems::cartesian_coordinates
specfem::coordinate_systems::transform<
    specfem::coordinate_systems::cartesian_coordinates,
    specfem::coordinate_systems::geographic_coordinates,
    specfem::coordinate_systems::utm_projection_config>(
    const specfem::coordinate_systems::geographic_coordinates &geo,
    const specfem::coordinate_systems::utm_projection_config &config);

/**
 * @brief Cartesian to geographic via UTM inverse projection.
 *
 * @param cart   Cartesian coordinates (meters).
 * @param config Projection configuration (zone, suppress flag).
 * @return Geographic coordinates (lon/lat in degrees, depth in meters).
 */
template <>
specfem::coordinate_systems::geographic_coordinates
specfem::coordinate_systems::transform<
    specfem::coordinate_systems::geographic_coordinates,
    specfem::coordinate_systems::cartesian_coordinates,
    specfem::coordinate_systems::utm_projection_config>(
    const specfem::coordinate_systems::cartesian_coordinates &cart,
    const specfem::coordinate_systems::utm_projection_config &config);
