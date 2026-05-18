#pragma once

namespace specfem {
namespace coordinate_systems {

/**
 * @brief Cartesian coordinates in meters.
 *
 * For UTM projections: x = easting, y = northing, z = depth (pass-through).
 */
struct cartesian_coordinates {
  double x; ///< meters (easting in UTM)
  double y; ///< meters (northing in UTM)
  double z; ///< meters (depth, passed through unchanged)
};

} // namespace coordinate_systems
} // namespace specfem
