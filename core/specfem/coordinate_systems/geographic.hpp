#pragma once

namespace specfem {
namespace coordinate_systems {

/**
 * @brief Geographic coordinates (decimal degrees + meters).
 */
struct geographic_coordinates {
  double longitude; ///< degrees (negative for West)
  double latitude;  ///< degrees
  double depth;     ///< meters (positive down)
};

} // namespace coordinate_systems
} // namespace specfem
