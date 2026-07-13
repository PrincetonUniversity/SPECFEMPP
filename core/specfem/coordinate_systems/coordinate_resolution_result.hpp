#pragma once

#include "specfem/point/global_coordinates.hpp"

#include <optional>
#include <sstream>
#include <string>

namespace specfem {
namespace coordinate_systems {

/**
 * @brief Outcome of resolving a generic coordinate to mesh space.
 *
 * Carries the resolved global coordinate plus extras that only exist for
 * certain resolution paths. Designed to grow as more resolution-type-specific
 * information is surfaced.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 */
template <specfem::element::dimension_tag DimensionTag>
struct CoordinateResolutionResult {
  specfem::point::global_coordinates<DimensionTag> global; ///< Resolved global
                                                           ///< coordinates in
                                                           ///< mesh space
  std::optional<type_real> topography; ///< Surface elevation used to resolve a
                                       ///< depth-based input; nullopt when no
                                       ///< topographic lookup occurred

  /**
   * @brief Human-readable description of the resolved coordinate.
   *
   * @return The resolved global coordinates and, when a topographic lookup
   * occurred, the surface elevation that was used.
   */
  std::string print() const {
    std::ostringstream oss;
    oss << global.print() << ", resolved from input";
    if (topography.has_value())
      oss << ", topography: " << *topography << " m";
    return oss.str();
  }
};

} // namespace coordinate_systems
} // namespace specfem
