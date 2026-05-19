#pragma once

#include "specfem/element/tags.hpp"
#include <string>

namespace specfem {
namespace coordinate_systems {

/**
 * @brief Abstract base class for coordinate representations.
 *
 * Concrete types hold raw coordinate data (cartesian, geographic, etc.)
 * and provide printing. Conversion to @ref specfem::point::global_coordinates
 * is handled by @ref specfem::assembly::resolve_coordinates, which lives in
 * the assembly layer and has access to the mesh.
 *
 * @tparam DimensionTag The dimension specification (dim2 or dim3)
 */
template <specfem::element::dimension_tag DimensionTag> class coordinates {
public:
  static constexpr auto dimension_tag = DimensionTag;

  virtual ~coordinates() = default;

  /**
   * @brief Human-readable description of the coordinate type and values.
   */
  virtual std::string print() const = 0;
};

} // namespace coordinate_systems
} // namespace specfem
