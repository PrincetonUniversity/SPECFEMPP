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
template <specfem::element::dimension_tag DimensionTag>
class input_coordinates {
public:
  static constexpr auto dimension_tag = DimensionTag;

  virtual ~input_coordinates() = default;

  /**
   * @brief Equality comparison using is_close on all members.
   *
   * Returns false if the concrete types differ.
   */
  virtual bool operator==(const input_coordinates &other) const = 0;

  bool operator!=(const input_coordinates &other) const {
    return !(*this == other);
  }

  /**
   * @brief Human-readable description of the coordinate type and values.
   */
  virtual std::string print() const = 0;
};

} // namespace coordinate_systems
} // namespace specfem
