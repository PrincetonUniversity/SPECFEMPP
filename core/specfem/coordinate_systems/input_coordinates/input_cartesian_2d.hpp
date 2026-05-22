#pragma once

#include "specfem/coordinate_systems/input_coordinates.hpp"

namespace specfem {
namespace coordinate_systems {

/**
 * @brief 2D Cartesian input coordinates (x, z) in meters.
 *
 * Trivial conversion to @ref specfem::point::global_coordinates — values
 * are copied directly.
 */
class input_cartesian_2d final
    : public input_coordinates<specfem::element::dimension_tag::dim2> {
public:
  double x; ///< meters
  double z; ///< meters

  input_cartesian_2d(double x, double z) : x(x), z(z) {}

  bool operator==(const input_coordinates<specfem::element::dimension_tag::dim2>
                      &other) const override;
  std::string print() const override;
};

} // namespace coordinate_systems
} // namespace specfem
