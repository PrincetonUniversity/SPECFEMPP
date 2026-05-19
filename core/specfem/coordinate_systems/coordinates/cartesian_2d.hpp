#pragma once

#include "specfem/coordinate_systems/coordinates.hpp"

namespace specfem {
namespace coordinate_systems {

/**
 * @brief 2D Cartesian coordinates (x, z) in meters.
 *
 * Trivial conversion to @ref specfem::point::global_coordinates — values
 * are copied directly.
 */
class cartesian_2d final
    : public coordinates<specfem::element::dimension_tag::dim2> {
public:
  double x; ///< meters
  double z; ///< meters

  cartesian_2d(double x, double z) : x(x), z(z) {}

  std::string print() const override;
};

} // namespace coordinate_systems
} // namespace specfem
