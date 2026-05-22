#pragma once

#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/coordinate_systems/input_coordinates.hpp"

namespace specfem {
namespace coordinate_systems {

/**
 * @brief 3D Cartesian input coordinates (x, y, z) in meters.
 *
 * Wraps the existing @ref cartesian_coordinates POD struct.
 * Trivial conversion to @ref specfem::point::global_coordinates — values
 * are copied directly.
 */
class input_cartesian_3d final
    : public input_coordinates<specfem::element::dimension_tag::dim3> {
public:
  cartesian_coordinates data; ///< Underlying (x, y, z) in meters

  input_cartesian_3d(double x, double y, double z) : data{ x, y, z } {}

  bool operator==(const input_coordinates<specfem::element::dimension_tag::dim3>
                      &other) const override;
  std::string print() const override;
};

} // namespace coordinate_systems
} // namespace specfem
