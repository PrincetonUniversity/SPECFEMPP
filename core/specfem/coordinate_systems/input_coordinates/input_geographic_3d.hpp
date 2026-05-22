#pragma once

#include "specfem/coordinate_systems/geographic.hpp"
#include "specfem/coordinate_systems/input_coordinates.hpp"

namespace specfem {
namespace coordinate_systems {

/**
 * @brief 3D geographic input coordinates (longitude, latitude, depth).
 *
 * Wraps the existing @ref geographic_coordinates POD struct.
 * Conversion to @ref specfem::point::global_coordinates requires
 * UTM projection configuration and topography (available at assembly time).
 */
class input_geographic_3d final
    : public input_coordinates<specfem::element::dimension_tag::dim3> {
public:
  geographic_coordinates data; ///< {longitude(deg), latitude(deg), depth(m)}

  input_geographic_3d(double longitude, double latitude, double depth)
      : data{ longitude, latitude, depth } {}

  bool operator==(const input_coordinates<specfem::element::dimension_tag::dim3>
                      &other) const override;
  std::string print() const override;
};

} // namespace coordinate_systems
} // namespace specfem
