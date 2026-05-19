#pragma once

#include "specfem/coordinate_systems/coordinates.hpp"
#include "specfem/coordinate_systems/geographic.hpp"

namespace specfem {
namespace coordinate_systems {

/**
 * @brief 3D geographic coordinates (longitude, latitude, depth).
 *
 * Wraps the existing @ref geographic_coordinates POD struct.
 * Conversion to @ref specfem::point::global_coordinates requires
 * UTM projection configuration and topography (available at assembly time).
 */
class geographic_3d final
    : public coordinates<specfem::element::dimension_tag::dim3> {
public:
  geographic_coordinates data; ///< {longitude(deg), latitude(deg), depth(m)}

  geographic_3d(double longitude, double latitude, double depth)
      : data{ longitude, latitude, depth } {}

  std::string print() const override;
};

} // namespace coordinate_systems
} // namespace specfem
