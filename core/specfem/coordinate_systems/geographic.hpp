#pragma once

#include "specfem/coordinate_systems/coordinates.hpp"
#include <string>

namespace specfem {
namespace coordinate_systems {

/**
 * @brief Geographic coordinates (decimal degrees + meters).
 *
 * Double precision required — single precision introduces ~1 m round-trip
 * error in UTM conversions.
 *
 * Conversion to @ref specfem::point::global_coordinates requires a UTM
 * projection configuration (regional) or geocentric conversion (global),
 * handled by @ref specfem::assembly::resolve_coordinates at assembly time.
 */
class geographic_coordinates final
    : public coordinates<specfem::element::dimension_tag::dim3> {
public:
  double longitude = 0.0; ///< degrees (negative for West)
  double latitude = 0.0;  ///< degrees
  double depth = 0.0;     ///< meters (positive down)

  /**
   * @brief Construct geographic coordinates.
   *
   * @param longitude Longitude in degrees (negative for West)
   * @param latitude Latitude in degrees
   * @param depth Depth in meters (positive down)
   */
  geographic_coordinates(double longitude, double latitude, double depth)
      : longitude(longitude), latitude(latitude), depth(depth) {}

  geographic_coordinates() = default;

  bool operator==(const coordinates<specfem::element::dimension_tag::dim3>
                      &other) const override;
  std::string print() const override;
};

} // namespace coordinate_systems
} // namespace specfem
