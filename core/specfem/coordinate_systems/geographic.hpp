#pragma once

#include "specfem/coordinate_systems/coordinates.hpp"
#include <string>

namespace specfem {
namespace coordinate_systems {

template <specfem::element::dimension_tag DimensionTag>
class cartesian_coordinates;
class geocentric_coordinates;
struct geocentric_projection_config;
struct utm_projection_config;

/**
 * @brief Geographic coordinates (decimal degrees + meters).
 *
 * Double precision required — single precision introduces ~1 m round-trip
 * error in UTM conversions.
 *
 * Conversion to @ref specfem::point::global_coordinates requires a UTM
 * projection configuration (regional) or geocentric conversion (global),
 * handled by @ref specfem::assembly::to at assembly time.
 */
class geographic_coordinates final
    : public coordinates<specfem::element::dimension_tag::dim3> {
public:
  double longitude; ///< degrees (negative for West)
  double latitude;  ///< degrees
  double depth;     ///< meters (positive down)

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

  /**
   * @brief Project longitude/latitude to UTM easting/northing.
   *
   * Depth is negated to z (passed through if projection is suppressed).
   * The result has no origin and still requires mesh/topographic resolution.
   *
   * @param config UTM zone, hemisphere, and projection suppression flag.
   * @return Cartesian coordinates in meters (degrees if suppressed).
   */
  cartesian_coordinates<specfem::element::dimension_tag::dim3>
  to_cartesian(const utm_projection_config &config) const;

  /**
   * @brief Convert to geocentric coordinates using a spherical reference.
   *
   * Colatitude is pi/2 minus latitude; radius is the reference surface radius
   * minus depth. Angles use globe's wrapping and polar nudging conventions.
   * Ellipticity and topographic resolution remain the caller's responsibility.
   *
   * @param config Reference surface radius in meters.
   * @return Geocentric coordinates (radius in meters, angles in radians).
   */
  geocentric_coordinates
  to_geocentric(const geocentric_projection_config &config) const;

  bool operator==(const coordinates<specfem::element::dimension_tag::dim3>
                      &other) const override;
  std::string print() const override;
};

} // namespace coordinate_systems
} // namespace specfem
