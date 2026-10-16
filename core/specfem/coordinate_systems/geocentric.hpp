#pragma once

#include "specfem/coordinate_systems/coordinates.hpp"
#include <string>

namespace specfem {
namespace coordinate_systems {

template <specfem::element::dimension_tag DimensionTag>
class cartesian_coordinates;
class geographic_coordinates;
struct geocentric_projection_config;

/**
 * @brief Geocentric spherical coordinates (physics/SPECFEM convention).
 *
 * @f$ \theta @f$ is colatitude (0 at North Pole, @f$ \pi @f$ at South Pole),
 * @f$ \phi @f$ is longitude (0 at the prime meridian). This matches globe's
 * angular conventions (`src/shared/rthetaphi_xyz.f90`).
 *
 * @f$ r @f$ is a physical radius in meters. The transforms to and from this
 * representation are exposed as named methods on the coordinate classes.
 * Geographic conversions require @ref geocentric_projection_config.
 */
class geocentric_coordinates final
    : public coordinates<specfem::element::dimension_tag::dim3> {
public:
  double r;     ///< meters (radius from the planet center)
  double theta; ///< radians (colatitude: 0 at North Pole, pi at South
                ///< Pole)
  double phi;   ///< radians (longitude: 0 at prime meridian)

  /**
   * @brief Construct geocentric coordinates.
   *
   * @param r Radius in meters
   * @param theta Colatitude in radians
   * @param phi Longitude in radians
   */
  geocentric_coordinates(double r, double theta, double phi)
      : r(r), theta(theta), phi(phi) {}

  geocentric_coordinates() = default;

  /**
   * @brief Convert to planet-centered Cartesian XYZ.
   *
   * \f$ x=r\sin\theta\cos\phi,\; y=r\sin\theta\sin\phi,
   * \; z=r\cos\theta \f$.
   *
   * @return Absolute Cartesian coordinates in meters with origin {0,0,0}.
   */
  cartesian_coordinates<specfem::element::dimension_tag::dim3>
  to_cartesian() const;

  /**
   * @brief Convert to geographic coordinates using a spherical reference.
   *
   * Latitude is 90 degrees minus colatitude; depth is the reference surface
   * radius minus r. Angles use globe's wrapping and polar nudging conventions.
   *
   * @param config Reference surface radius in meters.
   * @return Geographic coordinates (degrees, depth in meters positive down).
   */
  geographic_coordinates
  to_geographic(const geocentric_projection_config &config) const;

  bool operator==(const coordinates<specfem::element::dimension_tag::dim3>
                      &other) const override;
  std::string print() const override;
};

} // namespace coordinate_systems
} // namespace specfem
