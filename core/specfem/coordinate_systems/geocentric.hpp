#pragma once

#include "specfem/coordinate_systems/coordinates.hpp"
#include <string>

namespace specfem {
namespace coordinate_systems {

/**
 * @brief Geocentric spherical coordinates (physics/SPECFEM convention).
 *
 * @f$ \theta @f$ is colatitude (0 at North Pole, @f$ \pi @f$ at South Pole).
 *
 * Conversion to @ref specfem::point::global_coordinates requires ellipticity
 * and topography corrections (Globe3D simulations). Not yet implemented.
 */
class geocentric_coordinates final
    : public coordinates<specfem::element::dimension_tag::dim3> {
public:
  double r;     ///< meters (radius from Earth center)
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

  bool operator==(const coordinates<specfem::element::dimension_tag::dim3>
                      &other) const override;
  std::string print() const override;
};

} // namespace coordinate_systems
} // namespace specfem
