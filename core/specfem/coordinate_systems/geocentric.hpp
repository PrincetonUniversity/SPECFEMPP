#pragma once

#include "specfem/coordinate_systems/coordinates.hpp"
#include <array>
#include <string>

namespace specfem {
namespace coordinate_systems {

/**
 * @brief Geocentric spherical coordinates (physics/SPECFEM convention).
 *
 * @f$ \theta @f$ is colatitude (0 at North Pole, @f$ \pi @f$ at South Pole).
 *
 * Pure Cartesian conversion describes the supplied physical position; it does
 * not apply ellipticity or topography a second time. Source/receiver geographic
 * coordinate resolution is a separate operation.
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

  /**
   * @brief Convert physical Cartesian coordinates in metres to geocentric SI.
   * @param x Cartesian x in metres.
   * @param y Cartesian y in metres.
   * @param z Cartesian z in metres.
   * @return Radius in metres, colatitude in [0, pi], longitude in [0, 2*pi).
   *
   * Longitude is zero on the polar axis; both angles are zero at the origin.
   * Unlike globe's reduce(), exact zero angles are not perturbed by 1e-7:
   * atan2 already supplies a valid colatitude, and wrapping longitude suffices.
   * No pole perturbation is needed because this conversion divides by neither
   * radius nor sin(theta). This preserves Cartesian round trips at the axes.
   */
  static geocentric_coordinates from_cartesian(double x, double y, double z);

  /** @brief Return physical Cartesian coordinates in metres, in x/y/z order. */
  std::array<double, 3> to_cartesian() const;

  bool operator==(const coordinates<specfem::element::dimension_tag::dim3>
                      &other) const override;
  std::string print() const override;
};

} // namespace coordinate_systems
} // namespace specfem
