#pragma once

#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/coordinate_systems/geocentric.hpp"
#include "specfem/coordinate_systems/geographic.hpp"
#include "specfem/coordinate_systems/transform.hpp"

namespace specfem {
namespace coordinate_systems {

/**
 * @brief Configuration for the geographic @f$ \leftrightarrow @f$ geocentric
 * transform (the geocentric analogue of @ref utm_projection_config).
 *
 * @ref r_planet is the reference surface radius supplied by the caller. In the
 * perfect-sphere case it is the planet radius
 * (@ref specfem::globe::PlanetConstants::r_planet); the elliptical case will
 * supply the deformed surface radius along the ray instead. Ellipticity and
 * topography are deliberately not represented here: they are a resolution
 * policy of the globe resolver, not transform math, which keeps this namespace
 * free of any dependency on `specfem::globe`.
 */
struct geocentric_projection_config {
  double r_planet; ///< reference surface radius in meters
};

} // namespace coordinate_systems
} // namespace specfem

/**
 * @brief Geocentric spherical to Cartesian (forward).
 *
 * @f$ x = r\sin\theta\cos\phi,\; y = r\sin\theta\sin\phi,\; z = r\cos\theta
 * @f$. The returned Cartesian coordinates are absolute (origin `{0,0,0}`): the
 * planet center is the Cartesian origin.
 *
 * @param geo Geocentric coordinates (r in meters, theta/phi in radians).
 * @return Cartesian coordinates (meters) with absolute origin.
 */
template <>
specfem::coordinate_systems::cartesian_coordinates<
    specfem::element::dimension_tag::dim3>
specfem::coordinate_systems::transform<
    specfem::coordinate_systems::cartesian_coordinates<
        specfem::element::dimension_tag::dim3>,
    specfem::coordinate_systems::geocentric_coordinates>(
    const specfem::coordinate_systems::geocentric_coordinates &geo);

/**
 * @brief Cartesian to geocentric spherical (inverse).
 *
 * @f$ r = \lVert v \rVert,\; \theta = \arccos(z/r) \in [0,\pi],\;
 * \phi = \operatorname{atan2}(y,x) @f$ normalized to @f$ [0, 2\pi) @f$ to match
 * globe's `reduce()` convention. @f$ \theta = 0 @f$ (and @f$ \phi = 0 @f$) when
 * @f$ r = 0 @f$.
 *
 * @param cart Cartesian coordinates (meters).
 * @return Geocentric coordinates (r in meters, theta/phi in radians).
 */
template <>
specfem::coordinate_systems::geocentric_coordinates
specfem::coordinate_systems::transform<
    specfem::coordinate_systems::geocentric_coordinates,
    specfem::coordinate_systems::cartesian_coordinates<
        specfem::element::dimension_tag::dim3>>(
    const specfem::coordinate_systems::cartesian_coordinates<
        specfem::element::dimension_tag::dim3> &cart);

/**
 * @brief Geographic to geocentric spherical (forward).
 *
 * Perfect-sphere conversion: the geographic latitude is used directly as the
 * geocentric colatitude (no @f$ (1-f)^2 @f$ flattening), and depth is measured
 * from the reference surface radius.
 * @f$ \theta = \pi/2 - \text{lat}\cdot\pi/180 @f$,
 * @f$ \phi = \text{lon}\cdot\pi/180 @f$ normalized to @f$ [0, 2\pi) @f$,
 * @f$ r = r_\text{planet} - \text{depth} @f$.
 *
 * @param geo Geographic coordinates (degrees, meters; depth positive down).
 * @param config Reference surface radius.
 * @return Geocentric coordinates (r in meters, theta/phi in radians).
 */
template <>
specfem::coordinate_systems::geocentric_coordinates
specfem::coordinate_systems::transform<
    specfem::coordinate_systems::geocentric_coordinates,
    specfem::coordinate_systems::geographic_coordinates,
    specfem::coordinate_systems::geocentric_projection_config>(
    const specfem::coordinate_systems::geographic_coordinates &geo,
    const specfem::coordinate_systems::geocentric_projection_config &config);

/**
 * @brief Geocentric spherical to geographic (inverse).
 *
 * @f$ \text{lat} = 90 - \theta\cdot180/\pi @f$,
 * @f$ \text{lon} = \phi\cdot180/\pi @f$,
 * @f$ \text{depth} = r_\text{planet} - r @f$.
 *
 * @param geo Geocentric coordinates (r in meters, theta/phi in radians).
 * @param config Reference surface radius.
 * @return Geographic coordinates (lon/lat in degrees, depth in meters).
 */
template <>
specfem::coordinate_systems::geographic_coordinates
specfem::coordinate_systems::transform<
    specfem::coordinate_systems::geographic_coordinates,
    specfem::coordinate_systems::geocentric_coordinates,
    specfem::coordinate_systems::geocentric_projection_config>(
    const specfem::coordinate_systems::geocentric_coordinates &geo,
    const specfem::coordinate_systems::geocentric_projection_config &config);
