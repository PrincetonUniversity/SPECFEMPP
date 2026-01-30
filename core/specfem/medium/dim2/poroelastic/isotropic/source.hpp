#pragma once

#include "enumerations/medium.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::medium {

/**
 * @defgroup specfem_medium_dim2_compute_source_contribution_poroelastic
 *
 */

/**
 * @ingroup specfem_medium_dim2_compute_source_contribution_poroelastic
 * @brief Compute source contribution for 2D poroelastic isotropic media.
 *
 * Implements coupled force source contribution for fluid-saturated porous
 * media using Biot's theory. Sources inject body forces that generate
 * coupled solid-fluid wave propagation.
 *
 * **Source equations:**
 * - \f$ \ddot{u}_x = S_{sx}(t) \cdot L_x(\mathbf{x}) \cdot \left(1 -
 * \frac{\phi}{\alpha}\right) \f$
 * - \f$ \ddot{u}_z = S_{sz}(t) \cdot L_z(\mathbf{x}) \cdot \left(1 -
 * \frac{\phi}{\alpha}\right) \f$
 * - \f$ \ddot{w}_x = S_{fx}(t) \cdot L_x(\mathbf{x}) \cdot \left(1 -
 * \frac{\rho_f}{\bar{\rho}}\right) \f$
 * - \f$ \ddot{w}_z = S_{fz}(t) \cdot L_z(\mathbf{x}) \cdot \left(1 -
 * \frac{\rho_f}{\bar{\rho}}\right) \f$
 *
 * where:
 * - \f$ u_x, u_z \f$: solid displacement components
 * - \f$ w_x, w_z \f$: fluid relative displacement components
 * - \f$ \phi \f$: porosity, \f$ \alpha \f$: tortuosity
 * - \f$ \rho_f \f$: fluid density, \f$ \rho_s \f$: solid density, \f$
 * \bar{\rho} \f$: bulk density defined as \f$ \bar{\rho} = \phi \rho_f + (1 -
 * \phi) \rho_s \f$
 *
 * @param point_source Source parameters (STF components, interpolants)
 * @param point_properties Poroelastic material properties
 * @return Acceleration contributions [\f$\ddot{u}_x, \ddot{u}_z, \ddot{w}_x,
 * \ddot{w}_z\f$]
 */
template <typename PointSourceType, typename PointPropertiesType>
KOKKOS_INLINE_FUNCTION auto impl_compute_source_contribution(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::poroelastic>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const PointSourceType &point_source,
    const PointPropertiesType &point_properties) {

  constexpr bool using_simd = PointPropertiesType::simd::using_simd;

  using PointAccelerationType =
      specfem::point::acceleration<specfem::dimension::type::dim2,
                                   specfem::element::medium_tag::poroelastic,
                                   using_simd>;

  PointAccelerationType result;

  result(0) = point_source.stf(0) * point_source.lagrange_interpolant(0) *
              (1.0 - point_properties.phi() / point_properties.tortuosity());
  result(1) = point_source.stf(1) * point_source.lagrange_interpolant(1) *
              (1.0 - point_properties.phi() / point_properties.tortuosity());
  result(2) = point_source.stf(2) * point_source.lagrange_interpolant(2) *
              (1.0 - point_properties.rho_f() / point_properties.rho_bar());
  result(3) = point_source.stf(3) * point_source.lagrange_interpolant(3) *
              (1.0 - point_properties.rho_f() / point_properties.rho_bar());

  return result;
}

} // namespace specfem::medium
