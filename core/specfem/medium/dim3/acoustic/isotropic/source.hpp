#pragma once

#include "specfem/element.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::medium_physics {

/**
 * @defgroup specfem_medium_dim3_compute_source_contribution_acoustic
 *
 */

/**
 * @ingroup specfem_medium_dim3_compute_source_contribution_acoustic
 * @brief Compute source contribution for 3D acoustic isotropic media.
 *
 * **Forward/backward source equation:**
 * \f$ \ddot{\chi} = -\frac{S(t) \cdot L(\mathbf{x})}{\kappa} \f$
 *
 * Negative sign ensures \f$ +S(t) \f$ produces \f$ +p \f$ (pressure =
 * \f$-\ddot{\chi}\f$).
 *
 * **Adjoint source equation:**
 * \f$ \ddot{\chi}^\dagger = S^\dagger(t) \cdot L(\mathbf{x}) \f$
 *
 * Adjoint sources are injected with positive sign and without \f$\kappa\f$
 * scaling (following Peter et al. eq. A-8 / Fortran SPECFEM3D convention):
 * the adjoint source file already incorporates the correct sign and the mass
 * matrix subsequently supplies the \f$\kappa\f$ factor.
 *
 * @tparam PointSourceType Point-wise source parameters (carries wavefield tag)
 * @tparam PointPropertiesType Point-wise material properties
 * @param point_source Source parameters (STF, interpolant)
 * @param point_properties Material properties (\f$ \kappa \f$)
 * @return Acceleration contribution for acoustic potential
 */
template <typename PointSourceType, typename PointPropertiesType>
KOKKOS_INLINE_FUNCTION auto impl_compute_source_contribution(
    const std::integral_constant<
        specfem::element::dimension_tag,
        specfem::element::dimension_tag::dim3> /*unused*/,
    const std::integral_constant<
        specfem::element::medium_tag,
        specfem::element::medium_tag::acoustic> /*unused*/,
    const std::integral_constant<
        specfem::element::property_tag,
        specfem::element::property_tag::isotropic> /*unused*/,
    const PointSourceType &point_source,
    const PointPropertiesType &point_properties) {
  constexpr bool using_simd = PointPropertiesType::simd::using_simd;

  using PointAccelerationType = specfem::point::acceleration<
      specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                          specfem::element::medium_tag::acoustic, using_simd>>;

  PointAccelerationType result;

  if constexpr (PointSourceType::wavefield_tag ==
                specfem::simulation::field_type::adjoint) {
    // Adjoint convention (SPECFEM3D Fortran: compute_add_sources_acoustic.f90,
    // line 245): inject as-is with positive sign and without 1/kappa.
    // The adjoint source file already encodes the correct sign from the
    // optimization problem (Peter et al. eq. A-8), and the mass-matrix
    // multiply (x kappa) that follows this call supplies the kappa factor.
    // Using the forward formula here would negate the adjoint field and
    // drop a factor of kappa, flipping the kernel sign and suppressing its
    // amplitude by ~kappa (~1e10 Pa for typical rock).
    result(0) = point_source.stf(0) * point_source.lagrange_interpolant(0);
  } else {
    // Forward/backward convention (SPECFEM3D Fortran:
    // compute_add_sources_acoustic.f90, line 125): divide by kappa so that
    // after the mass-matrix multiply (x kappa) the net contribution to
    // ddot_chi is -S*L. The negative sign makes pressure p = -ddot_chi
    // positive for a positive source time function.
    result(0) = -point_source.stf(0) * point_source.lagrange_interpolant(0) /
                point_properties.kappa();
  }

  return result;
}

} // namespace specfem::medium_physics
