#pragma once

#include "specfem/element.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium_physics {

// clang-format off
/**
 * @defgroup specfem_cosserat_coupled_stress_computation_dim2_elastic_isotropic_cosserat
 *
 */

/**
  * @ingroup specfem_cosserat_coupled_stress_computation_dim2_elastic_isotropic_cosserat
 * @brief Compute couple stress contribution for 2D elastic isotropic Cosserat media.
 *
 * Implements moment equilibrium equation for micropolar continuum with
 * rotational degrees of freedom. Contributes to the spin component of the
 * acceleration scratch view from stress tensor asymmetry.
 *
 * **Moment equilibrium equation:**
 * \f$ j\ddot{\phi}_y = (\sigma_{xz} - \sigma_{zx}) \cdot w_{iz} \cdot w_{ix} \cdot J \f$
 *
 * @note The caller (epilogue) applies a global sign negation when reading from
 *       the scatter accumulator, so this function uses @c += (positive sign).
 *       The final contribution to the global acceleration field is
 *       @c -(sigma_xz - sigma_zx)*factor, matching the stiffness sign convention.
 *
 * @param point_properties Cosserat material properties
 * @param factor           Integration factor w(iz)*w(ix)*J for this GLL point
 * @param point_stress     Physical stress tensor at this GLL point
 * @param acceleration[in,out] Acceleration delta (rotational component modified)
 */
// clang-format on
template <typename T, typename PointPropertiesType, typename PointStressType,
          typename PointAccelerationType>
KOKKOS_INLINE_FUNCTION void impl_compute_cosserat_couple_stress(
    const std::true_type,
    const std::integral_constant<specfem::element::dimension_tag,
                                 specfem::element::dimension_tag::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::elastic_psv_t>,
    const std::integral_constant<
        specfem::element::property_tag,
        specfem::element::property_tag::isotropic_cosserat>,
    const PointPropertiesType &point_properties, const T factor,
    const PointStressType &point_stress, PointAccelerationType &acceleration) {

  // T(1,0) = sigma_xz,  T(0,1) = sigma_zx  (see stress.hpp layout)
  const auto sigma_xz = point_stress.T(1, 0);
  const auto sigma_zx = point_stress.T(0, 1);

  acceleration(2) += (sigma_xz - sigma_zx) * factor;
};

} // namespace medium_physics
} // namespace specfem
