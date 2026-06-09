#pragma once

#include "specfem/algorithms.hpp"
#include "specfem/element.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium_physics {

// clang-format off
/**
 * @defgroup specfem_cosserat_coupled_stress_computation_dim3_elastic_isotropic_cosserat
 *
 */

/**
  * @ingroup specfem_cosserat_coupled_stress_computation_dim3_elastic_isotropic_cosserat
 * @brief Compute couple stress contribution for 3D elastic isotropic Cosserat media.
 *
 * Implements moment equilibrium equation for micropolar continuum with
 * rotational degrees of freedom. Computes angular acceleration from
 * stress tensor asymmetry due to couple stress effects.
 *
 * **Moment equilibrium equation:**
 * \f$ j\ddot{\phi} = \nabla\cdot \sigma_c + \epsilon : \sigma \f$
 * \nabla\cdot \sigma_c is already handled by existing wavefield logic,
 * so we just need the adjustments given by
 * \f$ j\ddot{\phi}_x = (\sigma_{zy} - \sigma_{yz}) \cdot w_{iz} \cdot w_{iy} \cdot w_{ix} \cdot J \f$
 * \f$ j\ddot{\phi}_y = (\sigma_{xz} - \sigma_{zx}) \cdot w_{iz} \cdot w_{iy} \cdot w_{ix} \cdot J \f$
 * \f$ j\ddot{\phi}_z = (\sigma_{yx} - \sigma_{xy}) \cdot w_{iz} \cdot w_{iy} \cdot w_{ix} \cdot J \f$
 *
 * where:
 * - \f$ j \f$: rotational inertia
 * - \f$ \phi \f$: rotation vector (microrotation field)
 * - \f$ \sigma_{ij} \neq \sigma_{ji} \f$: asymmetric stress tensor
 *
 * @note The caller (epilogue) applies a global sign negation when reading from
 *       the scatter accumulator, so this function uses @c += (positive sign).
 *       The final contribution to the global acceleration field is
 *       @c -(sigma_a - sigma_b)*factor, matching the stiffness sign convention.
 *
 * @param point_properties Cosserat material properties
 * @param factor           Integration factor w(iz)*w(iy)*w(ix)*J for this GLL
 *                         point
 * @param point_stress     Physical stress tensor at this GLL point
 * @param acceleration[in,out] Acceleration delta (rotational components
 *                         modified)
 */
// clang-format on
template <typename T, typename PointPropertiesType, typename PointStressType,
          typename PointAccelerationType>
KOKKOS_INLINE_FUNCTION void impl_compute_cosserat_couple_stress(
    const std::true_type,
    const std::integral_constant<specfem::element::dimension_tag,
                                 specfem::element::dimension_tag::dim3>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::elastic_spin>,
    const std::integral_constant<
        specfem::element::property_tag,
        specfem::element::property_tag::isotropic_cosserat>,
    const PointPropertiesType &point_properties, const T factor,
    const PointStressType &point_stress, PointAccelerationType &acceleration) {

  // T(i,j) is the physical stress tensor (see stress.hpp layout)
  const auto sigma_xy = point_stress.T(1, 0);
  const auto sigma_yx = point_stress.T(0, 1);
  const auto sigma_xz = point_stress.T(2, 0);
  const auto sigma_zx = point_stress.T(0, 2);
  const auto sigma_yz = point_stress.T(2, 1);
  const auto sigma_zy = point_stress.T(1, 2);

  acceleration(3) += (sigma_zy - sigma_yz) * factor;
  acceleration(4) += (sigma_xz - sigma_zx) * factor;
  acceleration(5) += (sigma_yx - sigma_xy) * factor;
};

} // namespace medium_physics
} // namespace specfem
