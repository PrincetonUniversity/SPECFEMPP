#pragma once

#include "specfem/algorithms.hpp"
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
 * rotational degrees of freedom. Computes angular acceleration from
 * stress tensor asymmetry due to couple stress effects.
 *
 * **Moment equilibrium equation:**
 * \f$ j\ddot{\phi}_y = -(\sigma_{xz} - \sigma_{zx}) \f$
 *
 * **Coordinate transformation:**
 * \f$ \mathbf{J}^{-1} = \frac{1}{\det(\mathbf{J})} \begin{bmatrix} \gamma_z & -\xi_z \\ -\gamma_x & \xi_x \end{bmatrix} \f$
 *
 * where:
 * - \f$ j \f$: rotational inertia
 * - \f$ \phi_y \f$: rotation about y-axis
 * - \f$ \sigma_{xz} \neq \sigma_{zx} \f$: asymmetric stress tensor
 * - \f$ \mathbf{J} \f$: Jacobian transformation matrix
 *
 * @param point_jacobian_matrix Coordinate transformation matrix
 * @param point_properties Cosserat material properties
 * @param factor Integration scaling factor
 * @param F Stress integrand components in reference coordinates
 * @param acceleration[in,out] Acceleration field (rotational component modified)
 */
// clang-format on
template <typename T, typename PointJacobianMatrixType,
          typename PointStressIntegrandViewType, typename PointPropertiesType,
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
    const PointJacobianMatrixType &point_jacobian_matrix,
    const PointPropertiesType &point_properties, const T factor,
    const PointStressIntegrandViewType &F,
    PointAccelerationType &acceleration) {
  const auto jacobian_inv =
      specfem::algorithms::inverse(point_jacobian_matrix.tensor());
  const auto stress = F * jacobian_inv / point_jacobian_matrix.jacobian;
  // Reassign stress components due to transpose in its original definition
  const auto sigma_xz = stress(1, 0);
  const auto sigma_zx = stress(0, 1);

  // Add to acceleration
  acceleration(2) -= (sigma_xz - sigma_zx) * factor;
};

} // namespace medium_physics
} // namespace specfem
