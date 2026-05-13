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
 * \f$ j\ddot{\phi}_x = -(\sigma_{zy} - \sigma_{yz}) \f$
 * \f$ j\ddot{\phi}_y = -(\sigma_{xz} - \sigma_{zx}) \f$
 * \f$ j\ddot{\phi}_z = -(\sigma_{yx} - \sigma_{xy}) \f$
 *
 * **Coordinate transformation:**
 * If
 * \f$ \mathbf{J} = \begin{bmatrix} \xi_x & \xi_y & \xi_z \\ \eta_x & \eta_y & \eta_z \\ \gamma_x & \gamma_y & \gamma_z \end{bmatrix} \f$
 * then
 * \f$ \mathbf{J}^{-1} = \frac{1}{\det(\mathbf{J})}
 * \begin{bmatrix}
 * (\eta_y\gamma_z - \eta_z\gamma_y) & -(\xi_y\gamma_z - \xi_z\gamma_y) & (\xi_y\eta_z - \xi_z\eta_y) \\
 * -(\eta_x\gamma_z - \eta_z\gamma_x) & (\xi_x\gamma_z - \xi_z\gamma_x) & -(\xi_x\eta_z - \xi_z\eta_x) \\
 * (\eta_x\gamma_y - \eta_y\gamma_x) & -(\xi_x\gamma_y - \xi_y\gamma_x) & (\xi_x\eta_y - \xi_y\eta_x)
 * \end{bmatrix}
 * \f$
 *
 * where:
 * - \f$ j \f$: rotational inertia
 * - \f$ \phi \f$: rotation vector (microrotation field)
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
                                 specfem::element::dimension_tag::dim3>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::elastic_spin>,
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
  const auto sigma_xy = stress(1, 0);
  const auto sigma_yx = stress(0, 1);
  const auto sigma_xz = stress(2, 0);
  const auto sigma_zx = stress(0, 2);
  const auto sigma_yz = stress(2, 1);
  const auto sigma_zy = stress(1, 2);

  // Add to acceleration
  acceleration(3) -= (sigma_zy - sigma_yz) * factor;
  acceleration(4) -= (sigma_xz - sigma_zx) * factor;
  acceleration(5) -= (sigma_yx - sigma_xy) * factor;
};

} // namespace medium_physics
} // namespace specfem
