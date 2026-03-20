#pragma once

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
 * \f$ j\ddot{\phi}_x = (\sigma_{zy} - \sigma_{yz}) \f$
 * \f$ j\ddot{\phi}_y = (\sigma_{xz} - \sigma_{zx}) \f$
 * \f$ j\ddot{\phi}_z = (\sigma_{yx} - \sigma_{xy}) \f$
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

  // TODO: figure out what these are called in 3D

  const auto &xix = point_jacobian_matrix.xix;
  const auto &xiy = point_jacobian_matrix.xiy;
  const auto &xiz = point_jacobian_matrix.xiz;
  const auto &etax = point_jacobian_matrix.etax;
  const auto &etay = point_jacobian_matrix.etay;
  const auto &etaz = point_jacobian_matrix.etaz;
  const auto &gammax = point_jacobian_matrix.gammax;
  const auto &gammay = point_jacobian_matrix.gammay;
  const auto &gammaz = point_jacobian_matrix.gammaz;
  const auto &jacobian = point_jacobian_matrix.jacobian;

  // Compute inverse Jacobian elements (standard 2x2 matrix inversion)
  const auto det = xix * (etay * gammaz - etaz * gammay) -
                   xiy * (etax * gammaz - etaz * gammax) +
                   xiz * (etax * gammay - etay * gammax);
  const auto invD = static_cast<T>(1.0) / det;

  // Standard 2x2 matrix inverse:
  //   J = [xix     xiy     xiz    ]
  //       [etax    etay    etaz   ]
  //       [gammax  gammay  gammaz ]
  // Then the inverse Jacobian matrix is:
  //   J^-1 = [∂x/∂ξ ∂x/∂η ∂x/∂γ]              [ (etay*gammaz-etaz*gammay)
  //   -(xiy*gammaz-xiz*gammay)   (xiy*etaz-xiz*etay) ]
  //          [∂y/∂ξ ∂y/∂η ∂y/∂γ]  = (1/det) * [ -(etax*gammaz-etaz*gammax)
  //          (xix*gammaz-xiz*gammax)  -(xix*etaz-xiz*etax) ] [∂z/∂ξ ∂z/∂η
  //          ∂z/∂γ]              [ (etax*gammay-etay*gammax)
  //          -(xix*gammay-xiy*gammax)   (xix*etay-xiy*etax) ]
  const auto xxi = (etay * gammaz - etaz * gammay) * invD;    // ∂x/∂ξ
  const auto xeta = -(etax * gammaz - etaz * gammax) * invD;  // ∂x/∂η
  const auto xgamma = (etax * gammay - etay * gammax) * invD; // ∂x/∂γ
  const auto yxi = -(xiy * gammaz - xiz * gammay) * invD;     // ∂y/∂ξ
  const auto yeta = (xix * gammaz - xiz * gammax) * invD;     // ∂y/∂η
  const auto ygamma = -(xix * gammay - xiy * gammax) * invD;  // ∂y/∂γ
  const auto zxi = (xiy * gammax - xiz * etay) * invD;        // ∂z/∂ξ
  const auto zeta = -(xix * gammax - xiz * etax) * invD;      // ∂z/∂η
  const auto zgamma = (xix * gammay - xiy * etax) * invD;     // ∂z/∂γ

  // Transform Stress integrand F to stress tensor T
  // const auto t_00 = (F(0, 0) * xxi + F(0, 1) * xgamma); // σ_xx
  const auto t_10 = F(1, 0) * xxi + F(1, 1) * xeta + F(1, 2) * xgamma; // σ_xy
  const auto t_20 = F(2, 0) * xxi + F(2, 1) * xeta + F(2, 2) * xgamma; // σ_xz
  const auto t_01 = F(0, 0) * yxi + F(0, 1) * yeta + F(0, 2) * ygamma; // σ_yx
  // const auto t_11 = (F(1, 0) * yxi + F(1, 1) * yeta + F(1, 2) * ygamma); //
  // σ_yy
  const auto t_21 = F(2, 0) * yxi + F(2, 1) * yeta + F(2, 2) * ygamma; // σ_yz
  const auto t_02 = F(0, 0) * zxi + F(0, 1) * zeta + F(0, 2) * zgamma; // σ_zx
  const auto t_12 = F(1, 0) * zxi + F(1, 1) * zeta + F(1, 2) * zgamma; // σ_zy
  // const auto t_22 = (F(2, 0) * zxi + F(2, 1) * zeta + F(2, 2) * zgamma); //
  // σ_zz

  // Reassign stress components due to transpose in its original definition
  const auto sigma_xy = t_10;
  const auto sigma_yx = t_01;
  const auto sigma_xz = t_20;
  const auto sigma_zx = t_02;
  const auto sigma_yz = t_21;
  const auto sigma_zy = t_12;

  // Add to acceleration
  acceleration(3) -= (sigma_zy - sigma_yz) * factor / jacobian;
  acceleration(4) -= (sigma_xz - sigma_zx) * factor / jacobian;
  acceleration(5) -= (sigma_yx - sigma_xy) * factor / jacobian;
};

} // namespace medium_physics
} // namespace specfem
