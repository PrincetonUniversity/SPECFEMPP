#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <typename T, typename PointJacobianMatrixType,
          typename PointStressIntegrandViewType, typename PointPropertiesType,
          typename PointAccelerationType>
KOKKOS_INLINE_FUNCTION void impl_compute_cosserat_couple_stress(
    const std::true_type,
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::elastic_psv_t>,
    const std::integral_constant<
        specfem::element::property_tag,
        specfem::element::property_tag::isotropic_cosserat>,
    const PointJacobianMatrixType &point_jacobian_matrix,
    const PointPropertiesType &point_properties, const T factor,
    const PointStressIntegrandViewType &F,
    PointAccelerationType &acceleration) {

  const auto &xix = point_jacobian_matrix.xix;
  const auto &xiz = point_jacobian_matrix.xiz;
  const auto &gammax = point_jacobian_matrix.gammax;
  const auto &gammaz = point_jacobian_matrix.gammaz;
  const auto &jacobian = point_jacobian_matrix.jacobian;

  // Compute inverse Jacobian elements (standard 2x2 matrix inversion)
  const auto det = xix * gammaz - xiz * gammax;
  const auto invD = static_cast<T>(1.0) / det;

  // Standard 2x2 matrix inverse: inv([a b; c d]) = (1/det) * [d -b; -c a]
  //   J = [xix     xiz    ]
  //       [gammax  gammaz ]
  // Then the inverse Jacobian matrix is:
  //   J^-1 = [∂x/∂ξ ∂x/∂γ]  = (1/det) * [gammaz  -xiz ]
  //          [∂z/∂ξ ∂z/∂γ]              [-gammax  xix ]
  const auto xxi = gammaz * invD;  // ∂x/∂ξ
  const auto xgamma = -xiz * invD; // ∂x/∂γ
  const auto zxi = -gammax * invD; // ∂z/∂ξ
  const auto zgamma = xix * invD;  // ∂z/∂γ

  // Transform Stress integrand F to stress tensor T
  // const auto t_00 = (F(0, 0) * xxi + F(0, 1) * xgamma); // σ_xx
  const auto t_10 = F(1, 0) * xxi + F(1, 1) * xgamma; // σ_xz
  const auto t_01 = F(0, 0) * zxi + F(0, 1) * zgamma; // σ_zx
  // const auto t_11 = (F(1, 0) * zxi + F(1, 1) * zgamma); // σ_zz

  // Reassign stress components due to transpose in its original definition
  const auto sigma_xz = t_10;
  const auto sigma_zx = t_01;

  // Add to acceleration
  acceleration(2) -= (sigma_xz - sigma_zx) * factor / jacobian;
};

} // namespace medium
} // namespace specfem
