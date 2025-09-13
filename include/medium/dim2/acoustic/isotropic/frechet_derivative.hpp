#pragma once

#include "enumerations/medium.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <typename PointPropertiesType, typename AdjointPointVelocityType,
          typename AdjointPointAccelerationType,
          typename BackwardPointDisplacementType,
          typename PointFieldDerivativesType>
KOKKOS_FUNCTION specfem::point::kernels<
    PointPropertiesType::dimension_tag, PointPropertiesType::medium_tag,
    PointPropertiesType::property_tag, PointPropertiesType::simd::using_simd>
impl_compute_frechet_derivatives(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::acoustic>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const PointPropertiesType &properties,
    const AdjointPointVelocityType &adjoint_velocity,
    const AdjointPointAccelerationType &adjoint_acceleration,
    const BackwardPointDisplacementType &backward_displacement,
    const PointFieldDerivativesType &adjoint_derivatives,
    const PointFieldDerivativesType &backward_derivatives,
    const type_real &dt) {

  const auto rho_kl =
      (adjoint_derivatives.du(0, 0) * backward_derivatives.du(0, 0) +
       adjoint_derivatives.du(0, 1) * backward_derivatives.du(0, 1)) *
      properties.rho_inverse() * dt;

  const auto kappa_kl =
      (adjoint_acceleration.get_data() * backward_displacement.get_data()) *
      static_cast<type_real>(1.0) / properties.kappa() * dt;

  return { rho_kl, kappa_kl };
}

} // namespace medium
} // namespace specfem
