#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "point/partial_derivatives.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <typename T, typename PointPartialDerivativesType,
          typename PointStressIntegrandViewType, typename PointPropertiesType,
          typename PointAccelerationType>
KOKKOS_FUNCTION void impl_compute_couple_stress(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::elastic_psv_t>,
    const std::integral_constant<
        specfem::element::property_tag,
        specfem::element::property_tag::isotropic_cosserat>,
    const PointPartialDerivativesType &point_partial_derivatives,
    const PointPropertiesType &point_properties, const T factor,
    const PointStressIntegrandViewType &F,
    PointAccelerationType &acceleration) {

  const auto &xix = point_partial_derivatives.xix;
  const auto &xiz = point_partial_derivatives.xiz;
  const auto &gammax = point_partial_derivatives.gammax;
  const auto &gammaz = point_partial_derivatives.gammaz;

  // Matrix inversion factor
  const auto invf = static_cast<T>(1.0) / (xix * gammaz - xiz * gammax);

  // Invert matrix (double check!)
  const auto xxi = invf * gammaz;
  const auto zxi = -invf * xiz;
  const auto xgamma = -invf * gammax;
  const auto zgamma = invf * xix;

  // Get the stress tensor back
  // point_stress(0,0) = F(0,0) * xxi + F(0,1) * zxi;
  const auto t_xz = F(0, 0) * xgamma + F(0, 1) * zgamma;
  const auto t_zx = F(1, 0) * xxi + F(1, 1) * zxi;
  // point_stress(1,1) = F(1,0) * xgamma + F(1,1) * zgamma;

  // Add to acceleration t_{zx} - t_{xz}
  // Notes on spin
  acceleration.acceleration(3) += factor * (t_zx - t_xz);
};

} // namespace medium
} // namespace specfem
