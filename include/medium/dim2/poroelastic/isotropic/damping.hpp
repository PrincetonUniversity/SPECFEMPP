#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <typename T, typename PointPropertiesType, typename PointVelocityType,
          typename PointAccelerationType>
KOKKOS_FUNCTION void impl_compute_damping_force(
    const std::true_type,
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::poroelastic>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const T factor, const PointPropertiesType &point_properties,
    const PointVelocityType &velocity, PointAccelerationType &acceleration) {

  // viscous damping
  const auto viscx =
      point_properties.eta_f() * point_properties.inverse_permxx() *
          velocity.velocity(2) +
      point_properties.eta_f() * point_properties.inverse_permxz() *
          velocity.velocity(3);

  const auto viscz =
      point_properties.eta_f() * point_properties.inverse_permxz() *
          velocity.velocity(2) +
      point_properties.eta_f() * point_properties.inverse_permzz() *
          velocity.velocity(3);

  acceleration.acceleration(0) +=
      point_properties.phi() / point_properties.tortuosity() * viscx;
  acceleration.acceleration(1) +=
      point_properties.phi() / point_properties.tortuosity() * viscz;

  acceleration.acceleration(2) -= viscx;
  acceleration.acceleration(3) -= viscz;
}
} // namespace medium
} // namespace specfem
