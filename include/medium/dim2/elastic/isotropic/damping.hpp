#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <typename T, typename PointPropertiesType, typename PointVelocityType,
          typename PointAccelerationType>
KOKKOS_FUNCTION void impl_compute_damping_force(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::elastic_psv>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const T factor, const PointPropertiesType &point_properties,
    const PointVelocityType &velocity, PointAccelerationType &acceleration) {};

template <typename T, typename PointPropertiesType, typename PointVelocityType,
          typename PointAccelerationType>
KOKKOS_FUNCTION void impl_compute_damping_force(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::elastic_sh>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const T factor, const PointPropertiesType &point_properties,
    const PointVelocityType &velocity, PointAccelerationType &acceleration) {};

} // namespace medium
} // namespace specfem
