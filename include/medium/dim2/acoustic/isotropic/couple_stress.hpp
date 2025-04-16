#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
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
                                 specfem::element::medium_tag::acoustic>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const PointPartialDerivativesType &point_partial_derivatives,
    const PointPropertiesType &point_properties, const T factor,
    const PointStressIntegrandViewType &F,
    PointAccelerationType &acceleration) {};

} // namespace medium
} // namespace specfem
