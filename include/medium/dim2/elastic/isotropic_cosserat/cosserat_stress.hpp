#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <typename PointPropertiesType, typename PointDisplacementType,
          typename PointStressType>
KOKKOS_INLINE_FUNCTION void impl_compute_cosserat_stress(
    std::true_type,
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::elastic_psv_t>,
    const std::integral_constant<
        specfem::element::property_tag,
        specfem::element::property_tag::isotropic_cosserat>,
    const PointPropertiesType &properties,
    const PointDisplacementType &point_displacement,
    PointStressType &point_stress) {

  using ViewType = typename PointStressType::ViewType;

  // Stress and diplacement alias
  auto &T = point_stress.T;
  const auto &u = point_displacement.displacement;

  const typename ViewType::value_type factor =
      static_cast<type_real>(2.0) * properties.nu() * u(2);

  // sigma_xz = ... +  (2 \nu \phi_{y})
  T(0, 1) += factor;

  // sigma_zx = ... +  (-2 \nu \phi_{y})
  T(1, 0) -= factor;

  return;
};

} // namespace medium
} // namespace specfem
