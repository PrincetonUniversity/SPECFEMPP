#pragma once

#include "algorithms/dot.hpp"
#include "algorithms/gradient.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/macros.hpp"
#include "enumerations/medium.hpp"
#include "medium/compute_stress.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <typename ChunkIndexType, typename ChunkFieldType,
          typename QuadratureType, typename WavefieldViewType>
KOKKOS_FUNCTION void impl_compute_wavefield(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::acoustic>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const ChunkIndexType &chunk_index,
    const specfem::compute::assembly &assembly,
    const QuadratureType &quadrature, const ChunkFieldType &field,
    const specfem::wavefield::type wavefield_component,
    WavefieldViewType wavefield) {

  using FieldDerivativesType =
      specfem::point::field_derivatives<specfem::dimension::type::dim2,
                                        specfem::element::medium_tag::acoustic,
                                        false>;

  using PointPropertyType = specfem::point::properties<
      specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
      specfem::element::property_tag::isotropic, false>;

  const auto &properties = assembly.properties;

  const auto &active_field = [&]() {
    if (wavefield_component == specfem::wavefield::type::displacement) {
      return field.displacement;
    } else if (wavefield_component == specfem::wavefield::type::velocity) {
      return field.velocity;
    } else if (wavefield_component == specfem::wavefield::type::acceleration) {
      return field.acceleration;
    } else if (wavefield_component == specfem::wavefield::type::pressure) {
      return field.acceleration;
    } else {
      KOKKOS_ABORT_WITH_LOCATION("Unsupported wavefield component for 2D acoustic isotropic media.");
    }
  }();

  if (wavefield_component == specfem::wavefield::type::pressure) {
    specfem::execution::for_each_level(
        chunk_index.get_iterator(),
        [&](const typename ChunkIndexType::iterator_type::index_type
                &iterator_index) {
          const auto index = iterator_index.get_index();
          const int ielement = iterator_index.get_policy_index();
          wavefield(ielement, index.iz, index.ix, 0) =
              -1.0 * active_field(ielement, index.iz, index.ix, 0);
        });

    return;
  }

  specfem::algorithms::gradient(
      chunk_index, assembly.jacobian_matrix, quadrature.hprime_gll,
      active_field,
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index,
          const FieldDerivativesType::value_type &du) {
        const auto index = iterator_index.get_index();
        const int ielement = iterator_index.get_policy_index();
        PointPropertyType point_property;

        specfem::compute::load_on_device(index, properties, point_property);

        FieldDerivativesType point_field_derivatives(du);

        const auto point_stress =
            impl_compute_stress(point_property, point_field_derivatives);

        wavefield(ielement, index.iz, index.ix, 0) = point_stress.T(0, 0);
        wavefield(ielement, index.iz, index.ix, 1) = point_stress.T(0, 1);
      });

  return;
}

} // namespace medium
} // namespace specfem
