#pragma once

#include "algorithms/dot.hpp"
#include "algorithms/gradient.hpp"
#include "enumerations/medium.hpp"
#include "medium/compute_stress.hpp"
#include "point/field_derivatives.hpp"
#include "point/properties.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <typename MemberType, typename IteratorType, typename ChunkFieldType,
          typename QuadratureType, typename WavefieldViewType>
KOKKOS_FUNCTION void impl_compute_wavefield(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::elastic_sv>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::anisotropic>,
    const MemberType &team, const IteratorType &iterator,
    const specfem::compute::assembly &assembly,
    const QuadratureType &quadrature, const ChunkFieldType &field,
    const specfem::wavefield::type wavefield_component,
    WavefieldViewType wavefield) {

  using FieldDerivativesType = specfem::point::field_derivatives<
      specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_sv,
      false>;

  using PointPropertyType = specfem::point::properties<
      specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_sv,
      specfem::element::property_tag::anisotropic, false>;

  const auto &properties = assembly.properties;

  const auto &active_field = [&]() {
    if (wavefield_component == specfem::wavefield::type::displacement) {
      return field.displacement;
    } else if (wavefield_component == specfem::wavefield::type::velocity) {
      return field.velocity;
    } else if (wavefield_component == specfem::wavefield::type::acceleration) {
      return field.acceleration;
    } else if (wavefield_component == specfem::wavefield::type::pressure) {
      return field.displacement;
    } else {
      Kokkos::abort("component not supported");
    }
  }();

  if (wavefield_component == specfem::wavefield::type::pressure) {

    // over all elements and GLL points
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, iterator.chunk_size()),
        [&](const int &i) {
          const auto iterator_index = iterator(i);
          const auto &index = iterator_index.index;

          // Load the properties
          PointPropertyType point_property;
          specfem::compute::load_on_device(index, properties, point_property);

          // cannot compute pressure for an anisotropic material if c12 or c23
          // are zero
          if (point_property.c12 < 1.e-7 || point_property.c23 < 1.e-7) {
            Kokkos::abort("C_12 or C_23 are zero, cannot compute pressure. "
                          "Check your material properties. Or, deactivate the "
                          "pressure computation.");
          }
        });

    specfem::algorithms::gradient(
        team, iterator, assembly.partial_derivatives, quadrature.hprime_gll,
        active_field,
        [&](const typename IteratorType::index_type &iterator_index,
            const FieldDerivativesType::ViewType &du) {
          const auto &index = iterator_index.index;
          PointPropertyType point_property;

          specfem::compute::load_on_device(index, properties, point_property);

          // P_SV case
          // sigma_xx
          const auto sigma_xx = point_property.c11 * du(0, 0) +
                                point_property.c13 * du(1, 1) +
                                point_property.c15 * (du(1, 0) + du(0, 1));

          // sigma_zz
          const auto sigma_zz = point_property.c13 * du(0, 0) +
                                point_property.c33 * du(1, 1) +
                                point_property.c35 * (du(1, 0) + du(0, 1));

          // sigma_yy
          const auto sigma_yy = point_property.c12 * du(0, 0) +
                                point_property.c23 * du(1, 1) +
                                point_property.c25 * (du(1, 0) + du(0, 1));

          wavefield(iterator_index.ielement, index.iz, index.ix, 0) =
              -1.0 * (sigma_xx + sigma_zz + sigma_yy) / 3.0;
        });

    return;
  }

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, iterator.chunk_size()), [&](const int &i) {
        const auto iterator_index = iterator(i);
        const auto &index = iterator_index.index;
        wavefield(iterator_index.ielement, index.iz, index.ix, 0) =
            active_field(iterator_index.ielement, index.iz, index.ix, 0);
        wavefield(iterator_index.ielement, index.iz, index.ix, 1) =
            active_field(iterator_index.ielement, index.iz, index.ix, 1);
      });

  return;
}

} // namespace medium
} // namespace specfem
