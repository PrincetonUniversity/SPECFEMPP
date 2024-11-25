#pragma once

#include "algorithms/gradient.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "point/field_derivatives.hpp"
#include "point/properties.hpp"
#include "point/stress.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <bool UseSIMD>
KOKKOS_INLINE_FUNCTION specfem::point::stress<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    UseSIMD>
impl_compute_stress(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, UseSIMD> &properties,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        UseSIMD> &field_derivatives) {

  const auto &du = field_derivatives.du;

  specfem::datatype::VectorPointViewType<type_real, 2, 1, UseSIMD> T;

  T(0, 0) = properties.rho_inverse * du(0, 0);
  T(1, 0) = properties.rho_inverse * du(1, 0);

  return { T };
}

template <typename MemberType, typename IteratorType, typename ChunkFieldType,
          typename QuadratureType, typename WavefieldViewType>
KOKKOS_FUNCTION void impl_compute_wavefield(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::acoustic>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const MemberType &team, const IteratorType &iterator,
    const specfem::compute::assembly &assembly,
    const QuadratureType &quadrature, const ChunkFieldType &field,
    const specfem::wavefield::component wavefield_component,
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
    if (wavefield_component == specfem::wavefield::component::displacement) {
      return field.displacement;
    } else if (wavefield_component == specfem::wavefield::component::velocity) {
      return field.velocity;
    } else if (wavefield_component ==
               specfem::wavefield::component::acceleration) {
      return field.acceleration;
    } else if (wavefield_component == specfem::wavefield::component::pressure) {
      return field.acceleration;
    } else {
      Kokkos::abort("component not supported");
    }
  }();

  if (wavefield_component == specfem::wavefield::component::pressure) {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, iterator.chunk_size()),
                         [&](const int &i) {
                           const auto iterator_index = iterator(i);
                           const auto index = iterator_index.index;
                           wavefield(index.ispec, index.iz, index.ix, 0) =
                               -1.0 * active_field(iterator_index.ielement,
                                                   index.iz, index.ix, 0);
                         });

    return;
  }

  specfem::algorithms::gradient(
      team, iterator, assembly.partial_derivatives, quadrature.hprime_gll,
      active_field,
      [&](const typename IteratorType::index_type &iterator_index,
          const FieldDerivativesType::ViewType &du) {
        const auto &index = iterator_index.index;
        PointPropertyType point_property;

        specfem::compute::load_on_device(index, properties, point_property);

        FieldDerivativesType point_field_derivatives(du);

        const auto point_stress =
            impl_compute_stress(point_property, point_field_derivatives);

        wavefield(index.ispec, index.iz, index.ix, 0) = point_stress.T(0, 0);
        wavefield(index.ispec, index.iz, index.ix, 1) = point_stress.T(1, 0);
      });

  return;
}

} // namespace medium
} // namespace specfem
