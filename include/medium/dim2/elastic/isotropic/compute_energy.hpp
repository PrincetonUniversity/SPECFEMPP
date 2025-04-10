#pragma once

#include "algorithms/gradient.hpp"
#include "compute/assembly/assembly.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"
#include "medium/compute_stress.hpp"
#include "point/field.hpp"
#include "point/field_derivatives.hpp"
#include "point/properties.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {
template <specfem::wavefield::simulation_field Wavefield,
          specfem::element::property_tag PropertyTag, typename MemberType,
          typename IteratorType, typename QuadratureType,
          typename ChunkFieldType, typename CallbackFunctor>
KOKKOS_INLINE_FUNCTION void impl_compute_energy(
    std::integral_constant<specfem::dimension::type,
                           specfem::dimension::type::dim2>,
    std::integral_constant<specfem::element::medium_tag,
                           specfem::element::medium_tag::elastic_psv>,
    std::integral_constant<specfem::element::property_tag, PropertyTag>,
    const MemberType &team, const IteratorType &iterator,
    const specfem::compute::assembly &assembly,
    const QuadratureType &quadrature, const ChunkFieldType &element_field,
    CallbackFunctor callback) {

  constexpr int using_simd = ChunkFieldType::simd::using_simd;
  const auto &active_field = element_field.displacement;

  const auto &properties = assembly.properties;
  const auto &field = assembly.fields.get_simulation_field<Wavefield>();

  using PointPropertyType =
      specfem::point::properties<specfem::dimension::type::dim2,
                                 specfem::element::medium_tag::elastic_psv,
                                 PropertyTag, using_simd>;

  using FieldDerivativesType = specfem::point::field_derivatives<
      specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_psv,
      using_simd>;

  using PointFieldType =
      specfem::point::field<specfem::dimension::type::dim2,
                            specfem::element::medium_tag::elastic_psv, false,
                            true, false, false, using_simd>;

  specfem::algorithms::gradient(
      team, iterator, assembly.partial_derivatives, quadrature.hprime_gll,
      active_field,
      [&](const typename IteratorType::index_type &iterator_index,
          const typename FieldDerivativesType::ViewType &du) {
        const auto index = iterator_index.index;
        FieldDerivativesType field_derivatives(du);

        PointPropertyType point_property;
        specfem::compute::load_on_device(index, properties, point_property);

        PointFieldType point_field;
        specfem::compute::load_on_device(index, field, point_field);

        const auto stress =
            specfem::medium::compute_stress(point_property, field_derivatives);

        const auto vn2 = point_field.velocity(0) * point_field.velocity(0) +
                         point_field.velocity(1) * point_field.velocity(1);

        const auto kinetic_energy =
            static_cast<type_real>(0.5) * point_property.rho() * vn2;

        const auto potential_energy =
            static_cast<type_real>(0.5) *
            (stress.T(0, 0) * du(0, 0) + stress.T(1, 1) * du(1, 1) +
             static_cast<type_real>(2.0) * stress.T(0, 1) *
                 (du(0, 1) + du(1, 0)));
        const auto total_energy = kinetic_energy + potential_energy;

        callback(iterator_index, total_energy);
      });
}

template <specfem::wavefield::simulation_field Wavefield,
          specfem::element::property_tag PropertyTag, typename MemberType,
          typename IteratorType, typename QuadratureType,
          typename ChunkFieldType, typename CallbackFunctor>
KOKKOS_INLINE_FUNCTION void impl_compute_energy(
    std::integral_constant<specfem::dimension::type,
                           specfem::dimension::type::dim2>,
    std::integral_constant<specfem::element::medium_tag,
                           specfem::element::medium_tag::elastic_sh>,
    std::integral_constant<specfem::element::property_tag, PropertyTag>,
    const MemberType &team, const IteratorType &iterator,
    const specfem::compute::assembly &assembly,
    const QuadratureType &quadrature, const ChunkFieldType &element_field,
    CallbackFunctor callback) {

  constexpr int using_simd = ChunkFieldType::simd::using_simd;
  const auto &active_field = element_field.displacement;

  const auto &properties = assembly.properties;
  const auto &field = assembly.fields.get_simulation_field<Wavefield>();

  using PointPropertyType =
      specfem::point::properties<specfem::dimension::type::dim2,
                                 specfem::element::medium_tag::elastic_sh,
                                 PropertyTag, using_simd>;

  using FieldDerivativesType = specfem::point::field_derivatives<
      specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_sh,
      using_simd>;

  using PointFieldType =
      specfem::point::field<specfem::dimension::type::dim2,
                            specfem::element::medium_tag::elastic_sh, false,
                            true, false, false, using_simd>;

  specfem::algorithms::gradient(
      team, iterator, assembly.partial_derivatives, quadrature.hprime_gll,
      active_field,
      [&](const typename IteratorType::index_type &iterator_index,
          const typename FieldDerivativesType::ViewType &du) {
        const auto index = iterator_index.index;
        FieldDerivativesType field_derivatives(du);

        PointPropertyType point_property;
        specfem::compute::load_on_device(index, properties, point_property);

        PointFieldType point_field;
        specfem::compute::load_on_device(index, field, point_field);

        const auto stress =
            specfem::medium::compute_stress(point_property, field_derivatives);

        const auto vn2 = point_field.velocity(0) * point_field.velocity(0);

        const auto kinetic_energy =
            static_cast<type_real>(0.5) * point_property.rho() * vn2;

        const auto potential_energy =
            static_cast<type_real>(0.5) *
            (stress.T(0, 0) * du(0, 0) + stress.T(1, 0) * du(1, 0));
        const auto total_energy = kinetic_energy + potential_energy;

        callback(iterator_index, total_energy);
      });
}
} // namespace medium
} // namespace specfem
