#pragma once

#include "algorithms/dot.hpp"
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

template <typename PointPropertiesType, typename AdjointPointFieldType,
          typename BackwardPointFieldType, typename PointFieldDerivativesType>
KOKKOS_FUNCTION specfem::point::kernels<
    PointPropertiesType::dimension, PointPropertiesType::medium_tag,
    PointPropertiesType::property_tag, PointPropertiesType::simd::using_simd>
impl_compute_frechet_derivatives(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::acoustic>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const PointPropertiesType &properties,
    const AdjointPointFieldType &adjoint_field,
    const BackwardPointFieldType &backward_field,
    const PointFieldDerivativesType &adjoint_derivatives,
    const PointFieldDerivativesType &backward_derivatives,
    const type_real &dt) {

  const auto rho_kl =
      (adjoint_derivatives.du(0, 0) * backward_derivatives.du(0, 0) +
       adjoint_derivatives.du(1, 0) * backward_derivatives.du(1, 0)) *
      properties.rho_inverse * dt;

  const auto kappa_kl = specfem::algorithms::dot(adjoint_field.acceleration,
                                                 backward_field.displacement) *
                        static_cast<type_real>(1.0) / properties.kappa * dt;

  return { rho_kl, kappa_kl };
}

template <typename PointSourcesType, typename PointPropertiesType>
KOKKOS_INLINE_FUNCTION auto impl_compute_source_contribution(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::acoustic>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const PointSourcesType &point_source,
    const PointPropertiesType &point_properties) {

  using PointAccelerationType =
      specfem::point::field<PointPropertiesType::dimension,
                            PointPropertiesType::medium_tag, false, false, true,
                            false, PointPropertiesType::simd::using_simd>;

  PointAccelerationType result;

  result.acceleration(0) = point_source.stf(0) *
                           point_source.lagrange_interpolant(0) /
                           point_properties.kappa;

  return result;
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
      Kokkos::abort("component not supported");
    }
  }();

  if (wavefield_component == specfem::wavefield::type::pressure) {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, iterator.chunk_size()),
                         [&](const int &i) {
                           const auto iterator_index = iterator(i);
                           const auto index = iterator_index.index;
                           wavefield(iterator_index.ielement, index.iz,
                                     index.ix, 0) =
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

        wavefield(iterator_index.ielement, index.iz, index.ix, 0) =
            point_stress.T(0, 0);
        wavefield(iterator_index.ielement, index.iz, index.ix, 1) =
            point_stress.T(1, 0);
      });

  return;
}

} // namespace medium
} // namespace specfem
