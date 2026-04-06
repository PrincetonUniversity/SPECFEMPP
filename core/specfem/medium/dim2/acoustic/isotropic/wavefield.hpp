#pragma once

#include "specfem/algorithms.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/macros.hpp"
#include "specfem/medium_physics.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium_physics {

template <typename ChunkIndexType, typename DisplacementFieldType,
          typename VelocityFieldType, typename AccelerationFieldType,
          typename QuadratureType, typename WavefieldViewType>
KOKKOS_FUNCTION void impl_compute_wavefield(
    const std::integral_constant<specfem::element::dimension_tag,
                                 specfem::element::dimension_tag::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::acoustic>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const ChunkIndexType &chunk_index,
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
        &assembly,
    const QuadratureType &lagrange_derivative,
    const DisplacementFieldType &displacement,
    const VelocityFieldType &velocity,
    const AccelerationFieldType &acceleration,
    const specfem::enums::wavefield wavefield_type,
    WavefieldViewType wavefield) {

  using Tags =
      specfem::tags::Tags<specfem::element::dimension_tag::dim2,
                          specfem::element::medium_tag::acoustic,
                          specfem::element::property_tag::isotropic,
                          specfem::element::attenuation_tag::none, false>;

  using FieldDerivativesType = specfem::point::field_derivatives<Tags>;

  using PointPropertyType = specfem::point::properties<Tags>;

  const auto &properties = assembly.properties;

  const auto &active_field = [&]() {
    if (wavefield_type == specfem::enums::wavefield::displacement) {
      return displacement.get_data();
    } else if (wavefield_type == specfem::enums::wavefield::velocity) {
      return velocity.get_data();
    } else if (wavefield_type == specfem::enums::wavefield::acceleration) {
      return acceleration.get_data();
    } else if (wavefield_type == specfem::enums::wavefield::pressure) {
      return acceleration.get_data();
    } else {
      KOKKOS_ABORT_WITH_LOCATION(
          "Unsupported wavefield component for 2D acoustic isotropic media.");
    }
  }();

  if (wavefield_type == specfem::enums::wavefield::pressure) {
    specfem::execution::for_each_level(
        chunk_index.get_iterator(),
        [&](const typename ChunkIndexType::iterator_type::index_type
                &iterator_index) {
          const auto index = iterator_index.get_index();
          const int ielement = iterator_index.get_local_index().ispec;
          wavefield(ielement, index.iz, index.ix, 0) =
              -1.0 * active_field(ielement, index.iz, index.ix, 0);
        });

    return;
  }

  specfem::algorithms::gradient(
      chunk_index, assembly.jacobian_matrix, lagrange_derivative, active_field,
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index,
          const FieldDerivativesType::value_type &du) {
        const auto index = iterator_index.get_index();
        const int ielement = iterator_index.get_local_index().ispec;
        PointPropertyType point_property;

        specfem::assembly::load_on_device(index, properties, point_property);

        FieldDerivativesType point_field_derivatives(du);

        const auto point_stress =
            impl_compute_stress<Tags>(point_property, point_field_derivatives);

        wavefield(ielement, index.iz, index.ix, 0) = point_stress.T(0, 0);
        wavefield(ielement, index.iz, index.ix, 1) = point_stress.T(0, 1);
      });

  return;
}

} // namespace medium_physics
} // namespace specfem
