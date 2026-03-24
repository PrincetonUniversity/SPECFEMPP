#pragma once

#include "specfem/algorithms.hpp"
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
                                 specfem::element::dimension_tag::dim3>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::elastic_spin>,
    const std::integral_constant<
        specfem::element::property_tag,
        specfem::element::property_tag::isotropic_cosserat>,
    const ChunkIndexType &chunk_index,
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const QuadratureType &lagrange_derivative,
    const DisplacementFieldType &displacement,
    const VelocityFieldType &velocity,
    const AccelerationFieldType &acceleration,
    const specfem::enums::wavefield wavefield_type,
    WavefieldViewType wavefield) {

  using FieldDerivativesType = specfem::point::field_derivatives<
      specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                          specfem::element::medium_tag::elastic_spin, false> >;

  using PointPropertyType = specfem::point::properties<specfem::tags::Tags<
      specfem::element::dimension_tag::dim3,
      specfem::element::medium_tag::elastic_spin,
      specfem::element::property_tag::isotropic_cosserat, false> >;

  const auto &properties = assembly.properties;

  const auto &active_field = [&]() {
    if (wavefield_type == specfem::enums::wavefield::displacement) {
      return displacement.get_data();
    } else if (wavefield_type == specfem::enums::wavefield::velocity) {
      return velocity.get_data();
    } else if (wavefield_type == specfem::enums::wavefield::acceleration) {
      return acceleration.get_data();
    } else if (wavefield_type == specfem::enums::wavefield::pressure) {
      return displacement.get_data();
    } else if (wavefield_type == specfem::enums::wavefield::rotation) {
      return displacement.get_data();
    } else if (wavefield_type ==
               specfem::enums::wavefield::intrinsic_rotation) {
      return displacement.get_data();
    } else if (wavefield_type == specfem::enums::wavefield::curl) {
      return displacement.get_data();
    } else {
      KOKKOS_ABORT_WITH_LOCATION("Unsupported wavefield component for 3D "
                                 "elastic isotropic Cosserat media");
    }
  }();

  if (wavefield_type == specfem::enums::wavefield::pressure) {

    specfem::algorithms::gradient(
        chunk_index, assembly.jacobian_matrix, lagrange_derivative,
        active_field,
        [&](const typename ChunkIndexType::iterator_type::index_type
                &iterator_index,
            const FieldDerivativesType::value_type &du) {
          const auto index = iterator_index.get_index();
          const int ielement = iterator_index.get_policy_index();
          PointPropertyType point_property;

          specfem::assembly::load_on_device(index, properties, point_property);

          // // sigma_xx
          // const auto sigma_xx = point_property.lambdaplus2mu * du(0, 0) +
          //                       point_property.lambda * du(1, 1);
          //                       point_property.lambda * du(2, 2);

          // // sigma_yy
          // const auto sigma_yy = point_property.lambdaplus2mu * du(1, 1) +
          //                       point_property.lambda * du(0, 0);
          //                       point_property.lambda * du(2, 2);

          // // sigma_zz
          // const auto sigma_zz = point_property.lambdaplus2mu * du(2, 2) +
          //                       point_property.lambda * du(0, 0);
          //                       point_property.lambda * du(1, 1);

          // wavefield(iterator_index.ielement, index.iz, index.iy, index.ix, 0)
          // =
          //     -1.0 * (sigma_xx + sigma_zz + sigma_yy) / 3.0;
          wavefield(ielement, index.iz, index.iy, index.ix, 0) =
              -1.0 *
              ((point_property.lambda() + (2.0 / 3.0) * point_property.mu()) *
               (du(0, 0) + du(1, 1) + du(2, 2)));
        });

    return;
  } else if (wavefield_type == specfem::enums::wavefield::rotation) {
    specfem::execution::for_each_level(
        chunk_index.get_iterator(),
        [&](const typename ChunkIndexType::iterator_type::index_type
                &iterator_index) {
          const auto index = iterator_index.get_index();
          const int ielement = iterator_index.get_local_index().ispec;

          // The rotational component of the
          wavefield(ielement, index.iz, index.iy, index.ix, 0) =
              active_field(ielement, index.iz, index.iy, index.ix, 3);
          wavefield(ielement, index.iz, index.iy, index.ix, 1) =
              active_field(ielement, index.iz, index.iy, index.ix, 4);
          wavefield(ielement, index.iz, index.iy, index.ix, 2) =
              active_field(ielement, index.iz, index.iy, index.ix, 5);
        });
    return;

  } else if (wavefield_type == specfem::enums::wavefield::curl) {
    specfem::algorithms::gradient(
        chunk_index, assembly.jacobian_matrix, lagrange_derivative,
        active_field,
        [&](const typename ChunkIndexType::iterator_type::index_type
                &iterator_index,
            const FieldDerivativesType::value_type &du) {
          const auto index = iterator_index.get_index();
          const int ielement = iterator_index.get_local_index().ispec;

          // Here we compute the curl of the displacement field.
          wavefield(ielement, index.iz, index.iy, index.ix, 0) =
              du(2, 1) - du(1, 2);
          wavefield(ielement, index.iz, index.iy, index.ix, 1) =
              du(0, 2) - du(2, 0);
          wavefield(ielement, index.iz, index.iy, index.ix, 2) =
              du(1, 0) - du(0, 1);
        });

    return;
  } else if (wavefield_type == specfem::enums::wavefield::intrinsic_rotation) {
    specfem::algorithms::gradient(
        chunk_index, assembly.jacobian_matrix, lagrange_derivative,
        active_field,
        [&](const typename ChunkIndexType::iterator_type::index_type
                &iterator_index,
            const FieldDerivativesType::value_type &du) {
          const auto index = iterator_index.get_index();
          const int ielement = iterator_index.get_local_index().ispec;

          // Here we compute the intrinsic rotation wavefield from the
          // rotation field and the curl of the displacement field.
          wavefield(ielement, index.iz, index.iy, index.ix, 0) =
              active_field(ielement, index.iz, index.iy, index.ix, 3) -
              static_cast<type_real>(0.5) * du(2, 1) - du(1, 2);
          wavefield(ielement, index.iz, index.iy, index.ix, 1) =
              active_field(ielement, index.iz, index.iy, index.ix, 4) -
              static_cast<type_real>(0.5) * (du(0, 2) - du(2, 0));
          wavefield(ielement, index.iz, index.iy, index.ix, 2) =
              active_field(ielement, index.iz, index.iy, index.ix, 5) -
              static_cast<type_real>(0.5) * (du(1, 0) - du(0, 1));
        });

    return;
  }

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.get_index();
        const int ielement = iterator_index.get_local_index().ispec;
        wavefield(ielement, index.iz, index.iy, index.ix, 0) =
            active_field(ielement, index.iz, index.iy, index.ix, 0);
        wavefield(ielement, index.iz, index.iy, index.ix, 1) =
            active_field(ielement, index.iz, index.iy, index.ix, 1);
        wavefield(ielement, index.iz, index.iy, index.ix, 2) =
            active_field(ielement, index.iz, index.iy, index.ix, 2);
      });

  return;
}

} // namespace medium_physics
} // namespace specfem
