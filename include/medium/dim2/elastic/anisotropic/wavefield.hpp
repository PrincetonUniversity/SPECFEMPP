#pragma once

#include "enumerations/medium.hpp"
#include "medium/compute_stress.hpp"
#include "specfem/algorithms.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <typename ChunkIndexType, typename DisplacementFieldType,
          typename VelocityFieldType, typename AccelerationFieldType,
          typename QuadratureType, typename WavefieldViewType>
KOKKOS_FUNCTION void impl_compute_wavefield(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::elastic_psv>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::anisotropic>,
    const ChunkIndexType &chunk_index,
    const specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
    const QuadratureType &lagrange_derivative,
    const DisplacementFieldType &displacement,
    const VelocityFieldType &velocity,
    const AccelerationFieldType &acceleration,
    const specfem::wavefield::type wavefield_type,
    WavefieldViewType wavefield) {

  using FieldDerivativesType = specfem::point::field_derivatives<
      specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_psv,
      false>;

  using PointPropertyType = specfem::point::properties<
      specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_psv,
      specfem::element::property_tag::anisotropic, false>;

  const auto &properties = assembly.properties;

  const auto &active_field = [&]() {
    if (wavefield_type == specfem::wavefield::type::displacement) {
      return displacement.get_data();
    } else if (wavefield_type == specfem::wavefield::type::velocity) {
      return velocity.get_data();
    } else if (wavefield_type == specfem::wavefield::type::acceleration) {
      return acceleration.get_data();
    } else if (wavefield_type == specfem::wavefield::type::pressure) {
      return displacement.get_data();
    } else {
      KOKKOS_ABORT_WITH_LOCATION("Unsupported wavefield component for 2D "
                                 "elastic anisotropic P-SV media.");
    }
  }();

  if (wavefield_type == specfem::wavefield::type::pressure) {

    specfem::algorithms::gradient(
        chunk_index, assembly.jacobian_matrix, lagrange_derivative,
        active_field,
        [&](const typename ChunkIndexType::iterator_type::index_type
                &iterator_index,
            const FieldDerivativesType::value_type &du) {
          const auto index = iterator_index.get_index();
          const int ielement = iterator_index.get_local_index().ispec;
          PointPropertyType point_property;

          specfem::assembly::load_on_device(index, properties, point_property);

          // cannot compute pressure for an anisotropic material if c12 or c23
          // are zero
          if (point_property.c12() < 1.e-7 || point_property.c23() < 1.e-7) {
            Kokkos::abort("C_12 or C_23 are zero, cannot compute pressure. "
                          "Check your material properties. Or, deactivate the "
                          "pressure computation.");
          }

          // P_SV case
          // sigma_xx
          const auto sigma_xx = point_property.c11() * du(0, 0) +
                                point_property.c13() * du(1, 1) +
                                point_property.c15() * (du(1, 0) + du(0, 1));

          // sigma_zz
          const auto sigma_zz = point_property.c13() * du(0, 0) +
                                point_property.c33() * du(1, 1) +
                                point_property.c35() * (du(1, 0) + du(0, 1));

          // sigma_yy
          const auto sigma_yy = point_property.c12() * du(0, 0) +
                                point_property.c23() * du(1, 1) +
                                point_property.c25() * (du(1, 0) + du(0, 1));

          wavefield(ielement, index.iz, index.ix, 0) =
              -1.0 * (sigma_xx + sigma_zz + sigma_yy) / 3.0;
        });

    return;
  }

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.get_index();
        const int ielement = iterator_index.get_local_index().ispec;
        wavefield(ielement, index.iz, index.ix, 0) =
            active_field(ielement, index.iz, index.ix, 0);
        wavefield(ielement, index.iz, index.ix, 1) =
            active_field(ielement, index.iz, index.ix, 1);
      });

  return;
}

template <typename ChunkIndexType, typename DisplacementFieldType,
          typename VelocityFieldType, typename AccelerationFieldType,
          typename QuadratureType, typename WavefieldViewType>
KOKKOS_FUNCTION void impl_compute_wavefield(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::elastic_sh>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::anisotropic>,
    const ChunkIndexType &chunk_index,
    const specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
    const QuadratureType &lagrange_derivative,
    const DisplacementFieldType &displacement,
    const VelocityFieldType &velocity,
    const AccelerationFieldType &acceleration,
    const specfem::wavefield::type wavefield_type,
    WavefieldViewType wavefield) {

  if (wavefield_type == specfem::wavefield::type::pressure) {
    Kokkos::abort("Pressure not supported for SH anisotropic media");

    return;
  }

  using FieldDerivativesType = specfem::point::field_derivatives<
      specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_sh,
      false>;

  const auto &active_field = [&]() {
    if (wavefield_type == specfem::wavefield::type::displacement) {
      return displacement.get_data();
    } else if (wavefield_type == specfem::wavefield::type::velocity) {
      return velocity.get_data();
    } else if (wavefield_type == specfem::wavefield::type::acceleration) {
      return acceleration.get_data();
    } else if (wavefield_type == specfem::wavefield::type::pressure) {
      return displacement.get_data();
    } else {
      KOKKOS_ABORT_WITH_LOCATION("Unsupported wavefield component for 2D "
                                 "elastic anisotropic SH media.");
    }
  }();

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.get_index();
        const int ielement = iterator_index.get_local_index().ispec;
        wavefield(ielement, index.iz, index.ix, 0) =
            active_field(ielement, index.iz, index.ix, 0);
      });

  return;
}

} // namespace medium
} // namespace specfem
