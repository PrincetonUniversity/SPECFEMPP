#pragma once

#include "algorithms/gradient.hpp"
#include "enumerations/macros.hpp"
#include "enumerations/medium.hpp"
#include "medium/compute_stress.hpp"
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
                                 specfem::element::medium_tag::poroelastic>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const ChunkIndexType &chunk_index,
    const specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
    const QuadratureType &quadrature, const DisplacementFieldType &displacement,
    const VelocityFieldType &velocity,
    const AccelerationFieldType &acceleration,
    const specfem::wavefield::type wavefield_type,
    WavefieldViewType wavefield) {

  using FieldDerivativesType = specfem::point::field_derivatives<
      specfem::dimension::type::dim2, specfem::element::medium_tag::poroelastic,
      false>;

  using PointPropertyType = specfem::point::properties<
      specfem::dimension::type::dim2, specfem::element::medium_tag::poroelastic,
      specfem::element::property_tag::isotropic, false>;

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
                                 "poroelastic isotropic media.");
    }
  }();

  if (wavefield_type == specfem::wavefield::type::pressure) {

    specfem::algorithms::gradient(
        chunk_index, assembly.jacobian_matrix, quadrature.hprime_gll,
        active_field,
        [&](const typename ChunkIndexType::iterator_type::index_type
                &iterator_index,
            const FieldDerivativesType::value_type &du) {
          const auto index = iterator_index.get_index();
          const int ielement = iterator_index.get_local_index().ispec;
          PointPropertyType point_property;

          specfem::assembly::load_on_device(index, properties, point_property);

          // Fluid pressure
          // sigmap = -pf = C_biot*(dux_dxl + duz_dzl) + M_biot*(dwx_dxl +
          // dwz_dzl)

          wavefield(ielement, index.iz, index.ix, 0) =
              -(point_property.C_Biot() * (du(0, 0) + du(1, 1)) +
                point_property.M_Biot() * (du(2, 0) + du(3, 1)));
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

} // namespace medium
} // namespace specfem
