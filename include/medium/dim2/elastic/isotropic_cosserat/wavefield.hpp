#pragma once

#include "algorithms/dot.hpp"
#include "algorithms/gradient.hpp"
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
                                 specfem::element::medium_tag::elastic_psv_t>,
    const std::integral_constant<
        specfem::element::property_tag,
        specfem::element::property_tag::isotropic_cosserat>,
    const ChunkIndexType &chunk_index,
    const specfem::assembly::assembly &assembly,
    const QuadratureType &quadrature, const ChunkFieldType &field,
    const specfem::wavefield::type wavefield_component,
    WavefieldViewType wavefield) {

  using FieldDerivativesType = specfem::point::field_derivatives<
      specfem::dimension::type::dim2,
      specfem::element::medium_tag::elastic_psv_t, false>;

  using PointPropertyType = specfem::point::properties<
      specfem::dimension::type::dim2,
      specfem::element::medium_tag::elastic_psv_t,
      specfem::element::property_tag::isotropic_cosserat, false>;

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
    } else if (wavefield_component == specfem::wavefield::type::rotation) {
      return field.displacement;
    } else {
      Kokkos::abort("component not supported");
    }
  }();

  if (wavefield_component == specfem::wavefield::type::pressure) {

    specfem::algorithms::gradient(
        chunk_index, assembly.jacobian_matrix, quadrature.hprime_gll,
        active_field,
        [&](const typename ChunkIndexType::iterator_type::index_type
                &iterator_index,
            const FieldDerivativesType::value_type &du) {
          const auto index = iterator_index.get_index();
          const int ielement = iterator_index.get_policy_index();
          PointPropertyType point_property;

          specfem::assembly::load_on_device(index, properties, point_property);

          /*
           * Here we compute the pressure wavefield from the elastic field:
           *
           * To compute pressure in 2D in an elastic solid in the plane strain
           * convention i.e. in the P-SV case, one uses
           *
           *   pressure = - trace(sigma) / 3
           *
           * but taking into account the fact that the off-plane strain
           * epsilon_zz is zero by definition of the plane strain convention but
           * thus the off-plane stress sigma_zz is not equal to zero, one has
           * instead:
           *
           *   sigma_zz = lambda * (epsilon_xx + epsilon_yy),
           *
           * thus:
           *
           *   sigma_ij = lambda delta_ij trace(epsilon) + 2 mu
           *
           * and
           *
           *   epsilon_ij = lambda (epsilon_xx + epsilon_yy) + 2 mu epsilon_ij
           *
           * From which follows:
           *
           *   sigma_xx = lambda (epsilon_xx + epsilon_yy) + 2 mu epsilon_xx
           *
           *   sigma_yy = lambda (epsilon_xx + epsilon_yy) + 2 mu epsilon_yy
           *
           *   sigma_zz = lambda * (epsilon_xx + epsilon_yy)
           *
           *   pressure = - trace(sigma) / 3
           *
           *            = - (lambda + 2*mu/3) (epsilon_xx + epsilon_yy)
           *
           * We do not store lambda, but kappa, which in 3D is defined as:
           *
           *  kappa = lambda + 2/3 mu
           *
           * In 2D, we have:
           *
           *  kappa = lambda + mu
           *
           * I will assume, since it is derived from the 3D in the plane-strain
           * formulation, that it is
           *
           *  kappa = lambda + 2/3 * mu
           *
           */
          wavefield(ielement, index.iz, index.ix, 0) =
              -1.0 * point_property.kappa() * (du(0, 0) + du(1, 1));
        });

    return;
  } else if (wavefield_component == specfem::wavefield::type::rotation) {
    specfem::execution::for_each_level(
        chunk_index.get_iterator(),
        [&](const typename ChunkIndexType::iterator_type::index_type
                &iterator_index) {
          const auto index = iterator_index.get_index();
          const int ielement = iterator_index.get_policy_index();

          // The rotational component of the
          wavefield(ielement, index.iz, index.ix, 0) =
              active_field(ielement, index.iz, index.ix, 2);
        });
    return;
  }

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.get_index();
        const int ielement = iterator_index.get_policy_index();
        wavefield(ielement, index.iz, index.ix, 0) =
            active_field(ielement, index.iz, index.ix, 0);
        wavefield(ielement, index.iz, index.ix, 1) =
            active_field(ielement, index.iz, index.ix, 1);
      });

  return;
}

} // namespace medium
} // namespace specfem
