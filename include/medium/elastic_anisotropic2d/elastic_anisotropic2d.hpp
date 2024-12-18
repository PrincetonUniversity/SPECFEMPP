#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"
#include "globals.h"
#include "point/field_derivatives.hpp"
#include "point/properties.hpp"
#include "point/stress.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <bool UseSIMD>
KOKKOS_INLINE_FUNCTION specfem::point::stress<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    UseSIMD>
impl_compute_stress(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::anisotropic, UseSIMD> &properties,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        UseSIMD> &field_derivatives) {

  using datatype =
      typename specfem::datatype::simd<type_real, UseSIMD>::datatype;
  const auto &du = field_derivatives.du;

  datatype sigma_xx, sigma_zz, sigma_xz;

  // P_SV case
  // sigma_xx
  sigma_xx = properties.c11 * du(0, 0) + properties.c13 * du(1, 1) +
             properties.c15 * (du(1, 0) + du(0, 1));

  // sigma_zz
  sigma_zz = properties.c13 * du(0, 0) + properties.c33 * du(1, 1) +
             properties.c35 * (du(1, 0) + du(0, 1));

  // sigma_xz
  sigma_xz = properties.c15 * du(0, 0) + properties.c35 * du(1, 1) +
             properties.c55 * (du(1, 0) + du(0, 1));

  specfem::datatype::VectorPointViewType<type_real, 2, 2, UseSIMD> T;

  T(0, 0) = sigma_xx;
  T(0, 1) = sigma_xz;
  T(1, 0) = sigma_xz;
  T(1, 1) = sigma_zz;

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
                                 specfem::element::medium_tag::elastic>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::anisotropic>,
    const PointPropertiesType &properties,
    const AdjointPointFieldType &adjoint_field,
    const BackwardPointFieldType &backward_field,
    const PointFieldDerivativesType &adjoint_derivatives,
    const PointFieldDerivativesType &backward_derivatives,
    const type_real &dt) {

  static_assert(specfem::globals::simulation_wave == specfem::wave::p_sv ||
                    specfem::globals::simulation_wave == specfem::wave::sh,
                "Only P-SV and SH waves are supported.");

  if (specfem::globals::simulation_wave == specfem::wave::p_sv) {

    /*
    Note: Using # as adjoint modifier for the comments, so that `s` is the
    "standard" strainfield and `s#` is the adjoint strainfield. We use `ad_` as
    prefix for the adjoint wavefield and its derivatives, and `b_` as prefix
    for the "standard" backward wavefield and its derivatives.
    */

    // ad_dsxx = 0.5 * (ds#x/dx + ds#x/dx)
    const auto ad_dsxx = adjoint_derivatives.du(0, 0);

    // ad_dsxz = 0.5 * (ds#x/dz + ds#z/dx)
    const auto ad_dsxz =
        static_cast<type_real>(0.5) *
        (adjoint_derivatives.du(0, 1) + adjoint_derivatives.du(1, 0));

    // ad_dszz = 0.5 * (ds#z/dz + ds#z/dz)
    const auto ad_dszz = adjoint_derivatives.du(1, 1);

    // b_dsxx = 0.5 * (dsx/dx + dsx/dx) = dsx/dx
    const auto b_dsxx = backward_derivatives.du(0, 0);

    // b_dsxz = 0.5 * (dsx/dz + dsz/dx)
    const auto b_dsxz =
        static_cast<type_real>(0.5) *
        (backward_derivatives.du(0, 1) + backward_derivatives.du(1, 0));

    // b_dszz = 0.5 * (dsz/dz + dsz/dz) = dsz/dz
    const auto b_dszz = backward_derivatives.du(1, 1);

    // inner part of rho kernel equation 14
    // rho_kl = s#''_i * s_j
    auto rho_kl = specfem::algorithms::dot(adjoint_field.acceleration,
                                           backward_field.displacement);

    // Inner part of the 2-D version of Equation 15 in Tromp et al. 2005
    // That is \eps_{jk} \eps_{lm}
    auto c11_kl = ad_dsxx * b_dsxx;
    auto c13_kl = ad_dsxx * b_dszz + ad_dszz * b_dsxx;
    auto c15_kl = 2 * ad_dsxx * b_dsxz + ad_dsxz * b_dsxx;
    auto c33_kl = ad_dszz * b_dszz;
    auto c35_kl = 2 * b_dsxz * ad_dszz + ad_dsxz * b_dszz;
    auto c55_kl = 4 * ad_dsxz * b_dsxz;

    // Computing the rest of the integral.
    // rho from equation 14
    rho_kl = static_cast<type_real>(-1.0) * properties.rho * dt * rho_kl;
    c11_kl = static_cast<type_real>(-1.0) * c11_kl * properties.c11 * dt;
    c13_kl = static_cast<type_real>(-1.0) * c13_kl * properties.c13 * dt;
    c15_kl = static_cast<type_real>(-1.0) * c15_kl * properties.c15 * dt;
    c33_kl = static_cast<type_real>(-1.0) * c33_kl * properties.c33 * dt;
    c35_kl = static_cast<type_real>(-1.0) * c35_kl * properties.c35 * dt;
    c55_kl = static_cast<type_real>(-1.0) * c55_kl * properties.c55 * dt;

    return { rho_kl, c11_kl, c13_kl, c15_kl, c33_kl, c35_kl, c55_kl };

  } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
    /*
    SH (membrane) waves
    -------------------



    Note: Using # as adjoint modifier for the comments, so that `s` is the
    "standard" strainfield and `s#` is the adjoint strainfield. We use `ad_` as
    prefix for the adjoint wavefield and its derivatives, and `b_` as prefix for
    the "standard" backward wavefield and its derivatives.
    */

    // // ad_dsyx = 0.5 * (ds#y/dx + ds#x/dy) = 0.5 * (ds#y/dx)
    // const datatype ad_dsyx = static_cast<type_real>(0.5) *
    // adjoint_derivatives.du(0, 0);

    // // ad_dsyz = 0.5 * (ds#y/dz + ds#z/dy) = 0.5 * (ds#y/dz)
    // const datatype ad_dszz = static_cast<type_real>(0.5) *
    // adjoint_derivatives.du(1, 0);

    // // b_dsyx = 0.5 * (dsy/dx + dsx/dy) = 0.5 * dsy/dx
    // const datatype b_dsyx = static_cast<type_real>(0.5) *
    // backward_derivatives.du(0, 0);

    // // b_dsyz = 0.5 * (dsy/dz + dsz/dy) = 0.5 * dsz/dx
    // const datatype b_dsyz = static_cast<type_real>(0.5) *
    // backward_derivatives.du(1, 0));

    // // Inner part of the 2-D version of Equation 15 in Tromp et al. 2005
    // // That is \eps_{jk} \eps_{lm}
    // datatype c11_kl = 0; // ad_dsxx * b_dsxx
    // datatype c13_kl = ad_dsxx * b_dszz + ad_dszz * b_dsxx;
    // datatype c15_kl = 2 * ad_dsxx * b_dsxz + ad_dsxz * b_dsxx;
    // datatype c33_kl = ad_dszz * b_dszz;
    // datatype c35_kl = 2 * b_dsxz * ad_dszz + ad_dsxz * b_dszz;
    // datatype c55_kl = 4 * ad_dsxz * b_dsxz;

    // // Computing the rest of the integral.
    // // rho from equation 14
    // rho_kl = static_cast<type_real>(-1.0) * properties.rho * dt * rho_kl;
    // c11_kl = static_cast<type_real>(-1.0) * c11_kl * properties.c11 * dt;
    // c13_kl = static_cast<type_real>(-1.0) * c13_kl * properties.c13 * dt;
    // c15_kl = static_cast<type_real>(-1.0) * c15_kl * properties.c15 * dt;
    // c33_kl = static_cast<type_real>(-1.0) * c33_kl * properties.c33 * dt;
    // c35_kl = static_cast<type_real>(-1.0) * c35_kl * properties.c35 * dt;
    // c55_kl = static_cast<type_real>(-1.0) * c55_kl * properties.c55 * dt;

    /*
    I realized that we need the rest of the stiffness matrix for the SH wave,
    which is probably why anisotropic sh kernels aren't really supported in the
    speccfem2d fortran code. That would require a larger update to the
    anisotropic properties. I will leave this as a placeholder for now.
    Specifically, we need the following additional properties:
    - c44
    - c45/c54 (symmetric)
    */

    return { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  }
}

template <typename MemberType, typename IteratorType, typename ChunkFieldType,
          typename QuadratureType, typename WavefieldViewType>
KOKKOS_FUNCTION void impl_compute_wavefield(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::elastic>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::anisotropic>,
    const MemberType &team, const IteratorType &iterator,
    const specfem::compute::assembly &assembly,
    const QuadratureType &quadrature, const ChunkFieldType &field,
    const specfem::wavefield::type wavefield_component,
    WavefieldViewType wavefield) {

  using FieldDerivativesType =
      specfem::point::field_derivatives<specfem::dimension::type::dim2,
                                        specfem::element::medium_tag::elastic,
                                        false>;

  using PointPropertyType = specfem::point::properties<
      specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
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

          wavefield(index.ispec, index.iz, index.ix, 0) =
              -1.0 * (sigma_xx + sigma_zz + sigma_yy) / 3.0;
        });

    return;
  }

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, iterator.chunk_size()), [&](const int &i) {
        const auto iterator_index = iterator(i);
        const auto &index = iterator_index.index;
        wavefield(index.ispec, index.iz, index.ix, 0) =
            active_field(iterator_index.ielement, index.iz, index.ix, 0);
        wavefield(index.ispec, index.iz, index.ix, 1) =
            active_field(iterator_index.ielement, index.iz, index.ix, 1);
      });

  return;
}

} // namespace medium
} // namespace specfem
