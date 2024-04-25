#ifndef _DOMAIN_ELASTIC_ELEMENTS2D_ISOTROPIC_TPP
#define _DOMAIN_ELASTIC_ELEMENTS2D_ISOTROPIC_TPP

#include "compute/interface.hpp"
#include "domain/impl/elements/elastic/elastic2d_isotropic.hpp"
#include "domain/impl/elements/element.hpp"
#include "enumerations/interface.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "point/partial_derivatives.hpp"
#include "point/properties.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace {
template <int NGLL>
using StaticQuadraturePoints =
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>;

} // namespace

template <int NGLL, specfem::element::boundary_tag BC>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::property_tag::isotropic, BC,
    StaticQuadraturePoints<NGLL> >::
    compute_mass_matrix_component(
        const specfem::point::properties<medium_type::medium_tag,
                                         medium_type::property_tag> &properties,
        const specfem::point::partial_derivatives2<true> &partial_derivatives,
        specfem::kokkos::array_type<type_real, 2> &mass_matrix) const {

  constexpr int components = medium_type::components;

  static_assert(components == 2,
                "Number of components must be 2 for 2D isotropic elastic "
                "medium");

  mass_matrix[0] = properties.rho * partial_derivatives.jacobian;
  mass_matrix[1] = properties.rho * partial_derivatives.jacobian;

  return;
}

template <int NGLL, specfem::element::boundary_tag BC>
template <specfem::enums::time_scheme::type time_scheme>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::property_tag::isotropic, BC,
    StaticQuadraturePoints<NGLL> >::
    mass_time_contribution(
        const int &xz, const type_real &dt,
        const specfem::kokkos::array_type<type_real, dimension::dim> &weight,
        const specfem::point::partial_derivatives2<true> &partial_derivatives,
        const specfem::point::properties<medium_type::medium_tag,
                                         medium_type::property_tag> &properties,
        const specfem::point::boundary &boundary_type,
        specfem::kokkos::array_type<type_real, 2> &rmass_inverse) const {

  constexpr int components = medium_type::components;

  rmass_inverse[0] = 0.0;
  rmass_inverse[1] = 0.0;

  boundary_conditions.template mass_time_contribution<time_scheme>(
      xz, dt, weight, partial_derivatives, properties, boundary_type,
      rmass_inverse);

  return;
}

template <int NGLL, specfem::element::boundary_tag BC>
template <bool with_jacobian>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::property_tag::isotropic, BC,
    StaticQuadraturePoints<NGLL> >::
    compute_gradient(
        const int xz, const ElementQuadratureViewType s_hprime,
        const ElementFieldViewType u,
        const specfem::point::partial_derivatives2<with_jacobian>
            &partial_derivatives,
        const specfem::point::boundary &boundary_type,
        specfem::kokkos::array_type<type_real, medium_type::components> &dudxl,
        specfem::kokkos::array_type<type_real, medium_type::components> &dudzl)
        const {

  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);

  type_real du_dxi[medium_type::components] = { 0.0, 0.0 };
  type_real du_dgamma[medium_type::components] = { 0.0, 0.0 };

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
  for (int l = 0; l < NGLL; l++) {
    du_dxi[0] += s_hprime(ix, l) * u(iz, l, 0);
    du_dxi[1] += s_hprime(ix, l) * u(iz, l, 1);
    du_dgamma[0] += s_hprime(iz, l) * u(l, ix, 0);
    du_dgamma[1] += s_hprime(iz, l) * u(l, ix, 1);
  }
  // duxdx
  dudxl[0] = partial_derivatives.xix * du_dxi[0] +
             partial_derivatives.gammax * du_dgamma[0];

  // duxdz
  dudzl[0] = partial_derivatives.xiz * du_dxi[0] +
             partial_derivatives.gammaz * du_dgamma[0];

  // duzdx
  dudxl[1] = partial_derivatives.xix * du_dxi[1] +
             partial_derivatives.gammax * du_dgamma[1];

  // duzdz
  dudzl[1] = partial_derivatives.gammax * du_dxi[1] +
             partial_derivatives.gammaz * du_dgamma[1];

  boundary_conditions.enforce_gradient(xz, partial_derivatives, boundary_type,
                                       dudxl, dudzl);

  return;
}

template <int NGLL, specfem::element::boundary_tag BC>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::property_tag::isotropic, BC,
    StaticQuadraturePoints<NGLL> >::
    compute_stress(
        const int xz,
        const specfem::kokkos::array_type<type_real, medium_type::components>
            &dudxl,
        const specfem::kokkos::array_type<type_real, medium_type::components>
            &dudzl,
        const specfem::point::partial_derivatives2<true> &partial_derivatives,
        const specfem::point::properties<medium_type::medium_tag,
                                         medium_type::property_tag> &properties,
        const specfem::point::boundary &boundary_type,
        specfem::kokkos::array_type<type_real, medium_type::components>
            &stress_integrand_xi,
        specfem::kokkos::array_type<type_real, medium_type::components>
            &stress_integrand_gamma) const {

  type_real sigma_xx, sigma_zz, sigma_xz;

  if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
    // P_SV case
    // sigma_xx
    sigma_xx =
        properties.lambdaplus2mu * dudxl[0] + properties.lambda * dudzl[1];

    // sigma_zz
    sigma_zz =
        properties.lambdaplus2mu * dudzl[1] + properties.lambda * dudxl[0];

    // sigma_xz
    sigma_xz = properties.mu * (dudxl[1] + dudzl[0]);
  } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
    // SH-case
    // sigma_xx
    sigma_xx = properties.mu * dudxl[0]; // would be sigma_xy in
                                         // CPU-version

    // sigma_xz
    sigma_xz = properties.mu * dudzl[0]; // sigma_zy
  }

  stress_integrand_xi[0] =
      partial_derivatives.jacobian *
      (sigma_xx * partial_derivatives.xix + sigma_xz * partial_derivatives.xiz);
  stress_integrand_xi[1] =
      partial_derivatives.jacobian *
      (sigma_xz * partial_derivatives.xix + sigma_zz * partial_derivatives.xiz);
  stress_integrand_gamma[0] =
      partial_derivatives.jacobian * (sigma_xx * partial_derivatives.gammax +
                                      sigma_xz * partial_derivatives.gammaz);
  stress_integrand_gamma[1] =
      partial_derivatives.jacobian * (sigma_xz * partial_derivatives.gammax +
                                      sigma_zz * partial_derivatives.gammaz);

  boundary_conditions.enforce_stress(xz, partial_derivatives, properties,
                                     boundary_type, stress_integrand_xi,
                                     stress_integrand_gamma);

  return;
}

template <int NGLL, specfem::element::boundary_tag BC>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::property_tag::isotropic, BC,
    StaticQuadraturePoints<NGLL> >::
    compute_acceleration(
        const int &xz,
        const specfem::kokkos::array_type<type_real, dimension::dim> &weight,
        const ElementFieldViewType stress_integrand_xi,
        const ElementFieldViewType stress_integrand_gamma,
        const ElementQuadratureViewType s_hprimewgll,
        const specfem::point::partial_derivatives2<true> &partial_derivatives,
        const specfem::point::properties<medium_type::medium_tag,
                                         medium_type::property_tag> &properties,
        const specfem::point::boundary &boundary_type,
        const specfem::kokkos::array_type<type_real, medium_type::components>
            &velocity,
        specfem::kokkos::array_type<type_real, 2> &acceleration) const {

  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);
  type_real tempx1 = 0.0;
  type_real tempz1 = 0.0;
  type_real tempx3 = 0.0;
  type_real tempz3 = 0.0;

  constexpr int components = medium_type::components;

  static_assert(components == 2,
                "Number of components must be 2 for 2D isotropic elastic "
                "medium");

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
  for (int l = 0; l < NGLL; l++) {
    tempx1 += s_hprimewgll(ix, l) * stress_integrand_xi(iz, l, 0);
    tempz1 += s_hprimewgll(ix, l) * stress_integrand_xi(iz, l, 1);
    tempx3 += s_hprimewgll(iz, l) * stress_integrand_gamma(l, ix, 0);
    tempz3 += s_hprimewgll(iz, l) * stress_integrand_gamma(l, ix, 1);
  }

  acceleration[0] = -1.0 * (weight[1] * tempx1) - (weight[0] * tempx3);
  acceleration[1] = -1.0 * (weight[1] * tempz1) - (weight[0] * tempz3);

  boundary_conditions.enforce_traction(xz, weight, partial_derivatives,
                                       properties, boundary_type, velocity,
                                       acceleration);
}

#endif
