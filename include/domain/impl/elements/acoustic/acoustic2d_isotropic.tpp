#ifndef _DOMAIN_ACOUSTIC_ELEMENTS2D_ISOTROPIC_TPP
#define _DOMAIN_ACOUSTIC_ELEMENTS2D_ISOTROPIC_TPP

#include "compute/interface.hpp"
#include "domain/impl/elements/acoustic/acoustic2d_isotropic.hpp"
#include "domain/impl/elements/element.hpp"
#include "enumerations/interface.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace {
template <int NGLL>
using StaticQuadraturePoints =
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>;
} // namespace

template <int NGLL, specfem::element::boundary_tag BC>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::property_tag::isotropic, BC,
    StaticQuadraturePoints<NGLL> >::
    compute_mass_matrix_component(
        const specfem::point::properties<medium_type::medium_tag,
                                         medium_type::property_tag> &properties,
        const specfem::point::partial_derivatives2<true> &partial_derivatives,
        specfem::kokkos::array_type<type_real, medium_type::components>
            &mass_matrix) const {

  constexpr int components = medium_type::components;

  static_assert(components == 1, "Acoustic medium has only one component");

  mass_matrix[0] = partial_derivatives.jacobian / properties.kappa;

  return;
}

template <int NGLL, specfem::element::boundary_tag BC>
template <specfem::enums::time_scheme::type time_scheme>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::property_tag::isotropic, BC,
    StaticQuadraturePoints<NGLL> >::
    mass_time_contribution(
        const int &xz, const type_real &dt,
        const specfem::kokkos::array_type<type_real, dimension::dim> &weight,
        const specfem::point::partial_derivatives2<true> &partial_derivatives,
        const specfem::point::properties<medium_type::medium_tag,
                                         medium_type::property_tag> &properties,
        const specfem::point::boundary &boundary_type,
        specfem::kokkos::array_type<type_real, medium_type::components>
            &rmass_inverse) const {

  rmass_inverse[0] = 0.0;

  // comppute mass matrix component
  boundary_conditions.template mass_time_contribution<time_scheme>(
      xz, dt, weight, partial_derivatives, properties, boundary_type,
      rmass_inverse);

  return;
}

template <int NGLL, specfem::element::boundary_tag BC>
template <bool with_jacobian>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::property_tag::isotropic, BC,
    StaticQuadraturePoints<NGLL> >::
    compute_gradient(
        const int xz, const ElementQuadratureViewType s_hprime,
        const ElementFieldViewType field_chi,
        const specfem::point::partial_derivatives2<with_jacobian>
            &partial_derivatives,
        const specfem::point::boundary &boundary_type,
        specfem::kokkos::array_type<type_real, medium_type::components>
            &dchidxl,
        specfem::kokkos::array_type<type_real, medium_type::components>
            &dchidzl) const {

  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);

  type_real dchi_dxi = 0.0;
  type_real dchi_dgamma = 0.0;

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
  for (int l = 0; l < NGLL; l++) {
    dchi_dxi += s_hprime(ix, l) * field_chi(iz, l, 0);
    dchi_dgamma += s_hprime(iz, l) * field_chi(l, ix, 0);
  }

  // dchidx
  dchidxl[0] = dchi_dxi * partial_derivatives.xix +
               dchi_dgamma * partial_derivatives.gammax;

  // dchidz
  dchidzl[0] = dchi_dxi * partial_derivatives.xiz +
               dchi_dgamma * partial_derivatives.gammaz;

  boundary_conditions.enforce_gradient(xz, partial_derivatives, boundary_type,
                                       dchidxl, dchidzl);

  return;
}

template <int NGLL, specfem::element::boundary_tag BC>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::property_tag::isotropic, BC,
    StaticQuadraturePoints<NGLL> >::
    compute_stress(
        const int xz,
        const specfem::kokkos::array_type<type_real, medium_type::components>
            &dchidxl,
        const specfem::kokkos::array_type<type_real, medium_type::components>
            &dchidzl,
        const specfem::point::partial_derivatives2<true> &partial_derivatives,
        const specfem::point::properties<medium_type::medium_tag,
                                         medium_type::property_tag> &properties,
        const specfem::point::boundary &boundary_type,
        specfem::kokkos::array_type<type_real, medium_type::components>
            &stress_integrand_xi,
        specfem::kokkos::array_type<type_real, medium_type::components>
            &stress_integrand_gamma) const {

  // Precompute the factor
  type_real fac = partial_derivatives.jacobian * properties.rho_inverse;

  // Compute stress integrands 1 and 2
  // Here it is extremely important that this seems at odds with
  // equations (44) & (45) from Komatitsch and Tromp 2002 I. - Validation
  // The equations are however missing dxi/dx, dxi/dz, dzeta/dx, dzeta/dz
  // for the gradient of w^{\alpha\gamma}. In this->update_acceleration
  // the weights for the integration and the interpolated values for the
  // first derivatives of the lagrange polynomials are then collapsed
  stress_integrand_xi[0] = fac * (partial_derivatives.xix * dchidxl[0] +
                                  partial_derivatives.xiz * dchidzl[0]);
  stress_integrand_gamma[0] = fac * (partial_derivatives.gammax * dchidxl[0] +
                                     partial_derivatives.gammaz * dchidzl[0]);

  boundary_conditions.enforce_stress(xz, partial_derivatives, properties,
                                     boundary_type, stress_integrand_xi,
                                     stress_integrand_gamma);

  return;
}

template <int NGLL, specfem::element::boundary_tag BC>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
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
        specfem::kokkos::array_type<type_real, medium_type::components>
            &acceleration) const {

  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);
  type_real temp1l = 0.0;
  type_real temp2l = 0.0;

  constexpr int components = medium_type::components;

  static_assert(components == 1, "Acoustic medium has only one component");

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
  for (int l = 0; l < NGLL; l++) {
    temp1l += s_hprimewgll(ix, l) * stress_integrand_xi(iz, l, 0);
    temp2l += s_hprimewgll(iz, l) * stress_integrand_gamma(l, ix, 0);
  }

  acceleration[0] = -1.0 * ((weight[1] * temp1l) + (weight[0] * temp2l));

  boundary_conditions.enforce_traction(xz, weight, partial_derivatives,
                                       properties, boundary_type, velocity,
                                       acceleration);

  return;
}

#endif
