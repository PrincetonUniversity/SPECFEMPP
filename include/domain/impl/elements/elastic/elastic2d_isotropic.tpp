#ifndef _DOMAIN_ELASTIC_ELEMENTS2D_ISOTROPIC_TPP
#define _DOMAIN_ELASTIC_ELEMENTS2D_ISOTROPIC_TPP

#include "compute/interface.hpp"
#include "domain/impl/elements/elastic/elastic2d_isotropic.hpp"
#include "domain/impl/elements/element.hpp"
#include "enumerations/interface.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

template <int N, typename T>
using StaticScratchViewType =
    typename specfem::enums::element::quadrature::static_quadrature_points<
        N>::template ScratchViewType<T>;

// using field_type = Kokkos::Subview<
//     specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>, int,
//     std::remove_const_t<decltype(Kokkos::ALL)> >;

// -----------------------------------------------------------------------------
//                     SPECIALIZED ELEMENT
// -----------------------------------------------------------------------------
template <int NGLL, typename BC>
KOKKOS_FUNCTION specfem::domain::impl::elements::element<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic,
    BC>::element(const specfem::compute::partial_derivatives
                     &partial_derivatives,
                 const specfem::compute::properties &properties,
                 const specfem::compute::boundaries &boundary_conditions,
                 const quadrature_points_type &quadrature_points) {

#ifndef NDEBUG
  assert(partial_derivatives.xix.extent(1) == NGLL);
  assert(partial_derivatives.xix.extent(2) == NGLL);
  assert(partial_derivatives.gammax.extent(1) == NGLL);
  assert(partial_derivatives.gammax.extent(2) == NGLL);
  assert(partial_derivatives.xiz.extent(1) == NGLL);
  assert(partial_derivatives.xiz.extent(2) == NGLL);
  assert(partial_derivatives.gammaz.extent(1) == NGLL);
  assert(partial_derivatives.gammaz.extent(2) == NGLL);
  assert(partial_derivatives.jacobian.extent(1) == NGLL);
  assert(partial_derivatives.jacobian.extent(2) == NGLL);

  // Properties
  assert(properties.rho.extent(1) == NGLL);
  assert(properties.rho.extent(2) == NGLL);
  assert(properties.lambdaplus2mu.extent(1) == NGLL);
  assert(properties.lambdaplus2mu.extent(2) == NGLL);
  assert(properties.mu.extent(1) == NGLL);
  assert(properties.mu.extent(2) == NGLL);
#endif

  this->xix = partial_derivatives.xix;
  this->gammax = partial_derivatives.gammax;
  this->xiz = partial_derivatives.xiz;
  this->gammaz = partial_derivatives.gammaz;
  this->jacobian = partial_derivatives.jacobian;
  this->rho = properties.rho;
  this->lambdaplus2mu = properties.lambdaplus2mu;
  this->mu = properties.mu;

  this->boundary_conditions =
      boundary_conditions_type(boundary_conditions, quadrature_points);

  return;
}

template <int NGLL, typename BC>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic,
    BC>::compute_mass_matrix_component(const int &ispec, const int &xz,
                                       typename dimension::template array_type<
                                           type_real> &mass_matrix) const {
  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);

  constexpr int components = medium_type::components;

  static_assert(components == 2,
                "Number of components must be 2 for 2D isotropic elastic "
                "medium");

  mass_matrix[0] = this->rho(ispec, iz, ix) * this->jacobian(ispec, iz, ix);
  mass_matrix[1] = this->rho(ispec, iz, ix) * this->jacobian(ispec, iz, ix);

  return;
}

template <int NGLL, typename BC>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic, BC>::
    compute_gradient(
        const int &ispec, const int &ielement, const int &xz,
        const ScratchViewType<type_real, 1> s_hprime_xx,
        const ScratchViewType<type_real, 1> s_hprime_zz,
        const ScratchViewType<type_real, medium_type::components> u,
        typename dimension::template array_type<type_real> &dudxl,
        typename dimension::template array_type<type_real> &dudzl) const {

  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);

  const type_real xixl = this->xix(ispec, iz, ix);
  const type_real gammaxl = this->gammax(ispec, iz, ix);
  const type_real xizl = this->xiz(ispec, iz, ix);
  const type_real gammazl = this->gammaz(ispec, iz, ix);

  type_real du_dxi[medium_type::components] = { 0.0, 0.0 };
  type_real du_dgamma[medium_type::components] = { 0.0, 0.0 };

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
  for (int l = 0; l < NGLL; l++) {
    du_dxi[0] += s_hprime_xx(ix, l, 0) * u(iz, l, 0);
    du_dxi[1] += s_hprime_xx(ix, l, 0) * u(iz, l, 1);
    du_dgamma[0] += s_hprime_zz(iz, l, 0) * u(l, ix, 0);
    du_dgamma[1] += s_hprime_zz(iz, l, 0) * u(l, ix, 1);
  }
  // duxdx
  dudxl[0] = xixl * du_dxi[0] + gammaxl * du_dgamma[0];

  // duxdz
  dudzl[0] = xizl * du_dxi[0] + gammazl * du_dgamma[0];

  // duzdx
  dudxl[1] = xixl * du_dxi[1] + gammaxl * du_dgamma[1];

  // duzdz
  dudzl[1] = xizl * du_dxi[1] + gammazl * du_dgamma[1];

  boundary_conditions.enforce_gradient(ielement, xz, dudxl, dudzl);

  return;
}

template <int NGLL, typename BC>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic, BC>::
    compute_stress(
        const int &ispec, const int &ielement, const int &xz,
        const typename dimension::template array_type<type_real> &dudxl,
        const typename dimension::template array_type<type_real> &dudzl,
        typename dimension::template array_type<type_real> &stress_integrand_xi,
        typename dimension::template array_type<type_real>
            &stress_integrand_gamma) const {

  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);

  const type_real xixl = this->xix(ispec, iz, ix);
  const type_real gammaxl = this->gammax(ispec, iz, ix);
  const type_real xizl = this->xiz(ispec, iz, ix);
  const type_real gammazl = this->gammaz(ispec, iz, ix);
  const type_real jacobianl = this->jacobian(ispec, iz, ix);
  const type_real lambdaplus2mul = this->lambdaplus2mu(ispec, iz, ix);
  const type_real mul = this->mu(ispec, iz, ix);
  const type_real lambdal = lambdaplus2mul - 2.0 * mul;

  type_real sigma_xx, sigma_zz, sigma_xz;

  if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
    // P_SV case
    // sigma_xx
    sigma_xx = lambdaplus2mul * dudxl[0] + lambdal * dudzl[1];

    // sigma_zz
    sigma_zz = lambdaplus2mul * dudzl[1] + lambdal * dudxl[0];

    // sigma_xz
    sigma_xz = mul * (dudxl[1] + dudzl[0]);
  } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
    // SH-case
    // sigma_xx
    sigma_xx = mul * dudxl[0]; // would be sigma_xy in
                               // CPU-version

    // sigma_xz
    sigma_xz = mul * dudzl[0]; // sigma_zy
  }

  stress_integrand_xi[0] = jacobianl * (sigma_xx * xixl + sigma_xz * xizl);
  stress_integrand_xi[1] = jacobianl * (sigma_xz * xixl + sigma_zz * xizl);
  stress_integrand_gamma[0] =
      jacobianl * (sigma_xx * gammaxl + sigma_xz * gammazl);
  stress_integrand_gamma[1] =
      jacobianl * (sigma_xz * gammaxl + sigma_zz * gammazl);

  boundary_conditions.enforce_stress(ielement, xz, stress_integrand_xi,
                                     stress_integrand_gamma);

  return;
}

template <int NGLL, typename BC>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic, BC>::
    compute_acceleration(
        const int &ispec, const int &ielement, const int &xz,
        const type_real &wxglll, const type_real &wzglll,
        const ScratchViewType<type_real, medium_type::components>
            stress_integrand_xi,
        const ScratchViewType<type_real, medium_type::components>
            stress_integrand_gamma,
        const ScratchViewType<type_real, 1> s_hprimewgll_xx,
        const ScratchViewType<type_real, 1> s_hprimewgll_zz,
        typename dimension::template array_type<type_real> &acceleration)
        const {

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
    tempx1 += s_hprimewgll_xx(ix, l, 0) * stress_integrand_xi(iz, l, 0);
    tempz1 += s_hprimewgll_xx(ix, l, 0) * stress_integrand_xi(iz, l, 1);
    tempx3 += s_hprimewgll_zz(iz, l, 0) * stress_integrand_gamma(l, ix, 0);
    tempz3 += s_hprimewgll_zz(iz, l, 0) * stress_integrand_gamma(l, ix, 1);
  }

  acceleration[0] = -1.0 * (wzglll * tempx1) - (wxglll * tempx3);
  acceleration[1] = -1.0 * (wzglll * tempz1) - (wxglll * tempz3);

  boundary_conditions.enforce_traction(ielement, xz, acceleration);
}

#endif
