#ifndef _DOMAIN_ELASTIC_ELEMENTS2D_ISOTROPIC_TPP
#define _DOMAIN_ELASTIC_ELEMENTS2D_ISOTROPIC_TPP

#include "compute/interface.hpp"
#include "domain/impl/elements/elastic/elastic2d_isotropic.hpp"
#include "domain/impl/elements/element.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "specfem_enums.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

template <int N, typename T>
using StaticScratchViewType =
    typename specfem::enums::element::quadrature::static_quadrature_points<
        N>::template ScratchViewType<T>;

using field_type = Kokkos::Subview<
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>, int,
    std::remove_const_t<decltype(Kokkos::ALL)> >;

// -----------------------------------------------------------------------------
//                     SPECIALIZED ELEMENT
// -----------------------------------------------------------------------------
template <int NGLL>
KOKKOS_FUNCTION specfem::domain::impl::elements::element<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>::
    element(const int ispec,
            const specfem::compute::partial_derivatives partial_derivatives,
            const specfem::compute::properties properties)
    : ispec(ispec) {

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

  this->xix = Kokkos::subview(partial_derivatives.xix, ispec, Kokkos::ALL(),
                              Kokkos::ALL());
  this->gammax = Kokkos::subview(partial_derivatives.gammax, ispec,
                                 Kokkos::ALL(), Kokkos::ALL());
  this->xiz = Kokkos::subview(partial_derivatives.xiz, ispec, Kokkos::ALL(),
                              Kokkos::ALL());
  this->gammaz = Kokkos::subview(partial_derivatives.gammaz, ispec,
                                 Kokkos::ALL(), Kokkos::ALL());
  this->jacobian = Kokkos::subview(partial_derivatives.jacobian, ispec,
                                   Kokkos::ALL(), Kokkos::ALL());
  this->lambdaplus2mu = Kokkos::subview(properties.lambdaplus2mu, ispec,
                                        Kokkos::ALL(), Kokkos::ALL());
  this->mu =
      Kokkos::subview(properties.mu, ispec, Kokkos::ALL(), Kokkos::ALL());
  this->rho =
      Kokkos::subview(properties.rho, ispec, Kokkos::ALL(), Kokkos::ALL());

  return;
}

template <int NGLL>
KOKKOS_INLINE_FUNCTION
    type_real[medium::components] specfem::domain::impl::elements::element<
        specfem::enums::element::dimension::dim2,
        specfem::enums::element::medium::elastic,
        specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
        specfem::enums::element::property::isotropic>::
        compute_mass_matrix_component(const int &xz) const {
  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);

  static_assert(medium::components == 2,
                "Number of components must be 2 for 2D isotropic elastic "
                "medium");

  type_real[specfem::enums::element::medium::elastic::components] mass_matrix{
    this->rho(iz, ix) * this->jacobian(iz, ix),
    this->rho(iz, ix) * this->jacobian(iz, ix)
  };

  return mass_matrix;
}

template <int NGLL>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>::
    compute_gradient(const int &xz,
                     const StaticScratchViewType<NGLL, type_real> s_hprime_xx,
                     const StaticScratchViewType<NGLL, type_real> s_hprime_zz,
                     const StaticScratchViewType<NGLL, type_real> field_x,
                     const StaticScratchViewType<NGLL, type_real> field_z,
                     type_real *duxdxl, type_real *duxdzl, type_real *duzdxl,
                     type_real *duzdzl) const {

  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);

  const type_real xixl = this->xix(iz, ix);
  const type_real gammaxl = this->gammax(iz, ix);
  const type_real xizl = this->xiz(iz, ix);
  const type_real gammazl = this->gammaz(iz, ix);

  type_real sum_hprime_x1 = 0.0;
  type_real sum_hprime_x3 = 0.0;
  type_real sum_hprime_z1 = 0.0;
  type_real sum_hprime_z3 = 0.0;

  for (int l = 0; l < NGLL; l++) {
    sum_hprime_x1 += s_hprime_xx(ix, l) * field_x(iz, l);
    sum_hprime_x3 += s_hprime_xx(ix, l) * field_z(iz, l);
    sum_hprime_z1 += s_hprime_zz(iz, l) * field_x(l, ix);
    sum_hprime_z3 += s_hprime_zz(iz, l) * field_z(l, ix);
  }
  // duxdx
  *duxdxl = xixl * sum_hprime_x1 + gammaxl * sum_hprime_x3;

  // duxdz
  *duxdzl = xizl * sum_hprime_x1 + gammazl * sum_hprime_x3;

  // duzdx
  *duzdxl = xixl * sum_hprime_z1 + gammaxl * sum_hprime_z3;

  // duzdz
  *duzdzl = xizl * sum_hprime_z1 + gammazl * sum_hprime_z3;

  return;
}

template <int NGLL>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>::
    compute_stress(const int &xz, const type_real &duxdxl,
                   const type_real &duxdzl, const type_real &duzdxl,
                   const type_real &duzdzl, type_real *stress_integrand_1l,
                   type_real *stress_integrand_2l,
                   type_real *stress_integrand_3l,
                   type_real *stress_integrand_4l) const {

  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);

  const type_real xixl = this->xix(iz, ix);
  const type_real gammaxl = this->gammax(iz, ix);
  const type_real xizl = this->xiz(iz, ix);
  const type_real gammazl = this->gammaz(iz, ix);
  const type_real jacobianl = this->jacobian(iz, ix);
  const type_real lambdaplus2mul = this->lambdaplus2mu(iz, ix);
  const type_real mul = this->mu(iz, ix);
  const type_real lambdal = lambdaplus2mul - 2.0 * mul;

  type_real sigma_xx, sigma_zz, sigma_xz;

  if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
    // P_SV case
    // sigma_xx
    sigma_xx = lambdaplus2mul * duxdxl + lambdal * duzdzl;

    // sigma_zz
    sigma_zz = lambdaplus2mul * duzdzl + lambdal * duxdxl;

    // sigma_xz
    sigma_xz = mul * (duzdxl + duxdzl);
  } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
    // SH-case
    // sigma_xx
    sigma_xx = mul * duxdxl; // would be sigma_xy in
                             // CPU-version

    // sigma_xz
    sigma_xz = mul * duxdzl; // sigma_zy
  }

  *stress_integrand_1l = jacobianl * (sigma_xx * xixl + sigma_xz * xizl);
  *stress_integrand_2l = jacobianl * (sigma_xz * xixl + sigma_zz * xizl);
  *stress_integrand_3l = jacobianl * (sigma_xx * gammaxl + sigma_xz * gammazl);
  *stress_integrand_4l = jacobianl * (sigma_xz * gammaxl + sigma_zz * gammazl);

  return;
}

template <int NGLL>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>::
    update_acceleration(
        const int &xz, const type_real &wxglll, const type_real &wzglll,
        const StaticScratchViewType<NGLL, type_real> stress_integrand_1,
        const StaticScratchViewType<NGLL, type_real> stress_integrand_2,
        const StaticScratchViewType<NGLL, type_real> stress_integrand_3,
        const StaticScratchViewType<NGLL, type_real> stress_integrand_4,
        const StaticScratchViewType<NGLL, type_real> s_hprimewgll_xx,
        const StaticScratchViewType<NGLL, type_real> s_hprimewgll_zz,
        field_type field_dot_dot) const {

  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);
  type_real tempx1 = 0.0;
  type_real tempz1 = 0.0;
  type_real tempx3 = 0.0;
  type_real tempz3 = 0.0;

#pragma unroll
  for (int l = 0; l < NGLL; l++) {
    tempx1 += s_hprimewgll_xx(ix, l) * stress_integrand_1(iz, l);
    tempz1 += s_hprimewgll_xx(ix, l) * stress_integrand_2(iz, l);
    tempx3 += s_hprimewgll_zz(iz, l) * stress_integrand_3(l, ix);
    tempz3 += s_hprimewgll_zz(iz, l) * stress_integrand_4(l, ix);
  }

  const type_real sum_terms1 = -1.0 * (wzglll * tempx1) - (wxglll * tempx3);
  const type_real sum_terms3 = -1.0 * (wzglll * tempz1) - (wxglll * tempz3);
  Kokkos::atomic_add(&field_dot_dot(0), sum_terms1);
  Kokkos::atomic_add(&field_dot_dot(1), sum_terms3);
}

#endif
