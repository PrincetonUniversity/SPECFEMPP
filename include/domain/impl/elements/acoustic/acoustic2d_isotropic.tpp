#ifndef _DOMAIN_ACOUSTIC_ELEMENTS2D_ISOTROPIC_TPP
#define _DOMAIN_ACOUSTIC_ELEMENTS2D_ISOTROPIC_TPP

#include "compute/interface.hpp"
#include "domain/impl/elements/acoustic/acoustic2d_isotropic.hpp"
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
    specfem::enums::element::medium::acoustic,
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
  assert(properties.rho_inverse.extent(1) == NGLL);
  assert(properties.rho_inverse.extent(2) == NGLL);
  assert(properties.kappa.extent(1) == NGLL);
  assert(properties.kappa.extent(2) == NGLL);
#endif

  // Assert wave property. Acoustic only in sh. For now.
  // assert(specfem::globals::simulation_wave == specfem::wave::sh);

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
  this->rho_inverse = Kokkos::subview(properties.rho_inverse, ispec,
                                      Kokkos::ALL(), Kokkos::ALL());
  this->kappa =
      Kokkos::subview(properties.kappa, ispec, Kokkos::ALL(), Kokkos::ALL());
  return;
}

template <int NGLL>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::acoustic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>::
    compute_mass_matrix_component(const int &xz, type_real* mass_matrix) const{
  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);

  constexpr int components = medium_type::components;

  static_assert(components == 1,
                "Acoustic medium has only one component");

  mass_matrix[0] = this->jacobian(iz, ix) / this->kappa(iz, ix);

  return;
}

template <int NGLL>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::acoustic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>::
    compute_gradient(const int &xz,
                     const ScratchViewType<type_real, 1> s_hprime_xx,
                     const ScratchViewType<type_real, 1> s_hprime_zz,
                     const ScratchViewType<type_real, medium_type::components> field_chi,
                     type_real *dchidxl, type_real *dchidzl) const {

  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);

  const type_real xixl = this->xix(iz, ix);
  const type_real gammaxl = this->gammax(iz, ix);
  const type_real xizl = this->xiz(iz, ix);
  const type_real gammazl = this->gammaz(iz, ix);

  type_real dchi_dxi = 0.0;
  type_real dchi_dgamma = 0.0;

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
  for (int l = 0; l < NGLL; l++) {
    dchi_dxi += s_hprime_xx(ix, l, 0) * field_chi(iz, l, 0);
    dchi_dgamma += s_hprime_zz(iz, l, 0) * field_chi(l, ix, 0);
  }

  // dchidx
  dchidxl[0] = dchi_dxi * xixl + dchi_dgamma * gammaxl;

  // dchidz
  dchidzl[0] = dchi_dxi * xizl + dchi_dgamma * gammazl;

  return;
}

template <int NGLL>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::acoustic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>::
    compute_stress(const int &xz, const type_real *dchidxl,
                   const type_real *dchidzl, type_real *stress_integrand_xi,
                   type_real *stress_integrand_gamma) const {

  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);

  const type_real xixl = this->xix(iz, ix);
  const type_real gammaxl = this->gammax(iz, ix);
  const type_real xizl = this->xiz(iz, ix);
  const type_real gammazl = this->gammaz(iz, ix);
  const type_real jacobianl = this->jacobian(iz, ix);
  const type_real rho_inversel = this->rho_inverse(iz, ix);

  // Precompute the factor
  type_real fac = jacobianl * rho_inversel;

  // Compute stress integrands 1 and 2
  // Here it is extremely important that this seems at odds with
  // equations (44) & (45) from Komatitsch and Tromp 2002 I. - Validation
  // The equations are however missing dxi/dx, dxi/dz, dzeta/dx, dzeta/dz
  // for the gradient of w^{\alpha\gamma}. In this->update_acceleration
  // the weights for the integration and the interpolated values for the
  // first derivatives of the lagrange polynomials are then collapsed
  stress_integrand_xi[0] = fac * (xixl * dchidxl[0] + xizl * dchidzl[0]);
  stress_integrand_gamma[0] =
      fac * (gammaxl * dchidxl[0] + gammazl * dchidzl[0]);

  return;
}

template <int NGLL>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::acoustic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>::
    update_acceleration(const int &xz, const type_real &wxglll,
                        const type_real &wzglll,
                        const ScratchViewType<type_real, medium_type::components>
                            stress_integrand_xi,
                        const ScratchViewType<type_real, medium_type::components>
                            stress_integrand_gamma,
                        const ScratchViewType<type_real, 1> s_hprimewgll_xx,
                        const ScratchViewType<type_real, 1> s_hprimewgll_zz,
                        field_type field_dot_dot) const {

  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);
  type_real temp1l = 0.0;
  type_real temp2l = 0.0;

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
  for (int l = 0; l < NGLL; l++) {
    temp1l += s_hprimewgll_xx(ix, l, 0) * stress_integrand_xi(iz, l, 0);
    temp2l += s_hprimewgll_zz(iz, l, 0) * stress_integrand_gamma(l, ix, 0);
  }

  const type_real sum_terms = -1.0 * ((wzglll * temp1l) + (wxglll * temp2l));

  Kokkos::atomic_add(&field_dot_dot(0), sum_terms);
}

#endif
