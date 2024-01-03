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
                                       specfem::kokkos::array_type<type_real, 2>
                                           &mass_matrix) const {
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
template <specfem::enums::time_scheme::type time_scheme>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic, BC>::
    mass_time_contribution(
        const int &ispec, const int &ielement, const int &xz,
        const type_real &dt,
        const specfem::kokkos::array_type<type_real, dimension::dim> &weight,
        specfem::kokkos::array_type<type_real, 2> &rmass_inverse) const {

  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);

  constexpr int components = medium_type::components;

  const specfem::compute::element_partial_derivatives partial_derivatives =
      specfem::compute::element_partial_derivatives(
          this->xix(ispec, iz, ix), this->gammax(ispec, iz, ix),
          this->xiz(ispec, iz, ix), this->gammaz(ispec, iz, ix),
          this->jacobian(ispec, iz, ix));

  const specfem::compute::element_properties<medium_type::value,
                                             property_type::value>
      properties(this->lambdaplus2mu(ispec, iz, ix), this->mu(ispec, iz, ix),
                 this->rho(ispec, iz, ix));

  rmass_inverse[0] = 0.0;
  rmass_inverse[1] = 0.0;

  boundary_conditions.template mass_time_contribution<time_scheme>(
      ielement, xz, dt, weight, partial_derivatives, properties, rmass_inverse);

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
        specfem::kokkos::array_type<type_real, 2> &dudxl,
        specfem::kokkos::array_type<type_real, 2> &dudzl) const {

  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);

  const specfem::compute::element_partial_derivatives partial_derivatives =
      specfem::compute::element_partial_derivatives(
          this->xix(ispec, iz, ix), this->gammax(ispec, iz, ix),
          this->xiz(ispec, iz, ix), this->gammaz(ispec, iz, ix));

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

  boundary_conditions.enforce_gradient(ielement, xz, partial_derivatives, dudxl,
                                       dudzl);

  return;
}

template <int NGLL, typename BC>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic,
    BC>::compute_stress(const int &ispec, const int &ielement, const int &xz,
                        const specfem::kokkos::array_type<type_real, 2> &dudxl,
                        const specfem::kokkos::array_type<type_real, 2> &dudzl,
                        specfem::kokkos::array_type<type_real, 2>
                            &stress_integrand_xi,
                        specfem::kokkos::array_type<type_real, 2>
                            &stress_integrand_gamma) const {

  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);

  const specfem::compute::element_partial_derivatives partial_derivatives =
      specfem::compute::element_partial_derivatives(
          this->xix(ispec, iz, ix), this->gammax(ispec, iz, ix),
          this->xiz(ispec, iz, ix), this->gammaz(ispec, iz, ix),
          this->jacobian(ispec, iz, ix));

  const specfem::compute::element_properties<medium_type::value,
                                             property_type::value>
      properties(this->lambdaplus2mu(ispec, iz, ix), this->mu(ispec, iz, ix),
                 this->rho(ispec, iz, ix));

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

  boundary_conditions.enforce_stress(ielement, xz, partial_derivatives,
                                     properties, stress_integrand_xi,
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
        const specfem::kokkos::array_type<type_real, dimension::dim> &weight,
        const ScratchViewType<type_real, medium_type::components>
            stress_integrand_xi,
        const ScratchViewType<type_real, medium_type::components>
            stress_integrand_gamma,
        const ScratchViewType<type_real, 1> s_hprimewgll_xx,
        const ScratchViewType<type_real, 1> s_hprimewgll_zz,
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

  specfem::compute::element_partial_derivatives partial_derivatives;

  specfem::compute::element_properties<medium_type::value, property_type::value>
      properties;

  //   populate partial derivatives only if the boundary is stacey
  if constexpr (boundary_conditions_type::value ==
                specfem::enums::element::boundary_tag::stacey) {
    partial_derivatives = specfem::compute::element_partial_derivatives(
        this->xix(ispec, iz, ix), this->gammax(ispec, iz, ix),
        this->xiz(ispec, iz, ix), this->gammaz(ispec, iz, ix),
        this->jacobian(ispec, iz, ix));

    properties = specfem::compute::element_properties<medium_type::value,
                                                      property_type::value>(
        this->lambdaplus2mu(ispec, iz, ix), this->mu(ispec, iz, ix),
        this->rho(ispec, iz, ix));
  }

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
  for (int l = 0; l < NGLL; l++) {
    tempx1 += s_hprimewgll_xx(ix, l, 0) * stress_integrand_xi(iz, l, 0);
    tempz1 += s_hprimewgll_xx(ix, l, 0) * stress_integrand_xi(iz, l, 1);
    tempx3 += s_hprimewgll_zz(iz, l, 0) * stress_integrand_gamma(l, ix, 0);
    tempz3 += s_hprimewgll_zz(iz, l, 0) * stress_integrand_gamma(l, ix, 1);
  }

  acceleration[0] = -1.0 * (weight[1] * tempx1) - (weight[0] * tempx3);
  acceleration[1] = -1.0 * (weight[1] * tempz1) - (weight[0] * tempz3);

  boundary_conditions.enforce_traction(ielement, xz, weight,
                                       partial_derivatives, properties,
                                       velocity, acceleration);
}

#endif
