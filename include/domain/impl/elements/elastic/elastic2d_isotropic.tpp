#ifndef _DOMAIN_ELASTIC_ELEMENTS2D_ISOTROPIC_TPP
#define _DOMAIN_ELASTIC_ELEMENTS2D_ISOTROPIC_TPP

#include "compute/interface.hpp"
#include "domain/impl/elements/elastic/elastic2d.hpp"
#include "domain/impl/elements/elastic/elastic2d_isotropic.hpp"
#include "domain/impl/elements/element.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "specfem_enums.hpp"
#include "specfem_setup.hpp"

template <typename quadrature_points>
specfem::compute::partial_derivatives specfem::Domain::impl::elements::element<
    specfem::enums::element::dimension::dim2, quadrature_points,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::property::isotropic>::partial_derivatives =
    specfem::compute::partial_derivatives();

template <typename quadrature_points>
specfem::compute::properties specfem::Domain::impl::elements::element<
    specfem::enums::element::dimension::dim2, quadrature_points,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::property::isotropic>::properties =
    specfem::compute::properties();

template <typename quadrature_points>
specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
    specfem::Domain::impl::elements::element<
        specfem::enums::element::dimension::dim2, quadrature_points,
        specfem::enums::element::medium::elastic,
        specfem::enums::element::property::isotropic>::field_dot_dot =
        specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>();

template <typename quadrature_points>
specfem::Domain::impl::elements::element<
    specfem::enums::element::dimension::dim2, quadrature_points,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::property::isotropic>::
    element(const int ispec,
            const specfem::compute::partial_derivatives partial_derivatives,
            const specfem::compute::properties properties,
            const specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
                field_dot_dot)
    : ispec(ispec) {

  this->partial_derivatives = partial_derivatives;
  this->properties = properties;
  this->field_dot_dot = field_dot_dot;

  static_assert(this->partial_derivatives.xix.extent(1) ==
                quadrature_points::NGLL);
  static_assert(this->partial_derivatives.xix.extent(2) ==
                quadrature_points::NGLL);
  static_assert(this->partial_derivatives.gammax.extent(1) ==
                quadrature_points::NGLL);
  static_assert(this->partial_derivatives.gammax.extent(2) ==
                quadrature_points::NGLL);
  static_assert(this->partial_derivatives.xiz.extent(1) ==
                quadrature_points::NGLL);
  static_assert(this->partial_derivatives.xiz.extent(2) ==
                quadrature_points::NGLL);
  static_assert(this->partial_derivatives.gammaz.extent(1) ==
                quadrature_points::NGLL);
  static_assert(this->partial_derivatives.gammaz.extent(2) ==
                quadrature_points::NGLL);
  static_assert(this->partial_derivatives.jacobian.extent(1) ==
                quadrature_points::NGLL);
  static_assert(this->partial_derivatives.jacobian.extent(2) ==
                quadrature_points::NGLL);

  static_assert(this->properties.lambdaplus2mu.extent(1) =
                    quadrature_points::NGLL);
  static_assert(this->properties.mu.extent(1) = quadrature_points::NGLL);

  return;
}

template <typename quadrature_points>
KOKKOS_FUNCTION void specfem::Domain::impl::elements::element<
    specfem::enums::element::dimension::dim2, quadrature_points,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::property::isotropic>::
    compute_gradient(
        const int &xz,
        const specfem::kokkos::StaticDeviceScratchView2d<
            type_real, quadrature_points::NGLL, quadrature_points::NGLL>
            s_hprime_xx,
        const specfem::kokkos::StaticDeviceScratchView2d<
            type_real, quadrature_points::NGLL, quadrature_points::NGLL>
            s_hprime_zz,
        const specfem::kokkos::StaticDeviceScratchView2d<
            type_real, quadrature_points::NGLL, quadrature_points::NGLL>
            field_x,
        const specfem::kokkos::StaticDeviceScratchView2d<
            type_real, quadrature_points::NGLL, quadrature_points::NGLL>
            field_z,
        type_real &duxdxl, type_real &duxdzl, type_real &duzdxl,
        type_real &duzdzl) const {

  constexpr int NGLL = quadrature_points::NGLL;
  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);

  const type_real xixl = this->partial_derivatives.xix(ispec, iz, ix);
  const type_real xizl = this->partial_derivatives.xiz(ispec, iz, ix);
  const type_real gammaxl = this->partial_derivatives.gammax(ispec, iz, ix);
  const type_real gammazl = this->partial_derivatives.gammaz(ispec, iz, ix);

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
  duxdxl = xixl * sum_hprime_x1 + gammaxl * sum_hprime_x3;

  // duxdz
  duxdzl = xizl * sum_hprime_x1 + gammazl * sum_hprime_x3;

  // duzdx
  duzdxl = xixl * sum_hprime_z1 + gammaxl * sum_hprime_z3;

  // duzdz
  duzdzl = xizl * sum_hprime_z1 + gammazl * sum_hprime_z3;

  return;
}

template <typename quadrature_points>
KOKKOS_FUNCTION void specfem::Domain::impl::elements::element<
    specfem::enums::element::dimension::dim2, quadrature_points,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::property::isotropic>::
    compute_stress(const int &xz, const type_real &duxdxl,
                   const type_real &duxdzl, const type_real &duzdxl,
                   const type_real &duzdzl, type_real &stress_integrand_1l,
                   type_real &stress_integrand_2l,
                   type_real &stress_integrand_3l,
                   type_real &stress_integrand_4l) const {

  constexpr int NGLL = quadrature_points::NGLL;
  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);

  const type_real lambdaplus2mul =
      this->properties.lambdaplus2mu(ispec, iz, ix);
  const type_real mul = this->properties.mu(ispec, iz, ix);
  const type_real lambdal = lambdaplus2mul - 2.0 * mul;

  const type_real xixl = this->partial_derivatives.xix(ispec, iz, ix);
  const type_real xizl = this->partial_derivatives.xiz(ispec, iz, ix);
  const type_real gammaxl = this->partial_derivatives.gammax(ispec, iz, ix);
  const type_real gammazl = this->partial_derivatives.gammaz(ispec, iz, ix);
  const type_real jacobianl = this->partial_derivatives.jacobian(ispec, iz, ix);

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

  stress_integrand_1l = jacobianl * (sigma_xx * xixl + sigma_xz * xizl);
  stress_integrand_2l = jacobianl * (sigma_xz * xixl + sigma_zz * xizl);
  stress_integrand_3l = jacobianl * (sigma_xx * gammaxl + sigma_xz * gammazl);
  stress_integrand_4l = jacobianl * (sigma_xz * gammaxl + sigma_zz * gammazl);

  return;
}

template <typename quadrature_points>
KOKKOS_FUNCTION void specfem::Domain::impl::elements::element<
    specfem::enums::element::dimension::dim2, quadrature_points,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::property::isotropic>::
    update_acceleration(
        const int &xz, const int &iglob, const type_real &wxglll,
        const type_real &wzglll,
        const specfem::kokkos::StaticDeviceScratchView2d<
            type_real, quadrature_points::NGLL, quadrature_points::NGLL>
            stress_integrand_1,
        const specfem::kokkos::StaticDeviceScratchView2d<
            type_real, quadrature_points::NGLL, quadrature_points::NGLL>
            stress_integrand_2,
        const specfem::kokkos::StaticDeviceScratchView2d<
            type_real, quadrature_points::NGLL, quadrature_points::NGLL>
            stress_integrand_3,
        const specfem::kokkos::StaticDeviceScratchView2d<
            type_real, quadrature_points::NGLL, quadrature_points::NGLL>
            stress_integrand_4,
        const specfem::kokkos::StaticDeviceScratchView2d<
            type_real, quadrature_points::NGLL, quadrature_points::NGLL>
            s_hprimewgll_xx,
        const specfem::kokkos::StaticDeviceScratchView2d<
            type_real, quadrature_points::NGLL, quadrature_points::NGLL>
            s_hprimewgll_zz) const {

  constexpr int NGLL = quadrature_points::NGLL;
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
  Kokkos::atomic_add(&field_dot_dot(iglob, 0), sum_terms1);
  Kokkos::atomic_add(&field_dot_dot(iglob, 1), sum_terms3);
}

template <>
specfem::compute::partial_derivatives specfem::Domain::impl::elements::element<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::property::isotropic>::partial_derivatives =
    specfem::compute::partial_derivatives();

template <>
specfem::compute::properties specfem::Domain::impl::elements::element<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::property::isotropic>::properties =
    specfem::compute::properties();

template <>
specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
    specfem::Domain::impl::elements::element<
        specfem::enums::element::dimension::dim2,
        specfem::enums::element::medium::elastic,
        specfem::enums::element::property::isotropic>::field_dot_dot =
        specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>();

template <>
KOKKOS_FUNCTION specfem::Domain::impl::elements::element<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::property::isotropic>::
    element(const int ispec,
            const specfem::compute::partial_derivatives partial_derivatives,
            const specfem::compute::properties properties,
            const specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
                field_dot_dot)
    : ispec(ispec) {

  this->partial_derivatives = partial_derivatives;
  this->properties = properties;
  this->field_dot_dot = field_dot_dot;

  this->ngllx = partial_derivatives.xix.extent(2);
  this->ngllz = partial_derivatives.xix.extent(1);

  assert(partial_derivatives.xiz.extent(1) == ngllz);
  assert(partial_derivatives.xiz.extent(2) == ngllx);
  assert(partial_derivatives.gammax.extent(1) == ngllz);
  assert(partial_derivatives.gammax.extent(2) == ngllx);
  assert(partial_derivatives.gammaz.extent(1) == ngllz);
  assert(partial_derivatives.gammaz.extent(2) == ngllx);
  assert(partial_derivatives.jacobian.extent(1) == ngllz);
  assert(partial_derivatives.jacobian.extent(2) == ngllx);
  assert(properties.lambdaplus2mu.extent(1) == ngllz);
  assert(properties.lambdaplus2mu.extent(2) == ngllx);
  assert(properties.mu.extent(1) == ngllz);
  assert(properties.mu.extent(2) == ngllx);

  return;
}

template <>
KOKKOS_FUNCTION void specfem::Domain::impl::elements::element<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::property::isotropic>::
    compute_gradient(
        const int &xz,
        const specfem::kokkos::DeviceScratchView2d<type_real> s_hprime_xx,
        const specfem::kokkos::DeviceScratchView2d<type_real> s_hprime_zz,
        const specfem::kokkos::DeviceScratchView2d<type_real> field_x,
        const specfem::kokkos::DeviceScratchView2d<type_real> field_z,
        type_real &duxdxl, type_real &duxdzl, type_real &duzdxl,
        type_real &duzdzl) const {

  assert(s_hprime_xx.extent(0) == ngllx);
  assert(s_hprime_xx.extent(1) == ngllx);

  assert(s_hprime_xx.extent(0) == ngllz);
  assert(s_hprime_xx.extent(1) == ngllz);

  assert(field_x.extent(0) == ngllz);
  assert(field_x.extent(1) == ngllx);
  assert(field_z.extent(0) == ngllz);
  assert(field_z.extent(1) == ngllx);

  int iz, ix;
  sub2ind(xz, ngllx, iz, ix);

  const type_real xixl = this->partial_derivatives.xix(this->ispec, iz, ix);
  const type_real xizl = this->partial_derivatives.xiz(this->ispec, iz, ix);
  const type_real gammaxl =
      this->partial_derivatives.gammax(this->ispec, iz, ix);
  const type_real gammazl =
      this->partial_derivatives.gammaz(this->this->ispec, iz, ix);

  type_real sum_hprime_x1 = 0.0;
  type_real sum_hprime_x3 = 0.0;
  type_real sum_hprime_z1 = 0.0;
  type_real sum_hprime_z3 = 0.0;

  for (int l = 0; l < ngllx; l++) {
    sum_hprime_x1 += s_hprime_xx(ix, l) * field_x(iz, l);
    sum_hprime_x3 += s_hprime_xx(ix, l) * field_z(iz, l);
  }

  for (int l = 0; l < ngllz; l++) {
    sum_hprime_z1 += s_hprime_zz(iz, l) * field_x(l, ix);
    sum_hprime_z3 += s_hprime_zz(iz, l) * field_z(l, ix);
  }
  // duxdx
  duxdxl = xixl * sum_hprime_x1 + gammaxl * sum_hprime_x3;

  // duxdz
  duxdzl = xizl * sum_hprime_x1 + gammazl * sum_hprime_x3;

  // duzdx
  duzdxl = xixl * sum_hprime_z1 + gammaxl * sum_hprime_z3;

  // duzdz
  duzdzl = xizl * sum_hprime_z1 + gammazl * sum_hprime_z3;

  return;
}

template <>
KOKKOS_FUNCTION void specfem::Domain::impl::elements::element<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::property::isotropic>::
    compute_stress(const int &xz, const int &ngllx, const type_real &duxdxl,
                   const type_real &duxdzl, const type_real &duzdxl,
                   const type_real &duzdzl, type_real &stress_integrand_1l,
                   type_real &stress_integrand_2l,
                   type_real &stress_integrand_3l,
                   type_real &stress_integrand_4l) const {

  int ix, iz;
  sub2ind(xz, ngllx, iz, ix);

  const type_real lambdaplus2mul =
      this->properties.lambdaplus2mu(this->ispec, iz, ix);
  const type_real mul = this->properties.mu(thi->ispec, iz, ix);
  const type_real lambdal = lambdaplus2mul - 2.0 * mul;

  const type_real xixl = this->partial_derivatives.xix(this->ispec, iz, ix);
  const type_real xizl = this->partial_derivatives.xiz(this->ispec, iz, ix);
  const type_real gammaxl =
      this->partial_derivatives.gammax(this->ispec, iz, ix);
  const type_real gammazl =
      this->partial_derivatives.gammaz(this->ispec, iz, ix);
  const type_real jacobianl =
      this->partial_derivatives.jacobian(this->ispec, iz, ix);

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

  stress_integrand_1l = jacobianl * (sigma_xx * xixl + sigma_xz * xizl);
  stress_integrand_2l = jacobianl * (sigma_xz * xixl + sigma_zz * xizl);
  stress_integrand_3l = jacobianl * (sigma_xx * gammaxl + sigma_xz * gammazl);
  stress_integrand_4l = jacobianl * (sigma_xz * gammaxl + sigma_zz * gammazl);

  return;
}

template <>
KOKKOS_FUNCTION void specfem::Domain::impl::elements::element<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::property::isotropic>::
    update_acceleration(
        const int &xz, const int &iglob, const type_real &wxglll,
        const type_real &wzglll,
        const specfem::kokkos::DeviceScratchView2d<type_real>
            stress_integrand_1,
        const specfem::kokkos::DeviceScratchView2d<type_real>
            stress_integrand_2,
        const specfem::kokkos::DeviceScratchView2d<type_real>
            stress_integrand_3,
        const specfem::kokkos::DeviceScratchView2d<type_real>
            stress_integrand_4,
        const specfem::kokkos::DeviceScratchView2d<type_real> s_hprimewgll_xx,
        const specfem::kokkos::DeviceScratchView2d<type_real> s_hprimewgll_zz)
        const {

  assert(s_hprimewgll_xx.extent(0) == ngllx);
  assert(s_hprimewgll_xx.extent(1) == ngllx);

  assert(s_hprimewgll_zz.extent(0) == ngllz);
  assert(s_hprimewgll_zz.extent(1) == ngllz);

  assert(stress_integrand_2.extent(0) == ngllz);
  assert(stress_integrand_2.extent(1) == ngllx);

  assert(stress_integrand_3.extent(0) == ngllz);
  assert(stress_integrand_3.extent(1) == ngllx);

  assert(stress_integrand_4.extent(0) == ngllz);
  assert(stress_integrand_4.extent(1) == ngllx);

  int ix, iz;
  sub2ind(xz, ngllx, iz, ix);
  type_real tempx1 = 0.0;
  type_real tempz1 = 0.0;
  type_real tempx3 = 0.0;
  type_real tempz3 = 0.0;

  for (int l = 0; l < ngllx; l++) {
    tempx1 += s_hprimewgll_xx(ix, l) * stress_integrand_1(iz, l);
    tempz1 += s_hprimewgll_xx(ix, l) * stress_integrand_2(iz, l);
  }

  for (int l = 0; l < ngllz; l++) {
    tempx3 += s_hprimewgll_zz(iz, l) * stress_integrand_3(l, ix);
    tempz3 += s_hprimewgll_zz(iz, l) * stress_integrand_4(l, ix);
  }

  const type_real sum_terms1 = -1.0 * (wzglll * tempx1) - (wxglll * tempx3);
  const type_real sum_terms3 = -1.0 * (wzglll * tempz1) - (wxglll * tempz3);
  Kokkos::atomic_add(&field_dot_dot(iglob, 0), sum_terms1);
  Kokkos::atomic_add(&field_dot_dot(iglob, 1), sum_terms3);
}

#endif
