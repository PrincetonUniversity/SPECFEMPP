#pragma once

#include "algorithms/dot.hpp"
#include "elastic_anisotropic.hpp"
#include "globals.h"
#include "specfem_setup.hpp"

template <bool UseSIMD>
KOKKOS_FUNCTION specfem::point::kernels<specfem::dimension::type::dim2,
    specfem::element::medium_tag::elastic,
    specfem::element::property_tag::anisotropic, UseSIMD>
specfem::frechet_derivatives::impl::impl_compute_element_kernel(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::anisotropic, UseSIMD> &properties,
    const specfem::point::field<specfem::dimension::type::dim2,
                                specfem::element::medium_tag::elastic, false,
                                false, true, false, UseSIMD> &adjoint_field,
    const specfem::point::field<specfem::dimension::type::dim2,
                                specfem::element::medium_tag::elastic, true,
                                false, false, false, UseSIMD> &backward_field,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        UseSIMD> &adjoint_derivatives,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        UseSIMD> &backward_derivatives,
    const type_real &dt) {

  using datatype =
      typename specfem::datatype::simd<type_real, UseSIMD>::datatype;

  const datatype kappa = properties.lambdaplus2mu - properties.mu;

  if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
    const datatype ad_dsxx = adjoint_derivatives.du(0, 0);
    const datatype ad_dsxz =
        static_cast<type_real>(0.5) *
        (adjoint_derivatives.du(0, 1) + adjoint_derivatives.du(1, 0));
    const datatype ad_dszz = adjoint_derivatives.du(1, 1);

    const datatype b_dsxx = backward_derivatives.du(0, 0);
    const datatype b_dsxz =
        static_cast<type_real>(0.5) *
        (backward_derivatives.du(0, 1) + backward_derivatives.du(1, 0));
    const datatype b_dszz = backward_derivatives.du(1, 1);

    // rho kernel
    datatype rho_kl = specfem::algorithms::dot(adjoint_field.acceleration,
                                               backward_field.displacement);

    // Inner part of the 2-D version of Equation 15 in Tromp et al. 2005
    // That is \eps_{jk} \eps_{lm}
    datatype c11_kl = ad_dsxx * b_dsxx;
    datatype c13_kl = ad_dsxx * b_dszz + ad_dszz * b_dsxx;
    datatype c15_kl = 2 * (
          ad_dsxx * static_cast<type_real>(0.5) * (b_dsxz + b_dszx)
        + static_cast<type_real>(0.5) * (ad_dsxz + ad_dszx) * b_dsxx);
    datatype c33_kl = ad_dszz * b_dszz;
    datatype c35_kl = 2 * ( static_cast<type_real>(0.5) * (b_dsxz + b_dszx) * ad_dszz
                 + static_cast<type_real>(0.5) * (ad_dsxz + ad_dszx) * b_dszz);
    datatype c55_kl = 4 * static_cast<type_real>(0.5) * (ad_dsxz + ad_dszx) ;
               * static_cast<type_real>(0.5) * (b_dsxz + b_dszx);

    //
    const rho_kl = static_cast<type_real>(-1.0) * properties.rho * dt * rho_kl;
    const c11_kl = static_cast<type_real>(-1.0) * c11_kl * properties.c11 * dt;
    const c13_kl = static_cast<type_real>(-1.0) * c13_kl * properties.c13 * dt;
    const c15_kl = static_cast<type_real>(-1.0) * c15_kl * properties.c15 * dt;
    const c33_kl = static_cast<type_real>(-1.0) * c33_kl * properties.c33 * dt;
    const c35_kl = static_cast<type_real>(-1.0) * c35_kl * properties.c35 * dt;
    const c55_kl = static_cast<type_real>(-1.0) * c55_kl * properties.c55 * dt;

    return { rho_kl, c11_kl, c13_kl, c15_kl, c33_kl, c35_kl, c55_kl };
  } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
    static_assert("Full anisotropic kernels for SH waves are not supported yet.");
    return { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  } else {
    static_assert("Simulation wave not supported");
    return { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  }
}
