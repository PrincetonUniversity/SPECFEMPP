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
    const datatype ad_dsxx = adjoint_derivatives.du(0, 0);

    // ad_dsxz = 0.5 * (ds#x/dz + ds#z/dx)
    const datatype ad_dsxz =
        static_cast<type_real>(0.5) *
        (adjoint_derivatives.du(0, 1) + adjoint_derivatives.du(1, 0));

    // ad_dszz = 0.5 * (ds#z/dz + ds#z/dz)
    const datatype ad_dszz = adjoint_derivatives.du(1, 1);

    // b_dsxx = 0.5 * (dsx/dx + dsx/dx) = dsx/dx
    const datatype b_dsxx = backward_derivatives.du(0, 0);

    // b_dsxz = 0.5 * (dsx/dz + dsz/dx)
    const datatype b_dsxz =
        static_cast<type_real>(0.5) *
        (backward_derivatives.du(0, 1) + backward_derivatives.du(1, 0));

    // b_dszz = 0.5 * (dsz/dz + dsz/dz) = dsz/dz
    const datatype b_dszz = backward_derivatives.du(1, 1);

    // inner part of rho kernel equation 14
    // rho_kl = s#''_i * s_j
    datatype rho_kl = specfem::algorithms::dot(adjoint_field.acceleration,
                                               backward_field.displacement);

    // Inner part of the 2-D version of Equation 15 in Tromp et al. 2005
    // That is \eps_{jk} \eps_{lm}
    datatype c11_kl = ad_dsxx * b_dsxx;
    datatype c13_kl = ad_dsxx * b_dszz + ad_dszz * b_dsxx;
    datatype c15_kl = 2 * ad_dsxx * b_dsxz + ad_dsxz * b_dsxx;
    datatype c33_kl = ad_dszz * b_dszz;
    datatype c35_kl = 2 * b_dsxz * ad_dszz + ad_dsxz * b_dszz;
    datatype c55_kl = 4 * ad_dsxz * b_dsxz;

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
    // const datatype ad_dsyx = static_cast<type_real>(0.5) * adjoint_derivatives.du(0, 0);

    // // ad_dsyz = 0.5 * (ds#y/dz + ds#z/dy) = 0.5 * (ds#y/dz)
    // const datatype ad_dszz = static_cast<type_real>(0.5) * adjoint_derivatives.du(1, 0);

    // // b_dsyx = 0.5 * (dsy/dx + dsx/dy) = 0.5 * dsy/dx
    // const datatype b_dsyx = static_cast<type_real>(0.5) * backward_derivatives.du(0, 0);

    // // b_dsyz = 0.5 * (dsy/dz + dsz/dy) = 0.5 * dsz/dx
    // const datatype b_dsyz = static_cast<type_real>(0.5) * backward_derivatives.du(1, 0));

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
