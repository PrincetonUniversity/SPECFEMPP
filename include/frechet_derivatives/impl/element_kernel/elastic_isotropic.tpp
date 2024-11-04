#ifndef _FRECHET_DERIVATIVES_IMPL_ELEMENT_KERNEL_ELASTIC_ISOTROPIC_TPP
#define _FRECHET_DERIVATIVES_IMPL_ELEMENT_KERNEL_ELASTIC_ISOTROPIC_TPP

#include "algorithms/dot.hpp"
#include "elastic_isotropic.hpp"
#include "globals.h"
#include "specfem_setup.hpp"

template <bool UseSIMD>
KOKKOS_FUNCTION specfem::point::kernels<specfem::dimension::type::dim2,
    specfem::element::medium_tag::elastic,
    specfem::element::property_tag::isotropic, UseSIMD>
specfem::frechet_derivatives::impl::impl_compute_element_kernel(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic, UseSIMD> &properties,
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

    // const type_real kappa_kl =
    //     -1.0 * kappa * dt * ((ad_dsxx + ad_dszz) * (b_dsxx + b_dszz));
    // const type_real mu_kl = -2.0 * properties.mu * dt *
    //                         (ad_dsxx * b_dsxx + ad_dszz * b_dszz +
    //                          2.0 * ad_dsxz * b_dsxz - 1.0 / 3.0 * kappa_kl);
    // const type_real rho_kl =
    //     -1.0 * properties.rho * dt *
    //     (specfem::algorithms::dot(adjoint_field.acceleration,
    //                               backward_field.displacement));

    datatype kappa_kl = (ad_dsxx + ad_dszz) * (b_dsxx + b_dszz);
    datatype mu_kl = (ad_dsxx * b_dsxx + ad_dszz * b_dszz +
                      static_cast<type_real>(2.0) * ad_dsxz * b_dsxz -
                      static_cast<type_real>(1.0 / 3.0) * kappa_kl);
    datatype rho_kl = specfem::algorithms::dot(adjoint_field.acceleration,
                                               backward_field.displacement);

    kappa_kl = static_cast<type_real>(-1.0) * kappa * dt * kappa_kl;
    mu_kl = static_cast<type_real>(-2.0) * properties.mu * dt * mu_kl;
    rho_kl = static_cast<type_real>(-1.0) * properties.rho * dt * rho_kl;

    const datatype rhop_kl = rho_kl + kappa_kl + mu_kl;

    const datatype beta_kl = static_cast<type_real>(2.0) *
                             (mu_kl - static_cast<type_real>(4.0 / 3.0) *
                                          properties.mu / kappa * kappa_kl);

    const datatype alpha_kl =
        static_cast<type_real>(2.0) *
        (static_cast<type_real>(1.0) +
         static_cast<type_real>(4.0 / 3.0) * properties.mu / kappa) *
        kappa_kl;

    return { rho_kl, mu_kl, kappa_kl, rhop_kl, alpha_kl, beta_kl };
  } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
    const datatype kappa_kl = 0.0;
    const datatype mu_kl =
        static_cast<type_real>(-2.0) * properties.mu * dt *
        static_cast<type_real>(0.5) *
        (adjoint_derivatives.du(0, 0) * backward_derivatives.du(0, 0) +
         adjoint_derivatives.du(1, 0) * backward_derivatives.du(1, 0));
    const datatype rho_kl =
        static_cast<type_real>(-1.0) * properties.rho * dt *
        specfem::algorithms::dot(adjoint_field.acceleration,
                                 backward_field.displacement);

    const datatype rhop_kl = rho_kl + kappa_kl + mu_kl;
    const datatype alpha_kl = 0.0;
    const datatype beta_kl = static_cast<type_real>(2.0) * mu_kl;

    return { rho_kl, mu_kl, kappa_kl, rhop_kl, alpha_kl, beta_kl };
  } else {
    static_assert("Simulation wave not supported");
    return { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  }
}

#endif /* _FRECHET_DERIVATIVES_IMPL_ELEMENT_KERNEL_ELASTIC_ISOTROPIC_TPP */
