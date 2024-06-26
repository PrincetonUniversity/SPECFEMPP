#ifndef _FRECHET_DERIVATIVES_IMPL_ELEMENT_KERNEL_ELASTIC_ISOTROPIC_TPP
#define _FRECHET_DERIVATIVES_IMPL_ELEMENT_KERNEL_ELASTIC_ISOTROPIC_TPP

#include "algorithms/dot.hpp"
#include "element_kernel.hpp"
#include "globals.h"
#include "specfem_setup.hpp"

template <>
KOKKOS_FUNCTION specfem::point::kernels<
    specfem::element::medium_tag::elastic,
    specfem::element::property_tag::isotropic>
specfem::frechet_derivatives::impl::element_kernel<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::property_tag::isotropic>(
    const specfem::point::properties<specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
                                     specfem::element::property_tag::isotropic>
        &properties,
    const specfem::frechet_derivatives::impl::AdjointPointFieldType<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic>
        &adjoint_field,
    const specfem::frechet_derivatives::impl::BackwardPointFieldType<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic>
        &backward_field,
    const specfem::frechet_derivatives::impl::PointFieldDerivativesType<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic>
        &adjoint_derivatives,
    const specfem::frechet_derivatives::impl::PointFieldDerivativesType<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic>
        &backward_derivatives,
    const type_real &dt) {

  const type_real kappa = properties.lambdaplus2mu - properties.mu;

  if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
    const type_real ad_dsxx = adjoint_derivatives.du(0, 0);
    const type_real ad_dsxz =
        0.5 * (adjoint_derivatives.du(0, 1) + adjoint_derivatives.du(1, 0));
    const type_real ad_dszz = adjoint_derivatives.du(1, 1);

    const type_real b_dsxx = backward_derivatives.du(0, 0);
    const type_real b_dsxz =
        0.5 * (backward_derivatives.du(0, 1) + backward_derivatives.du(1, 0));
    const type_real b_dszz = backward_derivatives.du(1, 1);

    // const type_real kappa_kl =
    //     -1.0 * kappa * dt * ((ad_dsxx + ad_dszz) * (b_dsxx + b_dszz));
    // const type_real mu_kl = -2.0 * properties.mu * dt *
    //                         (ad_dsxx * b_dsxx + ad_dszz * b_dszz +
    //                          2.0 * ad_dsxz * b_dsxz - 1.0 / 3.0 * kappa_kl);
    // const type_real rho_kl =
    //     -1.0 * properties.rho * dt *
    //     (specfem::algorithms::dot(adjoint_field.acceleration,
    //                               backward_field.displacement));

    type_real kappa_kl = (ad_dsxx + ad_dszz) * (b_dsxx + b_dszz);
    type_real mu_kl = (ad_dsxx * b_dsxx + ad_dszz * b_dszz +
                       2.0 * ad_dsxz * b_dsxz - 1.0 / 3.0 * kappa_kl);
    type_real rho_kl = specfem::algorithms::dot(adjoint_field.acceleration,
                                                backward_field.displacement);

    kappa_kl = -1.0 * kappa * dt * kappa_kl;
    mu_kl = -2.0 * properties.mu * dt * mu_kl;
    rho_kl = -1.0 * properties.rho * dt * rho_kl;

    const type_real rhop_kl = rho_kl + kappa_kl + mu_kl;

    const type_real beta_kl =
        2.0 * (mu_kl - 4.0 / 3.0 * properties.mu / kappa * kappa_kl);

    const type_real alpha_kl =
        2.0 * (1.0 + 4.0 / 3.0 * properties.mu / kappa) * kappa_kl;

    return { rho_kl, mu_kl, kappa_kl, rhop_kl, alpha_kl, beta_kl };
  } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
    const type_real kappa_kl = 0.0;
    const type_real mu_kl =
        -2.0 * properties.mu * dt * 0.5 *
        (adjoint_derivatives.du(0, 0) * backward_derivatives.du(0, 0) +
         adjoint_derivatives.du(1, 0) * backward_derivatives.du(1, 0));
    const type_real rho_kl =
        -1.0 * properties.rho * dt *
        specfem::algorithms::dot(adjoint_field.acceleration,
                                 backward_field.displacement);

    const type_real rhop_kl = rho_kl + kappa_kl + mu_kl;
    const type_real alpha_kl = 0.0;
    const type_real beta_kl = 2.0 * mu_kl;

    return { rho_kl, mu_kl, kappa_kl, rhop_kl, alpha_kl, beta_kl };
  } else {
    static_assert("Simulation wave not supported");
    return { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  }
}

#endif /* _FRECHET_DERIVATIVES_IMPL_ELEMENT_KERNEL_ELASTIC_ISOTROPIC_TPP */
