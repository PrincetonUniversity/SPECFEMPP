#ifndef _FRECHET_DERIVATIVES_IMPL_ELEMENT_KERNEL_ELASTIC_ISOTROPIC_TPP
#define _FRECHET_DERIVATIVES_IMPL_ELEMENT_KERNEL_ELASTIC_ISOTROPIC_TPP

#include "algorithms/dot.hpp"
#include "globals.h"
#include "specfem_setup.hpp"

template <>
KOKKOS_FUNCTION specfem::point::kernels<
    specfem::element::medium_tag::elastic,
    specfem::element::property_tag::isotropic>
specfem::frechet_derivatives::impl::element_kernel<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::property_tag::isotropic>(
    const specfem::point::properties<specfem::element::medium_tag::elastic,
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

  if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
    const type_real ad_dsxx = adjoint_derivatives.du_dx[0];
    const type_real ad_dsxz =
        0.5 * (adjoint_derivatives.du_dx[1] + adjoint_derivatives.du_dz[0]);
    const type_real ad_dszz = adjoint_derivatives.du_dz[1];

    const type_real b_dsxx = backward_derivatives.du_dx[0];
    const type_real b_dsxz =
        0.5 * (backward_derivatives.du_dx[1] + backward_derivatives.du_dz[0]);
    const type_real b_dszz = backward_derivatives.du_dz[1];

    const type_real kappa_kl = (ad_dsxx + ad_dszz) * (b_dsxx + b_dszz);
    const type_real mu_kl = ad_dsxx * b_dsxx + ad_dszz * b_dszz +
                            2.0 * ad_dsxz * b_dsxz - 1.0 / 3.0 * kappa_kl;
    const type_real rho_kl = specfem::algorithms::dot(
        adjoint_field.acceleration, backward_field.displacement);

    return { kappa_kl, mu_kl, rho_kl };
  } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
    const type_real kappa_kl = 0.0;
    const type_real mu_kl =
        0.5 * specfem::algorithms::dot(adjoint_derivatives.du_dx,
                                       backward_derivatives.du_dx);
    const type_real rho_kl = specfem::algorithms::dot(
        adjoint_field.acceleration, backward_field.displacement);

    return { kappa_kl, mu_kl, rho_kl };
  } else {
    static_assert("Simulation wave not supported");
    return { 0.0, 0.0, 0.0 };
  }
}

#endif /* _FRECHET_DERIVATIVES_IMPL_ELEMENT_KERNEL_ELASTIC_ISOTROPIC_TPP */
