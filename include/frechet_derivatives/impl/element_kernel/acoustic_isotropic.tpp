#ifndef SPECFEM_FRECHET_DERIVATIVES_IMPL_ELEMENT_KERNEL_ACOUSTIC_ISOTROPIC_TPP
#define SPECFEM_FRECHET_DERIVATIVES_IMPL_ELEMENT_KERNEL_ACOUSTIC_ISOTROPIC_TPP

#include "algorithms/dot.hpp"
#include "specfem_setup.hpp"
#include "element_kernel.hpp"

template <>
KOKKOS_FUNCTION specfem::point::kernels<
    specfem::element::medium_tag::acoustic,
    specfem::element::property_tag::isotropic>
specfem::frechet_derivatives::impl::element_kernel<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::property_tag::isotropic>(
    const specfem::point::properties<specfem::element::medium_tag::acoustic,
                                     specfem::element::property_tag::isotropic>
        &properties,
    const specfem::frechet_derivatives::impl::AdjointPointFieldType<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic>
        &adjoint_field,
    const specfem::frechet_derivatives::impl::BackwardPointFieldType<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic>
        &backward_field,
    const specfem::frechet_derivatives::impl::PointFieldDerivativesType<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic>
        &adjoint_derivatives,
    const specfem::frechet_derivatives::impl::PointFieldDerivativesType<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic>
        &backward_derivatives,
    const type_real &dt) {

  const type_real rho_kl =
      (adjoint_derivatives.du_dx[0] * backward_derivatives.du_dx[0] +
       adjoint_derivatives.du_dz[0] * backward_derivatives.du_dz[0]) *
      properties.rho_inverse * dt;

  const type_real kappa_kl =
      specfem::algorithms::dot(adjoint_field.acceleration,
                               backward_field.displacement) *
      1.0 / properties.kappa * dt;

  return { rho_kl, kappa_kl };
}

#endif /* _FRECHET_DERIVATIVES_IMPL_ELEMENT_KERNEL_ACOUSTIC_ISOTROPIC_TPP */
