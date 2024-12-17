#pragma once
#include "acoustic_isotropic.hpp"
#include "algorithms/dot.hpp"
#include "specfem_setup.hpp"

template <bool UseSIMD>
KOKKOS_FUNCTION specfem::point::kernels<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::property_tag::isotropic, UseSIMD>
specfem::frechet_derivatives::impl::impl_compute_element_kernel(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, UseSIMD> &properties,
    const specfem::point::field<specfem::dimension::type::dim2,
                                specfem::element::medium_tag::acoustic, false,
                                false, true, false, UseSIMD> &adjoint_field,
    const specfem::point::field<specfem::dimension::type::dim2,
                                specfem::element::medium_tag::acoustic, true,
                                false, false, false, UseSIMD> &backward_field,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        UseSIMD> &adjoint_derivatives,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        UseSIMD> &backward_derivatives,
    const type_real &dt) {

  using datatype =
      typename specfem::datatype::simd<type_real, UseSIMD>::datatype;

  const datatype rho_kl =
      (adjoint_derivatives.du(0, 0) * backward_derivatives.du(0, 0) +
       adjoint_derivatives.du(1, 0) * backward_derivatives.du(1, 0)) *
      properties.rho_inverse * dt;

  const datatype kappa_kl =
      specfem::algorithms::dot(adjoint_field.acceleration,
                               backward_field.displacement) *
      static_cast<type_real>(1.0) / properties.kappa * dt;

  return { rho_kl, kappa_kl };
}
