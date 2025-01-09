#pragma once

#include "stress_integrand.hpp"

template <bool UseSIMD>
KOKKOS_FUNCTION KOKKOS_FUNCTION specfem::point::stress_integrand<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    UseSIMD>
specfem::medium::impl_compute_stress_integrands(
    const specfem::point::partial_derivatives<
        specfem::dimension::type::dim2, false, UseSIMD> &partial_derivatives,
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, UseSIMD> &properties,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        UseSIMD> &field_derivatives) {

  const auto &du = field_derivatives.du;
  // Precompute the factor
  auto fac = properties.rho_inverse;

  specfem::datatype::VectorPointViewType<type_real, 2, 1, UseSIMD> F;

  // Compute stress integrands 1 and 2
  // Here it is extremely important that this seems at odds with
  // equations (44) & (45) from Komatitsch and Tromp 2002 I. - Validation
  // The equations are however missing dxi/dx, dxi/dz, dzeta/dx, dzeta/dz
  // for the gradient of w^{\alpha\gamma}. In this->update_acceleration
  // the weights for the integration and the interpolated values for the
  // first derivatives of the lagrange polynomials are then collapsed
  F(0, 0) = fac * (partial_derivatives.xix * du(0, 0) +
                   partial_derivatives.xiz * du(1, 0));
  F(1, 0) = fac * (partial_derivatives.gammax * du(0, 0) +
                   partial_derivatives.gammaz * du(1, 0));

  return { F };
}
