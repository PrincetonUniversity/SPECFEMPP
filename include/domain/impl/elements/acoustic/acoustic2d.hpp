#ifndef _DOMAIN_ACOUSTIC_ELEMENTS2D_HPP
#define _DOMAIN_ACOUSTIC_ELEMENTS2D_HPP

#include "../element.hpp"
#include "domain/impl/elements/element.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "point/field_derivatives.hpp"
#include "point/partial_derivatives.hpp"
#include "point/properties.hpp"
#include "point/stress_integrand.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace domain {
namespace impl {
namespace elements {

// Acoustic 2D isotropic specialization
// stress_integrand = rho^{-1} * \sum_{i,k=1}^{2} \partial_i w^{\alpha\gamma}
// \partial_i \chi
template <>
KOKKOS_FUNCTION specfem::point::stress_integrand<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic>
compute_stress_integrands<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic,
                          specfem::element::property_tag::isotropic>(
    const specfem::point::partial_derivatives2<false> &partial_derivatives,
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic> &properties,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic>
        &field_derivatives) {

  const auto &du = field_derivatives.du;
  // Precompute the factor
  type_real fac = properties.rho_inverse;

  specfem::datatype::VectorPointViewType<type_real, 2, 1> F;

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

template <>
KOKKOS_FUNCTION specfem::point::field<specfem::dimension::type::dim2,
                                      specfem::element::medium_tag::acoustic,
                                      false, false, false, true>
mass_matrix_component<specfem::dimension::type::dim2,
                      specfem::element::medium_tag::acoustic,
                      specfem::element::property_tag::isotropic>(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic> &properties,
    const specfem::point::partial_derivatives2<true> &partial_derivatives) {

  return specfem::datatype::ScalarPointViewType<type_real, 1>(
      partial_derivatives.jacobian / properties.kappa);
}

} // namespace elements
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
