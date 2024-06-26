#ifndef _DOMAIN_ELASTIC_ELEMENTS2D_HPP
#define _DOMAIN_ELASTIC_ELEMENTS2D_HPP

#include "../element.hpp"
#include "domain/impl/elements/element.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "globals.h"
#include "point/field_derivatives.hpp"
#include "point/partial_derivatives.hpp"
#include "point/properties.hpp"
#include "point/stress_integrand.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace domain {
namespace impl {
namespace elements {

// Elastic 2D isotropic specialization
// stress_integrand = \sum_{i,k=1}^{2} F_{ik} \partial_i w^{\alpha\gamma}
template <>
KOKKOS_FUNCTION specfem::point::stress_integrand<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic>
compute_stress_integrands<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic,
                          specfem::element::property_tag::isotropic>(
    const specfem::point::partial_derivatives2<false> &partial_derivatives,
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic> &properties,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic>
        &field_derivatives) {

  const auto &du = field_derivatives.du;

  type_real sigma_xx, sigma_zz, sigma_xz;

  if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
    // P_SV case
    // sigma_xx
    sigma_xx =
        properties.lambdaplus2mu * du(0, 0) + properties.lambda * du(1, 1);

    // sigma_zz
    sigma_zz =
        properties.lambdaplus2mu * du(1, 1) + properties.lambda * du(0, 0);

    // sigma_xz
    sigma_xz = properties.mu * (du(0, 1) + du(1, 0));
  } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
    // SH-case
    // sigma_xx
    sigma_xx = properties.mu * du(0, 0); // would be sigma_xy in
                                         // CPU-version

    // sigma_xz
    sigma_xz = properties.mu * du(1, 0); // sigma_zy
  }

  specfem::datatype::VectorPointViewType<type_real, 2, 2> F;

  F(0, 0) =
      sigma_xx * partial_derivatives.xix + sigma_xz * partial_derivatives.xiz;
  F(0, 1) =
      sigma_xz * partial_derivatives.xix + sigma_zz * partial_derivatives.xiz;
  F(1, 0) = sigma_xx * partial_derivatives.gammax +
            sigma_xz * partial_derivatives.gammaz;
  F(1, 1) = sigma_xz * partial_derivatives.gammax +
            sigma_zz * partial_derivatives.gammaz;

  return { F };
}

template <>
KOKKOS_FUNCTION specfem::point::field<specfem::dimension::type::dim2,
                                      specfem::element::medium_tag::elastic,
                                      false, false, false, true>
mass_matrix_component<specfem::dimension::type::dim2,
                      specfem::element::medium_tag::elastic,
                      specfem::element::property_tag::isotropic>(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic> &properties,
    const specfem::point::partial_derivatives2<true> &partial_derivatives) {

  if constexpr (specfem::globals::simulation_wave == specfem::wave::p_sv) {
    return specfem::datatype::ScalarPointViewType<type_real, 2>(
        partial_derivatives.jacobian * properties.rho,
        partial_derivatives.jacobian * properties.rho);
  } else if constexpr (specfem::globals::simulation_wave == specfem::wave::sh) {
    return specfem::datatype::ScalarPointViewType<type_real, 2>(
        partial_derivatives.jacobian * properties.rho, 0);
  } else {
    static_assert("Unknown wave type");
    return specfem::datatype::ScalarPointViewType<type_real, 2>(
        0.0, 0.0); // dummy return
  }
}
} // namespace elements
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
