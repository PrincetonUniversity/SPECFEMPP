#ifndef _POINT_PROPERTIES_HPP
#define _POINT_PROPERTIES_HPP

#include "enumerations/medium.hpp"

namespace specfem {
namespace point {

template <specfem::element::medium_tag medium,
          specfem::element::property_tag property>
struct properties {};

template <>
struct properties<specfem::element::medium_tag::elastic,
                  specfem::element::property_tag::isotropic> {
  type_real lambdaplus2mu;
  type_real mu;
  type_real rho;
  type_real lambda;

  type_real rho_vp;
  type_real rho_vs;

  KOKKOS_FUNCTION
  properties() = default;

  KOKKOS_FUNCTION
  properties(const type_real &lambdaplus2mu, const type_real &mu,
             const type_real &rho)
      : lambdaplus2mu(lambdaplus2mu), mu(mu), rho(rho),
        lambda(lambdaplus2mu - 2 * mu), rho_vp(sqrt(rho * lambdaplus2mu)),
        rho_vs(sqrt(rho * mu)) {}
};

template <>
struct properties<specfem::element::medium_tag::acoustic,
                  specfem::element::property_tag::isotropic> {
  type_real lambdaplus2mu_inverse;
  type_real rho_inverse;
  type_real kappa;

  type_real rho_vpinverse;

  KOKKOS_FUNCTION
  properties() = default;

  KOKKOS_FUNCTION
  properties(const type_real &lambdaplus2mu_inverse,
             const type_real &rho_inverse, const type_real &kappa)
      : lambdaplus2mu_inverse(lambdaplus2mu_inverse), rho_inverse(rho_inverse),
        kappa(kappa), rho_vpinverse(sqrt(rho_inverse * lambdaplus2mu_inverse)) {
  }
};

} // namespace point
} // namespace specfem

#endif
