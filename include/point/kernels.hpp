#ifndef _SPECFEM_POINT_KERNELS_HPP_
#define _SPECFEM_POINT_KERNELS_HPP_

#include "enumerations/medium.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {
template <specfem::element::medium_tag type,
          specfem::element::property_tag property>
struct kernels;

template <>
struct kernels<specfem::element::medium_tag::elastic,
               specfem::element::property_tag::isotropic> {
  type_real rho;
  type_real mu;
  type_real kappa;
  type_real alpha;
  type_real beta;
  type_real rhop;

  KOKKOS_FUNCTION
  kernels() = default;

  KOKKOS_FUNCTION
  kernels(const type_real rho, const type_real mu, const type_real kappa,
          const type_real rhop, const type_real alpha, const type_real beta)
      : rho(rho), mu(mu), kappa(kappa), rhop(rhop), alpha(alpha), beta(beta) {}
};

template <>
struct kernels<specfem::element::medium_tag::acoustic,
               specfem::element::property_tag::isotropic> {
  type_real rho;
  type_real kappa;
  type_real rho_prime;
  type_real alpha;

  KOKKOS_FUNCTION
  kernels() = default;

  KOKKOS_FUNCTION
  kernels(const type_real rho, const type_real kappa) : rho(rho), kappa(kappa) {
    rho_prime = rho * kappa;
    alpha = 2.0 * kappa;
  }
};

} // namespace point
} // namespace specfem

#endif /* _SPECFEM_POINT_KERNELS_HPP_ */