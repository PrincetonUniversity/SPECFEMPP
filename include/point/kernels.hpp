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

  KOKKOS_FUNCTION
  kernels() = default;

  KOKKOS_FUNCTION
  kernels(const type_real rho, const type_real mu, const type_real kappa)
      : rho(rho), mu(mu), kappa(kappa) {}
};

template <>
struct kernels<specfem::element::medium_tag::acoustic,
               specfem::element::property_tag::isotropic> {
  type_real rho;
  type_real kappa;

  KOKKOS_FUNCTION
  kernels() = default;

  KOKKOS_FUNCTION
  kernels(const type_real rho, const type_real kappa)
      : rho(rho), kappa(kappa) {}
};

} // namespace point
} // namespace specfem

#endif /* _SPECFEM_POINT_KERNELS_HPP_ */
