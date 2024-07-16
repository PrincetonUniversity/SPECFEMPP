#ifndef _SPECFEM_POINT_KERNELS_HPP_
#define _SPECFEM_POINT_KERNELS_HPP_

#include "datatypes/simd.hpp"
#include "enumerations/medium.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {
template <specfem::element::medium_tag type,
          specfem::element::property_tag property, bool UseSIMD>
struct kernels;

template <bool UseSIMD>
struct kernels<specfem::element::medium_tag::elastic,
               specfem::element::property_tag::isotropic, UseSIMD> {
public:
  using simd = typename specfem::datatype::simd<type_real, UseSIMD>;

private:
  using value_type = typename simd::datatype;

public:
  constexpr static auto medium_tag = specfem::element::medium_tag::elastic;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  value_type rho;
  value_type mu;
  value_type kappa;
  value_type alpha;
  value_type beta;
  value_type rhop;

  KOKKOS_FUNCTION
  kernels() = default;

  KOKKOS_FUNCTION
  kernels(const value_type rho, const value_type mu, const value_type kappa,
          const value_type rhop, const value_type alpha, const value_type beta)
      : rho(rho), mu(mu), kappa(kappa), rhop(rhop), alpha(alpha), beta(beta) {}
};

template <bool UseSIMD>
struct kernels<specfem::element::medium_tag::acoustic,
               specfem::element::property_tag::isotropic, UseSIMD> {
public:
  using simd = typename specfem::datatype::simd<type_real, UseSIMD>;

  constexpr static auto medium_tag = specfem::element::medium_tag::acoustic;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

private:
  using value_type = typename simd::datatype;

public:
  value_type rho;
  value_type kappa;
  value_type rho_prime;
  value_type alpha;

  KOKKOS_FUNCTION
  kernels() = default;

  KOKKOS_FUNCTION
  kernels(const value_type rho, const value_type kappa)
      : rho(rho), kappa(kappa) {
    rho_prime = rho * kappa;
    alpha = static_cast<type_real>(2.0) * kappa;
  }
};

} // namespace point
} // namespace specfem

#endif /* _SPECFEM_POINT_KERNELS_HPP_ */
