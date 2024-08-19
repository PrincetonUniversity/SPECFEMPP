#ifndef _POINT_PROPERTIES_HPP
#define _POINT_PROPERTIES_HPP

#include "datatypes/simd.hpp"
#include "enumerations/medium.hpp"

namespace specfem {
namespace point {

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD>
struct properties {};

template <bool UseSIMD>
struct properties<specfem::dimension::type::dim2,
                  specfem::element::medium_tag::elastic,
                  specfem::element::property_tag::isotropic, UseSIMD> {

  using simd = specfem::datatype::simd<type_real, UseSIMD>;
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = specfem::element::medium_tag::elastic;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  using value_type = typename simd::datatype;

  value_type lambdaplus2mu;
  value_type mu;
  value_type rho;

  value_type rho_vp;
  value_type rho_vs;
  value_type lambda;

private:
  KOKKOS_FUNCTION
  properties(const value_type &lambdaplus2mu, const value_type &mu,
             const value_type &rho, std::false_type)
      : lambdaplus2mu(lambdaplus2mu), mu(mu), rho(rho),
        rho_vp(sqrt(rho * lambdaplus2mu)), rho_vs(sqrt(rho * mu)),
        lambda(lambdaplus2mu - 2.0 * mu) {}

  KOKKOS_FUNCTION
  properties(const value_type &lambdaplus2mu, const value_type &mu,
             const value_type &rho, std::true_type)
      : lambdaplus2mu(lambdaplus2mu), mu(mu), rho(rho),
        rho_vp(Kokkos::sqrt(rho * lambdaplus2mu)),
        rho_vs(Kokkos::sqrt(rho * mu)), lambda(lambdaplus2mu - 2.0 * mu) {}

public:
  KOKKOS_FUNCTION
  properties() = default;

  KOKKOS_FUNCTION
  properties(const value_type &lambdaplus2mu, const value_type &mu,
             const value_type &rho)
      : properties(lambdaplus2mu, mu, rho,
                   std::integral_constant<bool, UseSIMD>{}) {}
};

template <bool UseSIMD>
struct properties<specfem::dimension::type::dim2,
                  specfem::element::medium_tag::acoustic,
                  specfem::element::property_tag::isotropic, UseSIMD> {

  using simd = specfem::datatype::simd<type_real, UseSIMD>;
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = specfem::element::medium_tag::acoustic;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  using value_type = typename simd::datatype;

  value_type lambdaplus2mu_inverse;
  value_type rho_inverse;
  value_type kappa;

  value_type rho_vpinverse;

private:
  KOKKOS_FUNCTION
  properties(const value_type &lambdaplus2mu_inverse,
             const value_type &rho_inverse, const value_type &kappa,
             std::false_type)
      : lambdaplus2mu_inverse(lambdaplus2mu_inverse), rho_inverse(rho_inverse),
        kappa(kappa), rho_vpinverse(sqrt(rho_inverse * lambdaplus2mu_inverse)) {
  }

  KOKKOS_FUNCTION
  properties(const value_type &lambdaplus2mu_inverse,
             const value_type &rho_inverse, const value_type &kappa,
             std::true_type)
      : lambdaplus2mu_inverse(lambdaplus2mu_inverse), rho_inverse(rho_inverse),
        kappa(kappa),
        rho_vpinverse(Kokkos::sqrt(rho_inverse * lambdaplus2mu_inverse)) {}

public:
  KOKKOS_FUNCTION
  properties() = default;

  KOKKOS_FUNCTION
  properties(const value_type &lambdaplus2mu_inverse,
             const value_type &rho_inverse, const value_type &kappa)
      : properties(lambdaplus2mu_inverse, rho_inverse, kappa,
                   std::integral_constant<bool, UseSIMD>{}) {}
};

} // namespace point
} // namespace specfem

#endif
