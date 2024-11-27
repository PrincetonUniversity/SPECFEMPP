#ifndef _POINT_PROPERTIES_HPP
#define _POINT_PROPERTIES_HPP

#include "datatypes/simd.hpp"
#include "enumerations/medium.hpp"

namespace specfem {
namespace point {

/**
 * @brief Store properties of the medium at a quadrature point
 *
 * @tparam DimensionType Dimension of the spectral element
 * @tparam MediumTag Tag indicating the medium of the element
 * @tparam PropertyTag Tag indicating the property of the medium
 * @tparam UseSIMD Boolean indicating whether to use SIMD
 */
template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD>
struct properties;

/**
 * @brief Template specialization for 2D isotropic elastic media
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD
 */
template <bool UseSIMD>
struct properties<specfem::dimension::type::dim2,
                  specfem::element::medium_tag::elastic,
                  specfem::element::property_tag::isotropic, UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  constexpr static bool is_point_properties = true;
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = specfem::element::medium_tag::elastic;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;
  using value_type =
      typename simd::datatype; ///< Value type to store properties
  ///@}

  value_type mu;  ///< shear modulus @f$ \mu @f$
  value_type rho; ///< density @f$ \rho @f$

  value_type rho_vp;        ///< P-wave velocity @f$ \rho v_p @f$
  value_type rho_vs;        ///< S-wave velocity @f$ \rho v_s @f$
  value_type lambda;        ///< Lame's parameter @f$ \lambda @f$
  value_type lambdaplus2mu; ///< Lame's parameter @f$ \lambda + 2\mu @f$

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
  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION
  properties() = default;

  /**
   * @brief Construct a new properties object
   *
   * @param lambdaplus2mu @f$ \lambda + 2\mu @f$
   * @param mu @f$ \mu @f$
   * @param rho @f$ \rho @f$
   */
  KOKKOS_FUNCTION
  properties(const value_type &lambdaplus2mu, const value_type &mu,
             const value_type &rho)
      : properties(lambdaplus2mu, mu, rho,
                   std::integral_constant<bool, UseSIMD>{}) {}
  ///@}
};

/**
 * @brief Template specialization for 2D isotropic acoustic media
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD
 */
template <bool UseSIMD>
struct properties<specfem::dimension::type::dim2,
                  specfem::element::medium_tag::acoustic,
                  specfem::element::property_tag::isotropic, UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  constexpr static bool is_point_properties = true;
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = specfem::element::medium_tag::acoustic;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  using value_type =
      typename simd::datatype; ///< Value type to store properties
  ///@}

  value_type lambdaplus2mu_inverse; ///< @f$ \frac{1}{\lambda + 2\mu} @f$
  value_type rho_inverse;           ///< @f$ \frac{1}{\rho} @f$
  value_type kappa;                 ///< Bulk modulus @f$ \kappa @f$

  value_type rho_vpinverse; ///< @f$ \frac{1}{\rho v_p} @f$

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
  /**
   * @name Constructors
   *
   */
  ///@{

  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION
  properties() = default;

  /**
   * @brief Construct a new properties object
   *
   * @param lambdaplus2mu_inverse @f$ \frac{1}{\lambda + 2\mu} @f$
   * @param rho_inverse @f$ \frac{1}{\rho} @f$
   * @param kappa Bulk modulus @f$ \kappa @f$
   */
  KOKKOS_FUNCTION
  properties(const value_type &lambdaplus2mu_inverse,
             const value_type &rho_inverse, const value_type &kappa)
      : properties(lambdaplus2mu_inverse, rho_inverse, kappa,
                   std::integral_constant<bool, UseSIMD>{}) {}
  ///@}
};

} // namespace point
} // namespace specfem

#endif
