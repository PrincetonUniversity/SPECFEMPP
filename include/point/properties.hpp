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
                  specfem::element::medium_tag::elastic_sv,
                  specfem::element::property_tag::isotropic, UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  constexpr static bool is_point_properties = true;
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = specfem::element::medium_tag::elastic_sv;
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
        rho_vs(Kokkos::sqrt(rho * mu)),
        lambda(lambdaplus2mu - (static_cast<value_type>(2.0)) * mu) {}

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

  /**
   * @brief single value constructor
   *
   */
  KOKKOS_FUNCTION
  properties(const value_type value)
      : properties(value, value, value,
                   std::integral_constant<bool, UseSIMD>{}) {}
  ///@}

  /**
   * @brief Equality operator
   *
   */
  KOKKOS_FUNCTION
  bool operator==(const properties &rhs) const {
    return rho == rhs.rho && mu == rhs.mu && lambdaplus2mu == rhs.lambdaplus2mu;
  }

  /**
   * @brief Inequality operator
   *
   */
  KOKKOS_FUNCTION
  bool operator!=(const properties &rhs) const { return !(*this == rhs); }

  KOKKOS_FUNCTION
  bool operator==(const value_type value) {
    return rho == value && mu == value && lambdaplus2mu == value;
  }

  KOKKOS_FUNCTION
  bool operator!=(const value_type value) { return !(*this == value); }
};

template <bool UseSIMD>
struct properties<specfem::dimension::type::dim2,
                  specfem::element::medium_tag::elastic_sv,
                  specfem::element::property_tag::anisotropic, UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  constexpr static bool is_point_properties = true;
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = specfem::element::medium_tag::elastic_sv;
  constexpr static auto property_tag =
      specfem::element::property_tag::anisotropic;
  using value_type =
      typename simd::datatype; ///< Value type to store properties
  ///@}

  /**
   * @name Elastic constants
   *
   */
  ///@{
  value_type c11; ///< @f$ c_{11} @f$
  value_type c13; ///< @f$ c_{13} @f$
  value_type c15; ///< @f$ c_{15} @f$
  value_type c33; ///< @f$ c_{33} @f$
  value_type c35; ///< @f$ c_{35} @f$
  value_type c55; ///< @f$ c_{55} @f$
  value_type c12; ///< @f$ c_{12} @f$
  value_type c23; ///< @f$ c_{23} @f$
  value_type c25; ///< @f$ c_{25} @f$
  ///@}

  value_type rho;    ///< Density @f$ \rho @f$
  value_type rho_vp; ///< P-wave velocity @f$ \rho v_p @f$
  value_type rho_vs; ///< S-wave velocity @f$ \rho v_s @f$

private:
  KOKKOS_FUNCTION
  properties(const value_type &c11, const value_type &c13,
             const value_type &c15, const value_type &c33,
             const value_type &c35, const value_type &c55,
             const value_type &c12, const value_type &c23,
             const value_type &c25, const value_type &rho, std::false_type)
      : c11(c11), c13(c13), c15(c15), c33(c33), c35(c35), c55(c55), c12(c12),
        c23(c23), c25(c25), rho(rho), rho_vp(sqrt(rho * c33)),
        rho_vs(sqrt(rho * c55)) {}

  KOKKOS_FUNCTION
  properties(const value_type &c11, const value_type &c13,
             const value_type &c15, const value_type &c33,
             const value_type &c35, const value_type &c55,
             const value_type &c12, const value_type &c23,
             const value_type &c25, const value_type &rho, std::true_type)
      : c11(c11), c13(c13), c15(c15), c33(c33), c35(c35), c55(c55), c12(c12),
        c23(c23), c25(c25), rho(rho), rho_vp(Kokkos::sqrt(rho * c33)),
        rho_vs(Kokkos::sqrt(rho * c55)) {}

public:
  KOKKOS_FUNCTION
  properties() = default;

  KOKKOS_FUNCTION
  properties(const value_type &c11, const value_type &c13,
             const value_type &c15, const value_type &c33,
             const value_type &c35, const value_type &c55,
             const value_type &c12, const value_type &c23,
             const value_type &c25, const type_real &rho)
      : properties(c11, c13, c15, c33, c35, c55, c12, c23, c25, rho,
                   std::integral_constant<bool, UseSIMD>{}) {}

  /**
   * @brief single value constructor
   *
   */
  KOKKOS_FUNCTION
  properties(const value_type value)
      : properties(value, value, value, value, value, value, value, value,
                   value, value, std::integral_constant<bool, UseSIMD>{}) {}

  /**
   * @brief Equality operator
   *
   */
  KOKKOS_FUNCTION
  bool operator==(const properties &rhs) const {
    return rho == rhs.rho && c11 == rhs.c11 && c13 == rhs.c13 &&
           c15 == rhs.c15 && c33 == rhs.c33 && c35 == rhs.c35 && c55 == rhs.c55;
  }

  /**
   * @brief Inequality operator
   *
   */
  KOKKOS_FUNCTION
  bool operator!=(const properties &rhs) const { return !(*this == rhs); }

  KOKKOS_FUNCTION
  bool operator==(const value_type value) {
    return rho == value && c11 == value && c13 == value && c15 == value &&
           c33 == value && c35 == value && c55 == value;
  }

  KOKKOS_FUNCTION
  bool operator!=(const value_type value) { return !(*this == value); }
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

  value_type kappa_inverse; ///< @f$ \frac{1}{\lambda + 2\mu} @f$
  value_type rho_inverse;   ///< @f$ \frac{1}{\rho} @f$
  value_type kappa;         ///< Bulk modulus @f$ \kappa @f$

  value_type rho_vpinverse; ///< @f$ \frac{1}{\rho v_p} @f$

private:
  KOKKOS_FUNCTION
  properties(const value_type &rho_inverse, const value_type &kappa,
             std::false_type)
      : kappa_inverse(1.0 / kappa), rho_inverse(rho_inverse), kappa(kappa),
        rho_vpinverse(sqrt(rho_inverse * kappa_inverse)) {}

  KOKKOS_FUNCTION
  properties(const value_type &rho_inverse, const value_type &kappa,
             std::true_type)
      : kappa_inverse((static_cast<value_type>(1.0)) / kappa),
        rho_inverse(rho_inverse), kappa(kappa),
        rho_vpinverse(Kokkos::sqrt(rho_inverse * kappa_inverse)) {}

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
   * @param rho_inverse @f$ \frac{1}{\rho} @f$
   * @param kappa Bulk modulus @f$ \kappa @f$
   */
  KOKKOS_FUNCTION
  properties(const value_type &rho_inverse, const value_type &kappa)
      : properties(rho_inverse, kappa,
                   std::integral_constant<bool, UseSIMD>{}) {}
  /**
   * @brief single value constructor
   *
   */
  KOKKOS_FUNCTION
  properties(const value_type value)
      : properties(value, value, std::integral_constant<bool, UseSIMD>{}) {}
  ///@}

  /**
   * @brief Equality operator
   *
   */
  KOKKOS_FUNCTION
  bool operator==(const properties &rhs) const {
    return rho_inverse == rhs.rho_inverse && kappa == rhs.kappa;
  }

  /**
   * @brief Inequality operator
   *
   */
  KOKKOS_FUNCTION
  bool operator!=(const properties &rhs) const { return !(*this == rhs); }

  KOKKOS_FUNCTION
  bool operator==(const value_type value) {
    return rho_inverse == value && kappa == value;
  }

  KOKKOS_FUNCTION
  bool operator!=(const value_type value) { return !(*this == value); }
};

} // namespace point
} // namespace specfem

#endif
