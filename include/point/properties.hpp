#pragma once

#include "impl/point_data.hpp"

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
                  specfem::element::property_tag::isotropic, UseSIMD>
    : public impl::point_data<3, UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  using base_type = impl::point_data<3, UseSIMD>;
  using value_type = typename base_type::value_type;

  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = specfem::element::medium_tag::elastic;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  constexpr static bool is_point_properties = true;
  constexpr static int _counter = __COUNTER__;
  ///@}

  using base_type::base_type;

  DEFINE_POINT_VALUE(lambdaplus2mu) ///< Lame's parameter @f$ \lambda + 2\mu @f$
  DEFINE_POINT_VALUE(mu)            ///< shear modulus @f$ \mu @f$
  DEFINE_POINT_VALUE(rho)           ///< density @f$ \rho @f$

  KOKKOS_INLINE_FUNCTION constexpr value_type rho_vp() const {
    return Kokkos::sqrt(rho() * lambdaplus2mu()); ///< P-wave velocity @f$ \rho
                                                  ///< v_p @f$
  }

  KOKKOS_INLINE_FUNCTION constexpr value_type rho_vs() const {
    return Kokkos::sqrt(rho() * mu()); ///< S-wave velocity @f$ \rho v_s @f$
  }

  KOKKOS_INLINE_FUNCTION constexpr value_type lambda() const {
    return lambdaplus2mu() - (static_cast<value_type>(2.0)) *
                                 mu(); ///< Lame's parameter @f$ \lambda @f$
  }
};

template <bool UseSIMD>
struct properties<specfem::dimension::type::dim2,
                  specfem::element::medium_tag::elastic,
                  specfem::element::property_tag::anisotropic, UseSIMD>
    : public impl::point_data<10, UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  using base_type = impl::point_data<10, UseSIMD>;
  using value_type = typename base_type::value_type;

  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = specfem::element::medium_tag::elastic;
  constexpr static auto property_tag =
      specfem::element::property_tag::anisotropic;

  constexpr static bool is_point_properties = true;
  constexpr static int _counter = __COUNTER__;
  ///@}

  using base_type::base_type;

  /**
   * @name Elastic constants
   *
   */
  ///@{
  DEFINE_POINT_VALUE(c11) ///< @f$ c_{11} @f$
  DEFINE_POINT_VALUE(c13) ///< @f$ c_{13} @f$
  DEFINE_POINT_VALUE(c15) ///< @f$ c_{15} @f$
  DEFINE_POINT_VALUE(c33) ///< @f$ c_{33} @f$
  DEFINE_POINT_VALUE(c35) ///< @f$ c_{35} @f$
  DEFINE_POINT_VALUE(c55) ///< @f$ c_{55} @f$
  DEFINE_POINT_VALUE(c12) ///< @f$ c_{12} @f$
  DEFINE_POINT_VALUE(c23) ///< @f$ c_{23} @f$
  DEFINE_POINT_VALUE(c25) ///< @f$ c_{25} @f$
  DEFINE_POINT_VALUE(rho) ///< Density @f$ \rho @f$
  ///@}

  KOKKOS_INLINE_FUNCTION constexpr value_type rho_vp() const {
    return Kokkos::sqrt(rho() * c33()); ///< P-wave velocity @f$ \rho v_p @f$
  }

  KOKKOS_INLINE_FUNCTION constexpr value_type rho_vs() const {
    return Kokkos::sqrt(rho() * c55()); ///< S-wave velocity @f$ \rho v_s @f$
  }
};

/**
 * @brief Template specialization for 2D isotropic acoustic media
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD
 */
template <bool UseSIMD>
struct properties<specfem::dimension::type::dim2,
                  specfem::element::medium_tag::acoustic,
                  specfem::element::property_tag::isotropic, UseSIMD>
    : public impl::point_data<2, UseSIMD> {
  /**
   * @name Typedefs
   *
   */
  ///@{
  using base_type = impl::point_data<2, UseSIMD>;
  using value_type = typename base_type::value_type;

  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = specfem::element::medium_tag::acoustic;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  constexpr static bool is_point_properties = true;
  constexpr static int _counter = __COUNTER__;
  ///@}

  using base_type::base_type;

  DEFINE_POINT_VALUE(rho_inverse) ///< @f$ \frac{1}{\rho} @f$
  DEFINE_POINT_VALUE(kappa)       ///< Bulk modulus @f$ \kappa @f$

  KOKKOS_INLINE_FUNCTION constexpr value_type kappa_inverse() const {
    return (static_cast<value_type>(1.0)) /
           kappa(); ///< @f$ \frac{1}{\lambda + 2\mu} @f$
  }

  KOKKOS_INLINE_FUNCTION constexpr value_type rho_vpinverse() const {
    return Kokkos::sqrt(rho_inverse() * kappa_inverse()); ///< @f$ \frac{1}{\rho
                                                          ///< v_p} @f$
  }
};

} // namespace point
} // namespace specfem
