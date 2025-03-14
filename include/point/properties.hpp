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
template <specfem::element::medium_tag MediumTag, bool UseSIMD>
struct properties<specfem::dimension::type::dim2, MediumTag,
                  specfem::element::property_tag::isotropic, UseSIMD>
    : public impl::point_data<3, UseSIMD>,
      specfem::element::is_elastic<MediumTag> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  using base_type = impl::point_data<3, UseSIMD>;
  using value_type = typename base_type::value_type;

  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  constexpr static bool is_point_properties = true;
  ///@}

  using base_type::base_type;

  DEFINE_POINT_VALUE(lambdaplus2mu,
                     0)      ///< Lame's parameter @f$ \lambda + 2\mu @f$
  DEFINE_POINT_VALUE(mu, 1)  ///< shear modulus @f$ \mu @f$
  DEFINE_POINT_VALUE(rho, 2) ///< density @f$ \rho @f$

  KOKKOS_INLINE_FUNCTION const value_type rho_vp() const {
    return Kokkos::sqrt(rho() * lambdaplus2mu()); ///< P-wave velocity @f$ \rho
                                                  ///< v_p @f$
  }

  KOKKOS_INLINE_FUNCTION const value_type rho_vs() const {
    return Kokkos::sqrt(rho() * mu()); ///< S-wave velocity @f$ \rho v_s @f$
  }

  KOKKOS_INLINE_FUNCTION const value_type lambda() const {
    return lambdaplus2mu() - (static_cast<value_type>(2.0)) *
                                 mu(); ///< Lame's parameter @f$ \lambda @f$
  }
};

template <specfem::element::medium_tag MediumTag, bool UseSIMD>
struct properties<specfem::dimension::type::dim2, MediumTag,
                  specfem::element::property_tag::anisotropic, UseSIMD>
    : public impl::point_data<10, UseSIMD>,
      specfem::element::is_elastic<MediumTag> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  using base_type = impl::point_data<10, UseSIMD>;
  using value_type = typename base_type::value_type;

  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto property_tag =
      specfem::element::property_tag::anisotropic;

  constexpr static bool is_point_properties = true;
  ///@}

  using base_type::base_type;

  /**
   * @name Elastic constants
   *
   */
  ///@{
  DEFINE_POINT_VALUE(c11, 0) ///< @f$ c_{11} @f$
  DEFINE_POINT_VALUE(c13, 1) ///< @f$ c_{13} @f$
  DEFINE_POINT_VALUE(c15, 2) ///< @f$ c_{15} @f$
  DEFINE_POINT_VALUE(c33, 3) ///< @f$ c_{33} @f$
  DEFINE_POINT_VALUE(c35, 4) ///< @f$ c_{35} @f$
  DEFINE_POINT_VALUE(c55, 5) ///< @f$ c_{55} @f$
  DEFINE_POINT_VALUE(c12, 6) ///< @f$ c_{12} @f$
  DEFINE_POINT_VALUE(c23, 7) ///< @f$ c_{23} @f$
  DEFINE_POINT_VALUE(c25, 8) ///< @f$ c_{25} @f$
  DEFINE_POINT_VALUE(rho, 9) ///< Density @f$ \rho @f$
  ///@}

  KOKKOS_INLINE_FUNCTION const value_type rho_vp() const {
    return Kokkos::sqrt(rho() * c33()); ///< P-wave velocity @f$ \rho v_p @f$
  }

  KOKKOS_INLINE_FUNCTION const value_type rho_vs() const {
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
  ///@}

  using base_type::base_type;

  DEFINE_POINT_VALUE(rho_inverse, 0) ///< @f$ \frac{1}{\rho} @f$
  DEFINE_POINT_VALUE(kappa, 1)       ///< Bulk modulus @f$ \kappa @f$

  KOKKOS_INLINE_FUNCTION const value_type kappa_inverse() const {
    return (static_cast<value_type>(1.0)) /
           kappa(); ///< @f$ \frac{1}{\lambda + 2\mu} @f$
  }

  KOKKOS_INLINE_FUNCTION const value_type rho_vpinverse() const {
    return Kokkos::sqrt(rho_inverse() * kappa_inverse()); ///< @f$ \frac{1}{\rho
                                                          ///< v_p} @f$
  }
};

} // namespace point
} // namespace specfem
