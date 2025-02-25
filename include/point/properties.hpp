#pragma once

#include "datatypes/simd.hpp"
#include "enumerations/medium.hpp"

namespace specfem {
namespace point {

#define DEFINE_PROP(prop)                                                      \
  constexpr static int i_##prop = __COUNTER__ - _counter - 1;                  \
  KOKKOS_INLINE_FUNCTION constexpr value_type prop() const {                   \
    return Base::data[i_##prop];                                               \
  }                                                                            \
  KOKKOS_INLINE_FUNCTION constexpr void prop(value_type &val) {                \
    Base::data[i_##prop] = val;                                                \
  }

namespace impl {
template <int N, bool UseSIMD> struct impl_properties {
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type
  using value_type = typename simd::datatype;
  constexpr static bool is_point_properties = true;
  constexpr static auto nprops = N;

  value_type data[N];

  KOKKOS_FUNCTION
  impl_properties() = default;

  /**
   * @brief array constructor
   *
   */
  KOKKOS_FUNCTION
  impl_properties(const value_type *value) {
    for (int i = 0; i < N; ++i) {
      data[i] = value[i];
    }
  }

  /**
   * @brief value constructor
   *
   */
  template <typename... Args,
            typename std::enable_if_t<sizeof...(Args) == N, int> = 0>
  impl_properties(Args... args) : data{ args... } {}

  /**
   * @brief Equality operator
   *
   */
  KOKKOS_FUNCTION
  bool operator==(const impl_properties<N, UseSIMD> &rhs) const {
    for (int i = 0; i < N; ++i) {
      if (data[i] != rhs.data[i]) {
        return false;
      }
    }
    return true;
  }

  /**
   * @brief Inequality operator
   *
   */
  KOKKOS_FUNCTION
  bool operator!=(const impl_properties<N, UseSIMD> &rhs) const {
    return !(*this == rhs);
  }
};
} // namespace impl

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
    : public impl::impl_properties<3, UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = specfem::element::medium_tag::elastic;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  using simd = specfem::datatype::simd<type_real, UseSIMD>;
  using value_type = typename simd::datatype;
  constexpr static int _counter = __COUNTER__;
  ///@}
  using Base = impl::impl_properties<3, UseSIMD>;
  using Base::Base;

  DEFINE_PROP(lambdaplus2mu) ///< Lame's parameter @f$ \lambda + 2\mu @f$
  DEFINE_PROP(mu)            ///< shear modulus @f$ \mu @f$
  DEFINE_PROP(rho)           ///< density @f$ \rho @f$

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
    : public impl::impl_properties<10, UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = specfem::element::medium_tag::elastic;
  constexpr static auto property_tag =
      specfem::element::property_tag::anisotropic;

  using simd = specfem::datatype::simd<type_real, UseSIMD>;
  using value_type = typename simd::datatype;
  constexpr static int _counter = __COUNTER__;
  ///@}
  using Base = impl::impl_properties<10, UseSIMD>;
  using Base::Base;

  /**
   * @name Elastic constants
   *
   */
  ///@{
  DEFINE_PROP(c11) ///< @f$ c_{11} @f$
  DEFINE_PROP(c13) ///< @f$ c_{13} @f$
  DEFINE_PROP(c15) ///< @f$ c_{15} @f$
  DEFINE_PROP(c33) ///< @f$ c_{33} @f$
  DEFINE_PROP(c35) ///< @f$ c_{35} @f$
  DEFINE_PROP(c55) ///< @f$ c_{55} @f$
  DEFINE_PROP(c12) ///< @f$ c_{12} @f$
  DEFINE_PROP(c23) ///< @f$ c_{23} @f$
  DEFINE_PROP(c25) ///< @f$ c_{25} @f$
  DEFINE_PROP(rho) ///< Density @f$ \rho @f$
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
    : public impl::impl_properties<2, UseSIMD> {
  /**
   * @name Typedefs
   *
   */
  ///@{
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = specfem::element::medium_tag::acoustic;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  using simd = specfem::datatype::simd<type_real, UseSIMD>;
  using value_type = typename simd::datatype;
  constexpr static int _counter = __COUNTER__;
  ///@}
  using Base = impl::impl_properties<2, UseSIMD>;
  using Base::Base;

  DEFINE_PROP(rho_inverse) ///< @f$ \frac{1}{\rho} @f$
  DEFINE_PROP(kappa)       ///< Bulk modulus @f$ \kappa @f$

  KOKKOS_INLINE_FUNCTION constexpr value_type kappa_inverse() const {
    return (static_cast<value_type>(1.0)) /
           kappa(); ///< @f$ \frac{1}{\lambda + 2\mu} @f$
  }

  KOKKOS_INLINE_FUNCTION constexpr value_type rho_vpinverse() const {
    return Kokkos::sqrt(rho_inverse() * kappa_inverse()); ///< @f$ \frac{1}{\rho
                                                          ///< v_p} @f$
  }
};

#undef DEFINE_PROP

} // namespace point
} // namespace specfem
