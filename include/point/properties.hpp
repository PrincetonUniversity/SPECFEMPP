#pragma once

#include "datatypes/simd.hpp"
#include "enumerations/medium.hpp"

namespace specfem {
namespace point {

#define DEFINE_PROP_CONSTRUCTORS                                               \
  KOKKOS_FUNCTION                                                              \
  properties() = default;                                                      \
  KOKKOS_FUNCTION                                                              \
  properties(const value_type *value) : Base(value) { compute(); }             \
  template <typename... Args,                                                  \
            typename std::enable_if_t<sizeof...(Args) == Base::nprops, int> =  \
                0>                                                             \
  KOKKOS_FUNCTION properties(Args... args) : Base(args...) {                   \
    compute();                                                                 \
  }

namespace impl {
template <int N, int NALL, bool UseSIMD> struct impl_properties {
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type
  using value_type = typename simd::datatype;
  constexpr static auto nprops = N;
  constexpr static auto nprops_all = NALL;

  value_type data[NALL];

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
  bool operator==(const impl_properties<N, NALL, UseSIMD> &rhs) const {
    for (int i = 0; i < NALL; ++i) {
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
  bool operator!=(const impl_properties<N, NALL, UseSIMD> &rhs) const {
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
    : public impl::impl_properties<3, 6, UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  constexpr static bool is_point_properties = true;
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type
  using Base = impl::impl_properties<3, 6, UseSIMD>;
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = specfem::element::medium_tag::elastic;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;
  using value_type =
      typename simd::datatype; ///< Value type to store properties
  ///@}

  value_type &lambdaplus2mu =
      Base::data[0];               ///< Lame's parameter @f$ \lambda + 2\mu @f$
  value_type &mu = Base::data[1];  ///< shear modulus @f$ \mu @f$
  value_type &rho = Base::data[2]; ///< density @f$ \rho @f$

  value_type &rho_vp = Base::data[3]; ///< P-wave velocity @f$ \rho v_p @f$
  value_type &rho_vs = Base::data[4]; ///< S-wave velocity @f$ \rho v_s @f$
  value_type &lambda = Base::data[5]; ///< Lame's parameter @f$ \lambda @f$

  DEFINE_PROP_CONSTRUCTORS

  KOKKOS_INLINE_FUNCTION
  void compute() {
    rho_vp = Kokkos::sqrt(rho * lambdaplus2mu);
    rho_vs = Kokkos::sqrt(rho * mu);
    lambda = lambdaplus2mu - (static_cast<value_type>(2.0)) * mu;
  }
};

template <bool UseSIMD>
struct properties<specfem::dimension::type::dim2,
                  specfem::element::medium_tag::elastic,
                  specfem::element::property_tag::anisotropic, UseSIMD>
    : public impl::impl_properties<10, 12, UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  constexpr static bool is_point_properties = true;
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type
  using Base = impl::impl_properties<10, 12, UseSIMD>;
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = specfem::element::medium_tag::elastic;
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
  value_type &c11 = Base::data[0]; ///< @f$ c_{11} @f$
  value_type &c13 = Base::data[1]; ///< @f$ c_{13} @f$
  value_type &c15 = Base::data[2]; ///< @f$ c_{15} @f$
  value_type &c33 = Base::data[3]; ///< @f$ c_{33} @f$
  value_type &c35 = Base::data[4]; ///< @f$ c_{35} @f$
  value_type &c55 = Base::data[5]; ///< @f$ c_{55} @f$
  value_type &c12 = Base::data[6]; ///< @f$ c_{12} @f$
  value_type &c23 = Base::data[7]; ///< @f$ c_{23} @f$
  value_type &c25 = Base::data[8]; ///< @f$ c_{25} @f$
  ///@}

  value_type &rho = Base::data[9];     ///< Density @f$ \rho @f$
  value_type &rho_vp = Base::data[10]; ///< P-wave velocity @f$ \rho v_p @f$
  value_type &rho_vs = Base::data[11]; ///< S-wave velocity @f$ \rho v_s @f$

  DEFINE_PROP_CONSTRUCTORS

  KOKKOS_INLINE_FUNCTION
  void compute() {
    rho_vp = Kokkos::sqrt(rho * c33);
    rho_vs = Kokkos::sqrt(rho * c55);
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
    : public impl::impl_properties<2, 4, UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  constexpr static bool is_point_properties = true;
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type
  using Base = impl::impl_properties<2, 4, UseSIMD>;
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = specfem::element::medium_tag::acoustic;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  using value_type =
      typename simd::datatype; ///< Value type to store properties
  ///@}

  value_type &rho_inverse = Base::data[0]; ///< @f$ \frac{1}{\rho} @f$
  value_type &kappa = Base::data[1];       ///< Bulk modulus @f$ \kappa @f$

  value_type &kappa_inverse =
      Base::data[2]; ///< @f$ \frac{1}{\lambda + 2\mu} @f$
  value_type &rho_vpinverse = Base::data[3]; ///< @f$ \frac{1}{\rho v_p} @f$

  DEFINE_PROP_CONSTRUCTORS

  KOKKOS_INLINE_FUNCTION
  void compute() {
    kappa_inverse = (static_cast<value_type>(1.0)) / kappa;
    rho_vpinverse = Kokkos::sqrt(rho_inverse * kappa_inverse);
  }
};

#undef DEFINE_PROP_CONSTRUCTORS

} // namespace point
} // namespace specfem
