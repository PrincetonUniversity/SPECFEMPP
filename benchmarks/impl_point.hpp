#pragma once

#include "datatypes/simd.hpp"
#include "enumerations/medium.hpp"

namespace specfem {
namespace benchmarks {

#define DEFINE_PROP_CONSTRUCTORS                                               \
  using simd = specfem::datatype::simd<type_real, UseSIMD>;                    \
  using value_type = typename simd::datatype;                                  \
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

#define DEFINE_PROP(prop, i)                                                   \
  KOKKOS_INLINE_FUNCTION value_type prop() const { return Base::data[i]; }     \
  KOKKOS_INLINE_FUNCTION void prop(value_type val) { Base::data[i] = val; }

namespace impl {
template <int N, int N_EX, bool UseSIMD> struct impl_properties {
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type
  using value_type = typename simd::datatype;
  constexpr static bool is_point_properties = true;
  constexpr static auto nprops = N;
  constexpr static auto nprops_extra = N_EX;

  value_type data[N_EX];

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
  bool operator==(const impl_properties<N, N_EX, UseSIMD> &rhs) const {
    for (int i = 0; i < N_EX; ++i) {
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
  bool operator!=(const impl_properties<N, N_EX, UseSIMD> &rhs) const {
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
  using Base = impl::impl_properties<3, 6, UseSIMD>;
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = specfem::element::medium_tag::elastic;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;
  ///@}

  DEFINE_PROP_CONSTRUCTORS

  DEFINE_PROP(lambdaplus2mu, 0) ///< Lame's parameter @f$ \lambda + 2\mu @f$
  DEFINE_PROP(mu, 1)            ///< shear modulus @f$ \mu @f$
  DEFINE_PROP(rho, 2)           ///< density @f$ \rho @f$

  DEFINE_PROP(rho_vp, 3) ///< P-wave velocity @f$ \rho v_p @f$
  DEFINE_PROP(rho_vs, 4) ///< S-wave velocity @f$ \rho v_s @f$
  DEFINE_PROP(lambda, 5) ///< Lame's parameter @f$ \lambda @f$

  KOKKOS_INLINE_FUNCTION
  void compute() {
    rho_vp(Kokkos::sqrt(rho() * lambdaplus2mu()));
    rho_vs(Kokkos::sqrt(rho() * mu()));
    lambda(lambdaplus2mu() - (static_cast<value_type>(2.0)) * mu());
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
  using Base = impl::impl_properties<10, 12, UseSIMD>;
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = specfem::element::medium_tag::elastic;
  constexpr static auto property_tag =
      specfem::element::property_tag::anisotropic;
  ///@}

  DEFINE_PROP_CONSTRUCTORS

  /**
   * @name Elastic constants
   *
   */
  ///@{
  DEFINE_PROP(c11, 0) ///< @f$ c_{11} @f$
  DEFINE_PROP(c13, 1) ///< @f$ c_{13} @f$
  DEFINE_PROP(c15, 2) ///< @f$ c_{15} @f$
  DEFINE_PROP(c33, 3) ///< @f$ c_{33} @f$
  DEFINE_PROP(c35, 4) ///< @f$ c_{35} @f$
  DEFINE_PROP(c55, 5) ///< @f$ c_{55} @f$
  DEFINE_PROP(c12, 6) ///< @f$ c_{12} @f$
  DEFINE_PROP(c23, 7) ///< @f$ c_{23} @f$
  DEFINE_PROP(c25, 8) ///< @f$ c_{25} @f$
  ///@}

  DEFINE_PROP(rho, 9)     ///< Density @f$ \rho @f$
  DEFINE_PROP(rho_vp, 10) ///< P-wave velocity @f$ \rho v_p @f$
  DEFINE_PROP(rho_vs, 11) ///< S-wave velocity @f$ \rho v_s @f$

  KOKKOS_INLINE_FUNCTION
  void compute() {
    rho_vp(Kokkos::sqrt(rho() * c33()));
    rho_vs(Kokkos::sqrt(rho() * c55()));
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
  using Base = impl::impl_properties<2, 4, UseSIMD>;
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = specfem::element::medium_tag::acoustic;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;
  ///@}

  DEFINE_PROP_CONSTRUCTORS

  DEFINE_PROP(rho_inverse, 0)   ///< @f$ \frac{1}{\rho} @f$
  DEFINE_PROP(kappa, 1)         ///< Bulk modulus @f$ \kappa @f$
  DEFINE_PROP(kappa_inverse, 2) ///< @f$ \frac{1}{\lambda + 2\mu} @f$
  DEFINE_PROP(rho_vpinverse, 3) ///< @f$ \frac{1}{\rho v_p} @f$

private:
  KOKKOS_INLINE_FUNCTION
  void compute() {
    kappa_inverse((static_cast<value_type>(1.0)) / kappa());
    rho_vpinverse(Kokkos::sqrt(rho_inverse() * kappa_inverse()));
  }
};

#undef DEFINE_PROP_CONSTRUCTORS
#undef DEFINE_PROP

} // namespace benchmarks
} // namespace specfem
