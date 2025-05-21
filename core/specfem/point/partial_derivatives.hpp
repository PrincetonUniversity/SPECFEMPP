#pragma once

#include "datatypes/point_view.hpp"
#include "enumerations/accessor.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/specfem_enums.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {

/**
 * @brief Store spatial derivatives of the basis functions at a quadrature point
 *
 * @tparam DimensionTag Dimension of the spectral element
 * @tparam StoreJacobian Boolean indicating whether to store the Jacobian
 * @tparam UseSIMD Boolean indicating whether to use SIMD
 */
template <specfem::dimension::type DimensionTag, bool StoreJacobian,
          bool UseSIMD>
struct partial_derivatives;

/**
 * @brief Template specialization for 2D spectral elements without storing the
 * Jacobian
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD
 */
template <bool UseSIMD>
struct partial_derivatives<specfem::dimension::type::dim2, false, UseSIMD>
    : public specfem::accessor::Accessor<
          specfem::accessor::type::point,
          specfem::data_class::type::partial_derivatives,
          specfem::dimension::type::dim2, UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  using self_type =
      partial_derivatives<specfem::dimension::type::dim2, false,
                          UseSIMD>; ///< Type of the point partial derivatives;
  using base_type = specfem::accessor::Accessor<
      specfem::accessor::type::point,
      specfem::data_class::type::partial_derivatives,
      specfem::dimension::type::dim2, UseSIMD>; ///< Base type of the point
                                                ///< partial derivatives
  using simd = typename base_type::simd;        ///< SIMD data type
  using value_type = typename base_type::template scalar_type<type_real>;
  constexpr static bool store_jacobian = false;
  ///@}

  value_type xix;    ///< @xix
  value_type gammax; ///< @gammax
  value_type xiz;    ///< @xiz
  value_type gammaz; ///< @gammaz

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
  partial_derivatives() = default;

  /**
   * @brief Constructor with values
   *
   * @param xix @xix
   * @param gammax @gammax
   * @param xiz @xiz
   * @param gammaz @gammaz
   */
  KOKKOS_FUNCTION
  partial_derivatives(const value_type &xix, const value_type &gammax,
                      const value_type &xiz, const value_type &gammaz)
      : xix(xix), gammax(gammax), xiz(xiz), gammaz(gammaz) {}

  /**
   * @brief Constructor with constant value
   *
   * @param constant Value to initialize all members to
   */
  KOKKOS_FUNCTION
  partial_derivatives(const value_type constant)
      : xix(constant), gammax(constant), xiz(constant), gammaz(constant) {}

  KOKKOS_FUNCTION
  void init() {
    this->xix = 0.0;
    this->gammax = 0.0;
    this->xiz = 0.0;
    this->gammaz = 0.0;
    return;
  }

  // operator+
  KOKKOS_FUNCTION self_type operator+(const self_type &rhs) const {
    return self_type(xix + rhs.xix, gammax + rhs.gammax, xiz + rhs.xiz,
                     gammaz + rhs.gammaz);
  }

  // operator+=
  KOKKOS_FUNCTION self_type &operator+=(const self_type &rhs) {
    xix += rhs.xix;
    gammax += rhs.gammax;
    xiz += rhs.xiz;
    gammaz += rhs.gammaz;
    return *this;
  }

  // operator*
  KOKKOS_FUNCTION self_type operator*(const type_real &rhs) {
    return self_type(xix * rhs, gammax * rhs, xiz * rhs, gammaz * rhs);
  }
};

// operator*
template <
    typename PointPartialDerivativesType,
    std::enable_if_t<!PointPartialDerivativesType::store_jacobian &&
                         PointPartialDerivativesType::dimension_tag ==
                             specfem::dimension::type::dim2 &&
                         PointPartialDerivativesType::data_class ==
                             specfem::data_class::type::partial_derivatives,
                     int> = 0>
KOKKOS_FUNCTION PointPartialDerivativesType
operator*(const type_real &lhs, const PointPartialDerivativesType &rhs) {
  return PointPartialDerivativesType(rhs.xix * lhs, rhs.gammax * lhs,
                                     rhs.xiz * lhs, rhs.gammaz * lhs);
}

/**
 * @brief Template specialization for 3D spectral elements without storing the
 * Jacobian
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD
 */
template <bool UseSIMD>
struct partial_derivatives<specfem::dimension::type::dim3, false, UseSIMD>
    : public specfem::accessor::Accessor<
          specfem::accessor::type::point,
          specfem::data_class::type::partial_derivatives,
          specfem::dimension::type::dim3, UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  using self_type =
      partial_derivatives<specfem::dimension::type::dim3, false,
                          UseSIMD>; ///< Type of the point partial derivatives
  using base_type = specfem::accessor::Accessor<
      specfem::accessor::type::point,
      specfem::data_class::type::partial_derivatives,
      specfem::dimension::type::dim3, UseSIMD>; ///< Base type of the point
                                                ///< partial derivatives
  using simd = typename base_type::simd;        ///< SIMD data type
  using value_type = typename base_type::template scalar_type<type_real>;
  constexpr static bool store_jacobian = false;
  ///@}

  value_type xix;    ///< @xix
  value_type gammax; ///< @gammax
  value_type xiy;    ///< @xix
  value_type gammay; ///< @gammax
  value_type xiz;    ///< @xiz
  value_type gammaz; ///< @gammaz

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
  partial_derivatives() = default;

  /**
   * @brief Constructor with values
   *
   * @param xix @xix
   * @param gammax @gammax
   * @param xiz @xiz
   * @param gammaz @gammaz
   */
  KOKKOS_FUNCTION
  partial_derivatives(const value_type &xix, const value_type &gammax,
                      const value_type &xiy, const value_type &gammay,
                      const value_type &xiz, const value_type &gammaz)
      : xix(xix), gammax(gammax), xiy(xiy), gammay(gammay), xiz(xiz),
        gammaz(gammaz) {}

  /**
   * @brief Constructor with constant value
   *
   * @param constant Value to initialize all members to
   */
  KOKKOS_FUNCTION
  partial_derivatives(const value_type constant)
      : xix(constant), gammax(constant), xiy(constant), gammay(constant),
        xiz(constant), gammaz(constant) {}

  KOKKOS_FUNCTION
  void init() {
    this->xix = 0.0;
    this->gammax = 0.0;
    this->xiy = 0.0;
    this->gammay = 0.0;
    this->xiz = 0.0;
    this->gammaz = 0.0;
    return;
  }

  // operator+
  KOKKOS_FUNCTION self_type operator+(const self_type &rhs) const {
    return self_type(xix + rhs.xix, gammax + rhs.gammax, xiy + rhs.xiy,
                     gammay + rhs.gammay, xiz + rhs.xiz, gammaz + rhs.gammaz);
  }

  // operator+=
  KOKKOS_FUNCTION self_type &operator+=(const self_type &rhs) {
    xix += rhs.xix;
    gammax += rhs.gammax;
    xiy += rhs.xiy;
    gammay += rhs.gammay;
    xiz += rhs.xiz;
    gammaz += rhs.gammaz;
    return *this;
  }

  // operator*
  KOKKOS_FUNCTION self_type operator*(const type_real &rhs) {
    return self_type(xix * rhs, gammax * rhs, xiy * rhs, gammay * rhs,
                     xiz * rhs, gammaz * rhs);
  }
};

// operator*
template <
    typename PointPartialDerivativesType,
    std::enable_if_t<!PointPartialDerivativesType::store_jacobian &&
                         PointPartialDerivativesType::dimension_tag ==
                             specfem::dimension::type::dim3 &&
                         PointPartialDerivativesType::data_class ==
                             specfem::data_class::type::partial_derivatives,
                     int> = 0>
KOKKOS_FUNCTION PointPartialDerivativesType
operator*(const type_real &lhs, const PointPartialDerivativesType &rhs) {
  return PointPartialDerivativesType(rhs.xix * lhs, rhs.gammax * lhs,
                                     rhs.xiy * lhs, rhs.gammay * lhs,
                                     rhs.xiz * lhs, rhs.gammaz * lhs);
  ;
}

/**
 * @brief Template specialization for 2D spectral elements with storing the
 * Jacobian
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD
 */
template <bool UseSIMD>
struct partial_derivatives<specfem::dimension::type::dim2, true, UseSIMD>
    : public partial_derivatives<specfem::dimension::type::dim2, false,
                                 UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  using base_type = partial_derivatives<specfem::dimension::type::dim2, false,
                                        UseSIMD>; ///< Base type of the point
                                                  ///< partial derivatives
  using simd = typename base_type::simd;          ///< SIMD data type
  using value_type = typename base_type::value_type;
  constexpr static bool store_jacobian = true;
  ///@}

  value_type jacobian; ///< Jacobian

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
  partial_derivatives() = default;

  /**
   * @brief Constructor with values
   *
   * @param xix @xix
   * @param gammax @gammax
   * @param xiz @xiz
   * @param gammaz @gammaz
   * @param jacobian Jacobian
   */
  KOKKOS_FUNCTION
  partial_derivatives(const value_type &xix, const value_type &gammax,
                      const value_type &xiz, const value_type &gammaz,
                      const value_type &jacobian)
      : partial_derivatives<specfem::dimension::type::dim2, false, UseSIMD>(
            xix, gammax, xiz, gammaz),
        jacobian(jacobian) {}

  /**
   * @brief Constructor with constant value
   *
   * @param constant Value to initialize all members to
   */
  KOKKOS_FUNCTION
  partial_derivatives(const value_type constant)
      : partial_derivatives<specfem::dimension::type::dim2, false, UseSIMD>(
            constant),
        jacobian(constant) {}
  ///@}

  KOKKOS_FUNCTION
  void init() {
    this->xix = 0.0;
    this->gammax = 0.0;
    this->xiz = 0.0;
    this->gammaz = 0.0;
    this->jacobian = 0.0;
    return;
  }

  /**
   * @name Member functions
   *
   */
  ///@{

  /**
   * @brief Compute the normal vector at a quadrature point
   *
   * @param type Type of edge (bottom, top, left, right)
   * @return specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>
   * Normal vector
   */
  KOKKOS_FUNCTION specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>
  compute_normal(const specfem::enums::edge::type &type) const;
  ///@}

private:
  KOKKOS_INLINE_FUNCTION
  specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>
  impl_compute_normal_bottom() const {
    return { static_cast<value_type>(static_cast<type_real>(-1.0) *
                                     this->gammax * this->jacobian),
             static_cast<value_type>(static_cast<type_real>(-1.0) *
                                     this->gammaz * this->jacobian) };
  };

  KOKKOS_INLINE_FUNCTION
  specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>
  impl_compute_normal_top() const {
    return { static_cast<value_type>(this->gammax * this->jacobian),
             static_cast<value_type>(this->gammaz * this->jacobian) };
  };

  KOKKOS_INLINE_FUNCTION
  specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>
  impl_compute_normal_left() const {
    return { static_cast<value_type>(static_cast<type_real>(-1.0) * this->xix *
                                     this->jacobian),
             static_cast<value_type>(static_cast<type_real>(-1.0) * this->xiz *
                                     this->jacobian) };
  };

  KOKKOS_INLINE_FUNCTION
  specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>
  impl_compute_normal_right() const {
    return { static_cast<value_type>(this->xix * this->jacobian),
             static_cast<value_type>(this->xiz * this->jacobian) };
  };
};

/**
 * @brief Template specialization for 3D spectral elements with storing the
 * Jacobian
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD
 */
template <bool UseSIMD>
struct partial_derivatives<specfem::dimension::type::dim3, true, UseSIMD>
    : public partial_derivatives<specfem::dimension::type::dim3, false,
                                 UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  using base_type = partial_derivatives<specfem::dimension::type::dim2, false,
                                        UseSIMD>; ///< Base type of the point
                                                  ///< partial derivatives
  using simd = typename base_type::simd;          ///< SIMD data type
  using value_type = typename base_type::value_type;
  constexpr static bool store_jacobian = true;
  ///@}

  value_type jacobian; ///< Jacobian

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
  partial_derivatives() = default;

  /**
   * @brief Constructor with values
   *
   * @param xix @xix
   * @param gammax @gammax
   * @param xiy @xiy
   * @param gammay @gammay
   * @param xiz @xiz
   * @param gammaz @gammaz
   * @param jacobian Jacobian
   */
  KOKKOS_FUNCTION
  partial_derivatives(const value_type &xix, const value_type &gammax,
                      const value_type &xiy, const value_type &gammay,
                      const value_type &xiz, const value_type &gammaz,
                      const value_type &jacobian)
      : partial_derivatives<specfem::dimension::type::dim3, false, UseSIMD>(
            xix, gammax, xiy, gammay, xiz, gammaz),
        jacobian(jacobian) {}

  /**
   * @brief Constructor with constant value
   *
   * @param constant Value to initialize all members to
   */
  KOKKOS_FUNCTION
  partial_derivatives(const value_type constant)
      : partial_derivatives<specfem::dimension::type::dim3, false, UseSIMD>(
            constant),
        jacobian(constant) {}
  ///@}

  KOKKOS_FUNCTION
  void init() {
    this->xix = 0.0;
    this->gammax = 0.0;
    this->xiy = 0.0;
    this->gammay = 0.0;
    this->xiz = 0.0;
    this->gammaz = 0.0;
    this->jacobian = 0.0;
    return;
  }

  /**
   * @name Member functions
   *
   */
  ///@{

  //   /**
  //    * @brief Compute the normal vector at a quadrature point
  //    *
  //    * @param type Type of edge (bottom, top, left, right)
  //    * @return specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>
  //    * Normal vector
  //    */
  //   KOKKOS_FUNCTION specfem::datatype::ScalarPointViewType<type_real, 2,
  //   UseSIMD> compute_normal(const specfem::enums::edge::type &type) const;
  //   ///@}

  // private:
  //   KOKKOS_INLINE_FUNCTION
  //   specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>
  //   impl_compute_normal_bottom() const {
  //     return { static_cast<value_type>(static_cast<type_real>(-1.0) *
  //                                      this->gammax * this->jacobian),
  //              static_cast<value_type>(static_cast<type_real>(-1.0) *
  //                                      this->gammaz * this->jacobian) };
  //   };

  //   KOKKOS_INLINE_FUNCTION
  //   specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>
  //   impl_compute_normal_top() const {
  //     return { static_cast<value_type>(this->gammax * this->jacobian),
  //              static_cast<value_type>(this->gammaz * this->jacobian) };
  //   };

  //   KOKKOS_INLINE_FUNCTION
  //   specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>
  //   impl_compute_normal_left() const {
  //     return { static_cast<value_type>(static_cast<type_real>(-1.0) *
  //     this->xix *
  //                                      this->jacobian),
  //              static_cast<value_type>(static_cast<type_real>(-1.0) *
  //              this->xiz *
  //                                      this->jacobian) };
  //   };

  //   KOKKOS_INLINE_FUNCTION
  //   specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>
  //   impl_compute_normal_right() const {
  //     return { static_cast<value_type>(this->xix * this->jacobian),
  //              static_cast<value_type>(this->xiz * this->jacobian) };
  //   };
};

} // namespace point
} // namespace specfem
