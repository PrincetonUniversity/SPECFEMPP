#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
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
struct jacobian_matrix;

/**
 * @brief Template specialization for 2D spectral elements without storing the
 * Jacobian
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD
 */
template <bool UseSIMD>
struct jacobian_matrix<specfem::dimension::type::dim2, false, UseSIMD>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::jacobian_matrix,
          specfem::dimension::type::dim2, UseSIMD> {
private:
  using base_type = specfem::data_access::Accessor<
      specfem::data_access::AccessorType::point,
      specfem::data_access::DataClassType::jacobian_matrix,
      specfem::dimension::type::dim2,
      UseSIMD>; ///< Base type of the point
  ///< Jacobian matrix
public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = typename base_type::template simd<type_real>; ///< SIMD data type
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
  jacobian_matrix() = default;

  /**
   * @brief Constructor with values
   *
   * @param xix @xix
   * @param gammax @gammax
   * @param xiz @xiz
   * @param gammaz @gammaz
   */
  KOKKOS_FUNCTION
  jacobian_matrix(const value_type &xix, const value_type &gammax,
                  const value_type &xiz, const value_type &gammaz)
      : xix(xix), gammax(gammax), xiz(xiz), gammaz(gammaz) {}

  /**
   * @brief Constructor with constant value
   *
   * @param constant Value to initialize all members to
   */
  KOKKOS_FUNCTION
  jacobian_matrix(const value_type constant)
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
  KOKKOS_FUNCTION jacobian_matrix operator+(const jacobian_matrix &rhs) const {
    return { xix + rhs.xix, gammax + rhs.gammax, xiz + rhs.xiz,
             gammaz + rhs.gammaz };
  }

  // operator+=
  KOKKOS_FUNCTION jacobian_matrix &operator+=(const jacobian_matrix &rhs) {
    this->xix = this->xix + rhs.xix;
    this->gammax = this->gammax + rhs.gammax;
    this->xiz = this->xiz + rhs.xiz;
    this->gammaz = this->gammaz + rhs.gammaz;
    return *this;
  }

  // operator*
  KOKKOS_FUNCTION jacobian_matrix operator*(const type_real &rhs) {
    return { xix * rhs, gammax * rhs, xiz * rhs, gammaz * rhs };
  }
};

// operator*
template <typename PointJacobianMatrixType>
KOKKOS_FUNCTION std::enable_if_t<
    !PointJacobianMatrixType::store_jacobian &&
        PointJacobianMatrixType::dimension_tag ==
            specfem::dimension::type::dim2 &&
        specfem::data_access::is_point<PointJacobianMatrixType>::value &&
        specfem::data_access::is_jacobian_matrix<
            PointJacobianMatrixType>::value,
    PointJacobianMatrixType>
operator*(const type_real &lhs, const PointJacobianMatrixType &rhs) {
  return PointJacobianMatrixType(rhs.xix * lhs, rhs.gammax * lhs, rhs.xiz * lhs,
                                 rhs.gammaz * lhs);
}

/**
 * @brief Template specialization for 3D spectral elements without storing the
 * Jacobian
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD
 */
template <bool UseSIMD>
struct jacobian_matrix<specfem::dimension::type::dim3, false, UseSIMD>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::jacobian_matrix,
          specfem::dimension::type::dim3, UseSIMD> {
private:
  using base_type = specfem::data_access::Accessor<
      specfem::data_access::AccessorType::point,
      specfem::data_access::DataClassType::jacobian_matrix,
      specfem::dimension::type::dim3,
      UseSIMD>; ///< Base type of the point
                ///< Jacobian matrix
public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = typename base_type::template simd<type_real>; ///< SIMD data type
  using value_type = typename base_type::template scalar_type<type_real>;
  constexpr static bool store_jacobian = false;
  ///@}

  value_type xix;    ///< @xix
  value_type eta_x;  ///< @eta_x
  value_type gammax; ///< @gammax
  value_type xiy;    ///< @xiy
  value_type eta_y;  ///< @eta_y
  value_type gammay; ///< @gammay
  value_type xiz;    ///< @xiz
  value_type eta_z;  ///< @eta_z
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
  jacobian_matrix() = default;

  /**
   * @brief Constructor with values
   *
   * @param xix @xix
   * @param eta_x @eta_x
   * @param gammax @gammax
   * @param xiy @xiy
   * @param eta_y @eta_y
   * @param gammay @gammay
   * @param xiz @xiz
   * @param eta_z @eta_z
   * @param gammaz @gammaz
   */
  KOKKOS_FUNCTION
  jacobian_matrix(const value_type &xix, const value_type &eta_x,
                  const value_type &gammax, const value_type &xiy,
                  const value_type &eta_y, const value_type &gammay,
                  const value_type &xiz, const value_type &eta_z,
                  const value_type &gammaz)
      : xix(xix), eta_x(eta_x), gammax(gammax), xiy(xiy), eta_y(eta_y),
        gammay(gammay), xiz(xiz), eta_z(eta_z), gammaz(gammaz) {}

  /**
   * @brief Constructor with constant value
   *
   * @param constant Value to initialize all members to
   */
  KOKKOS_FUNCTION
  jacobian_matrix(const value_type constant)
      : xix(constant), gammax(constant), xiy(constant), gammay(constant),
        xiz(constant), gammaz(constant) {}

  KOKKOS_FUNCTION
  void init() {
    this->xix = 0.0;
    this->eta_x = 0.0;
    this->gammax = 0.0;
    this->xiy = 0.0;
    this->eta_y = 0.0;
    this->gammay = 0.0;
    this->xiz = 0.0;
    this->eta_z = 0.0;
    this->gammaz = 0.0;
    return;
  }

  // operator+
  KOKKOS_FUNCTION jacobian_matrix operator+(const jacobian_matrix &rhs) const {
    return { xix + rhs.xix, eta_x + rhs.eta_x, gammax + rhs.gammax,
             xiy + rhs.xiy, eta_y + rhs.eta_y, gammay + rhs.gammay,
             xiz + rhs.xiz, eta_z + rhs.eta_z, gammaz + rhs.gammaz };
  }

  // operator+=
  KOKKOS_FUNCTION jacobian_matrix &operator+=(const jacobian_matrix &rhs) {
    this->xix = this->xix + rhs.xix;
    this->eta_x = this->eta_x + rhs.eta_x;
    this->gammax = this->gammax + rhs.gammax;
    this->xiy = this->xiy + rhs.xiy;
    this->eta_y = this->eta_y + rhs.eta_y;
    this->gammay = this->gammay + rhs.gammay;
    this->xiz = this->xiz + rhs.xiz;
    this->eta_z = this->eta_z + rhs.eta_z;
    this->gammaz = this->gammaz + rhs.gammaz;
    return *this;
  }

  // operator*
  KOKKOS_FUNCTION jacobian_matrix operator*(const type_real &rhs) {
    return { xix * rhs,    eta_x * rhs, gammax * rhs, xiy * rhs,   eta_y * rhs,
             gammay * rhs, xiz * rhs,   eta_z * rhs,  gammaz * rhs };
  }
};

// operator*
template <typename PointJacobianMatrixType,
          std::enable_if_t<
              !PointJacobianMatrixType::store_jacobian &&
                  PointJacobianMatrixType::dimension_tag ==
                      specfem::dimension::type::dim3 &&
                  PointJacobianMatrixType::data_class ==
                      specfem::data_access::DataClassType::jacobian_matrix,
              int> = 0>
KOKKOS_FUNCTION PointJacobianMatrixType
operator*(const type_real &lhs, const PointJacobianMatrixType &rhs) {
  return PointJacobianMatrixType(
      rhs.xix * lhs, rhs.eta_x * lhs, rhs.gammax * lhs, rhs.xiy * lhs,
      rhs.eta_y * lhs, rhs.gammay * lhs, rhs.xiz * lhs, rhs.eta_z * lhs,
      rhs.gammaz * lhs);
}

/**
 * @brief Template specialization for 2D spectral elements with storing the
 * Jacobian
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD
 */
template <bool UseSIMD>
struct jacobian_matrix<specfem::dimension::type::dim2, true, UseSIMD>
    : public jacobian_matrix<specfem::dimension::type::dim2, false, UseSIMD> {
private:
  using base_type = jacobian_matrix<specfem::dimension::type::dim2, false,
                                    UseSIMD>; ///< Base type of the point
                                              ///< Jacobian matrix
public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = typename base_type::simd; ///< SIMD data type
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
  jacobian_matrix() = default;

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
  jacobian_matrix(const value_type &xix, const value_type &gammax,
                  const value_type &xiz, const value_type &gammaz,
                  const value_type &jacobian)
      : jacobian_matrix<specfem::dimension::type::dim2, false, UseSIMD>(
            xix, gammax, xiz, gammaz),
        jacobian(jacobian) {}

  /**
   * @brief Constructor with constant value
   *
   * @param constant Value to initialize all members to
   */
  KOKKOS_FUNCTION
  jacobian_matrix(const value_type constant)
      : jacobian_matrix<specfem::dimension::type::dim2, false, UseSIMD>(
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

  // operator+
  KOKKOS_FUNCTION jacobian_matrix operator+(const jacobian_matrix &rhs) =
      delete;

  // operator+=
  KOKKOS_FUNCTION jacobian_matrix &
  operator+=(const jacobian_matrix &rhs) = delete;

  // operator*
  KOKKOS_FUNCTION jacobian_matrix operator*(const type_real &rhs) = delete;

  /**
   * @name Member functions
   *
   */
  ///@{

  /**
   * @brief Compute the normal vector at a quadrature point
   *
   * @param type Type of edge (bottom, top, left, right)
   * @return specfem::datatype::VectorPointViewType<type_real, 2, UseSIMD>
   * Normal vector
   */
  specfem::datatype::VectorPointViewType<type_real, 2, UseSIMD>
  compute_normal(const specfem::enums::edge::type &type) const;
  ///@}

private:
  specfem::datatype::VectorPointViewType<type_real, 2, UseSIMD>
  impl_compute_normal_bottom() const {
    return { static_cast<value_type>(static_cast<type_real>(-1.0) *
                                     this->gammax * this->jacobian),
             static_cast<value_type>(static_cast<type_real>(-1.0) *
                                     this->gammaz * this->jacobian) };
  };

  specfem::datatype::VectorPointViewType<type_real, 2, UseSIMD>
  impl_compute_normal_top() const {
    return { static_cast<value_type>(this->gammax * this->jacobian),
             static_cast<value_type>(this->gammaz * this->jacobian) };
  };

  specfem::datatype::VectorPointViewType<type_real, 2, UseSIMD>
  impl_compute_normal_left() const {
    return { static_cast<value_type>(static_cast<type_real>(-1.0) * this->xix *
                                     this->jacobian),
             static_cast<value_type>(static_cast<type_real>(-1.0) * this->xiz *
                                     this->jacobian) };
  };

  specfem::datatype::VectorPointViewType<type_real, 2, UseSIMD>
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
struct jacobian_matrix<specfem::dimension::type::dim3, true, UseSIMD>
    : public jacobian_matrix<specfem::dimension::type::dim3, false, UseSIMD> {
private:
  using base_type = jacobian_matrix<specfem::dimension::type::dim2, false,
                                    UseSIMD>; ///< Base type of the point
                                              ///< Jacobian matrix
public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = typename base_type::simd; ///< SIMD data type
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
  jacobian_matrix() = default;

  /**
   * @brief Constructor with values
   *
   * @param xix @xix
   * @param eta_x @eta_x
   * @param gammax @gammax
   * @param xiy @xiy
   * @param eta_y @eta_y
   * @param gammay @gammay
   * @param xiz @xiz
   * @param eta_z @eta_z
   * @param gammaz @gammaz
   * @param jacobian Jacobian
   */
  KOKKOS_FUNCTION
  jacobian_matrix(const value_type &xix, const value_type &eta_x,
                  const value_type &gammax, const value_type &xiy,
                  const value_type &eta_y, const value_type &gammay,
                  const value_type &xiz, const value_type &eta_z,
                  const value_type &gammaz, const value_type &jacobian)
      : jacobian_matrix<specfem::dimension::type::dim3, false, UseSIMD>(
            xix, eta_x, gammax, xiy, eta_y, gammay, xiz, eta_z, gammaz),
        jacobian(jacobian) {}

  /**
   * @brief Constructor with constant value
   *
   * @param constant Value to initialize all members to
   */
  KOKKOS_FUNCTION
  jacobian_matrix(const value_type constant)
      : jacobian_matrix<specfem::dimension::type::dim3, false, UseSIMD>(
            constant),
        jacobian(constant) {}
  ///@}

  KOKKOS_FUNCTION
  void init() {
    this->xix = 0.0;
    this->eta_x = 0.0;
    this->gammax = 0.0;
    this->xiy = 0.0;
    this->eta_y = 0.0;
    this->gammay = 0.0;
    this->xiz = 0.0;
    this->eta_z = 0.0;
    this->gammaz = 0.0;
    this->jacobian = 0.0;
    return;
  }

  // operator+
  KOKKOS_FUNCTION jacobian_matrix operator+(const jacobian_matrix &rhs) =
      delete;

  // operator+=
  KOKKOS_FUNCTION jacobian_matrix &
  operator+=(const jacobian_matrix &rhs) = delete;

  // operator*
  KOKKOS_FUNCTION jacobian_matrix operator*(const type_real &rhs) = delete;

  /**
   * @name Member functions
   *
   */
  ///@{

  //   /**
  //    * @brief Compute the normal vector at a quadrature point
  //    *
  //    * @param type Type of edge (bottom, top, left, right)
  //    * @return specfem::datatype::VectorPointViewType<type_real, 2, UseSIMD>
  //    * Normal vector
  //    */
  //   KOKKOS_FUNCTION specfem::datatype::VectorPointViewType<type_real, 2,
  //   UseSIMD> compute_normal(const specfem::enums::edge::type &type) const;
  //   ///@}

  // private:
  //   KOKKOS_INLINE_FUNCTION
  //   specfem::datatype::VectorPointViewType<type_real, 2, UseSIMD>
  //   impl_compute_normal_bottom() const {
  //     return { static_cast<value_type>(static_cast<type_real>(-1.0) *
  //                                      this->gammax * this->jacobian),
  //              static_cast<value_type>(static_cast<type_real>(-1.0) *
  //                                      this->gammaz * this->jacobian) };
  //   };

  //   KOKKOS_INLINE_FUNCTION
  //   specfem::datatype::VectorPointViewType<type_real, 2, UseSIMD>
  //   impl_compute_normal_top() const {
  //     return { static_cast<value_type>(this->gammax * this->jacobian),
  //              static_cast<value_type>(this->gammaz * this->jacobian) };
  //   };

  //   KOKKOS_INLINE_FUNCTION
  //   specfem::datatype::VectorPointViewType<type_real, 2, UseSIMD>
  //   impl_compute_normal_left() const {
  //     return { static_cast<value_type>(static_cast<type_real>(-1.0) *
  //     this->xix *
  //                                      this->jacobian),
  //              static_cast<value_type>(static_cast<type_real>(-1.0) *
  //              this->xiz *
  //                                      this->jacobian) };
  //   };

  //   KOKKOS_INLINE_FUNCTION
  //   specfem::datatype::VectorPointViewType<type_real, 2, UseSIMD>
  //   impl_compute_normal_right() const {
  //     return { static_cast<value_type>(this->xix * this->jacobian),
  //              static_cast<value_type>(this->xiz * this->jacobian) };
  //   };
};

} // namespace point
} // namespace specfem
