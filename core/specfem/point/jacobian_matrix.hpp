#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include "specfem_setup.hpp"
#include "utilities/utilities.hpp"
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
  jacobian_matrix() {
    this->init();
    return;
  }

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

  // operator==
  KOKKOS_FUNCTION bool operator==(const jacobian_matrix &rhs) const {
    return (specfem::utilities::is_close(this->xix, rhs.xix)) &&
           (specfem::utilities::is_close(this->gammax, rhs.gammax)) &&
           (specfem::utilities::is_close(this->xiz, rhs.xiz)) &&
           (specfem::utilities::is_close(this->gammaz, rhs.gammaz));
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
  value_type etax;   ///< @etax
  value_type gammax; ///< @gammax
  value_type xiy;    ///< @xiy
  value_type etay;   ///< @etay
  value_type gammay; ///< @gammay
  value_type xiz;    ///< @xiz
  value_type etaz;   ///< @etaz
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
  jacobian_matrix() {
    this->init();
    return;
  }

  /**
   * @brief Constructor with values
   *
   * @param xix @xix
   * @param etax @etax
   * @param gammax @gammax
   * @param xiy @xiy
   * @param etay @etay
   * @param gammay @gammay
   * @param xiz @xiz
   * @param etaz @etaz
   * @param gammaz @gammaz
   */
  KOKKOS_FUNCTION
  jacobian_matrix(const value_type &xix, const value_type &etax,
                  const value_type &gammax, const value_type &xiy,
                  const value_type &etay, const value_type &gammay,
                  const value_type &xiz, const value_type &etaz,
                  const value_type &gammaz)
      : xix(xix), etax(etax), gammax(gammax), xiy(xiy), etay(etay),
        gammay(gammay), xiz(xiz), etaz(etaz), gammaz(gammaz) {}

  /**
   * @brief Constructor with constant value
   *
   * @param constant Value to initialize all members to
   */
  KOKKOS_FUNCTION
  jacobian_matrix(const value_type constant)
      : xix(constant), etax(constant), gammax(constant), xiy(constant),
        etay(constant), gammay(constant), xiz(constant), etaz(constant),
        gammaz(constant) {}

  KOKKOS_FUNCTION
  void init() {
    this->xix = 0.0;
    this->etax = 0.0;
    this->gammax = 0.0;
    this->xiy = 0.0;
    this->etay = 0.0;
    this->gammay = 0.0;
    this->xiz = 0.0;
    this->etaz = 0.0;
    this->gammaz = 0.0;
    return;
  }

  // operator+
  KOKKOS_FUNCTION jacobian_matrix operator+(const jacobian_matrix &rhs) const {
    return { xix + rhs.xix, etax + rhs.etax, gammax + rhs.gammax,
             xiy + rhs.xiy, etay + rhs.etay, gammay + rhs.gammay,
             xiz + rhs.xiz, etaz + rhs.etaz, gammaz + rhs.gammaz };
  }

  // operator+=
  KOKKOS_FUNCTION jacobian_matrix &operator+=(const jacobian_matrix &rhs) {
    this->xix = this->xix + rhs.xix;
    this->etax = this->etax + rhs.etax;
    this->gammax = this->gammax + rhs.gammax;
    this->xiy = this->xiy + rhs.xiy;
    this->etay = this->etay + rhs.etay;
    this->gammay = this->gammay + rhs.gammay;
    this->xiz = this->xiz + rhs.xiz;
    this->etaz = this->etaz + rhs.etaz;
    this->gammaz = this->gammaz + rhs.gammaz;
    return *this;
  }

  // operator*
  KOKKOS_FUNCTION jacobian_matrix operator*(const type_real &rhs) {
    return { xix * rhs,    etax * rhs, gammax * rhs, xiy * rhs,   etay * rhs,
             gammay * rhs, xiz * rhs,  etaz * rhs,   gammaz * rhs };
  }

  // operator==
  KOKKOS_FUNCTION bool operator==(const jacobian_matrix &rhs) const {
    return (specfem::utilities::is_close(this->xix, rhs.xix)) &&
           (specfem::utilities::is_close(this->etax, rhs.etax)) &&
           (specfem::utilities::is_close(this->gammax, rhs.gammax)) &&
           (specfem::utilities::is_close(this->xiy, rhs.xiy)) &&
           (specfem::utilities::is_close(this->etay, rhs.etay)) &&
           (specfem::utilities::is_close(this->gammay, rhs.gammay)) &&
           (specfem::utilities::is_close(this->xiz, rhs.xiz)) &&
           (specfem::utilities::is_close(this->etaz, rhs.etaz)) &&
           (specfem::utilities::is_close(this->gammaz, rhs.gammaz));
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
      rhs.xix * lhs, rhs.etax * lhs, rhs.gammax * lhs, rhs.xiy * lhs,
      rhs.etay * lhs, rhs.gammay * lhs, rhs.xiz * lhs, rhs.etaz * lhs,
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
  jacobian_matrix() {
    this->init();
    return;
  }

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
  compute_normal(const specfem::mesh_entity::dim2::type &type) const;
  ///@}

  KOKKOS_FUNCTION bool operator==(const jacobian_matrix &rhs) const {
    return (static_cast<base_type>(*this) == static_cast<base_type>(rhs)) &&
           (specfem::utilities::is_close(this->jacobian, rhs.jacobian));
  }

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
  using base_type = jacobian_matrix<specfem::dimension::type::dim3, false,
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
  jacobian_matrix() {
    this->init();
    return;
  }

  /**
   * @brief Constructor with values
   *
   * @param xix @xix
   * @param etax @etax
   * @param gammax @gammax
   * @param xiy @xiy
   * @param etay @etay
   * @param gammay @gammay
   * @param xiz @xiz
   * @param etaz @etaz
   * @param gammaz @gammaz
   * @param jacobian Jacobian
   */
  KOKKOS_FUNCTION
  jacobian_matrix(const value_type &xix, const value_type &etax,
                  const value_type &gammax, const value_type &xiy,
                  const value_type &etay, const value_type &gammay,
                  const value_type &xiz, const value_type &etaz,
                  const value_type &gammaz, const value_type &jacobian)
      : jacobian_matrix<specfem::dimension::type::dim3, false, UseSIMD>(
            xix, etax, gammax, xiy, etay, gammay, xiz, etaz, gammaz),
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
    this->etax = 0.0;
    this->gammax = 0.0;
    this->xiy = 0.0;
    this->etay = 0.0;
    this->gammay = 0.0;
    this->xiz = 0.0;
    this->etaz = 0.0;
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

  // operator==
  KOKKOS_FUNCTION bool operator==(const jacobian_matrix &rhs) const {
    return (static_cast<base_type>(*this) == static_cast<base_type>(rhs)) &&
           (specfem::utilities::is_close(this->jacobian, rhs.jacobian));
  }

  /**
   * @name Member functions
   *
   */
  ///@{

  /**
   * @brief Compute the normal vector at a quadrature point
   *
   * @param type Type of boundary face
   * @return specfem::datatype::VectorPointViewType<type_real, 3, UseSIMD>
   * Normal vector
   */
  specfem::datatype::VectorPointViewType<type_real, 3, UseSIMD>
  compute_normal(const specfem::mesh_entity::dim3::type &type) const;
  ///@}

private:
  specfem::datatype::VectorPointViewType<type_real, 3, UseSIMD>
  impl_compute_normal_bottom() const {
    return { static_cast<value_type>(static_cast<type_real>(-1.0) *
                                     this->gammax * this->jacobian),
             static_cast<value_type>(static_cast<type_real>(-1.0) *
                                     this->gammay * this->jacobian),
             static_cast<value_type>(static_cast<type_real>(-1.0) *
                                     this->gammaz * this->jacobian) };
  };

  specfem::datatype::VectorPointViewType<type_real, 3, UseSIMD>
  impl_compute_normal_top() const {
    return { static_cast<value_type>(this->gammax * this->jacobian),
             static_cast<value_type>(this->gammay * this->jacobian),
             static_cast<value_type>(this->gammaz * this->jacobian) };
  };

  specfem::datatype::VectorPointViewType<type_real, 3, UseSIMD>
  impl_compute_normal_left() const {
    return { static_cast<value_type>(static_cast<type_real>(-1.0) * this->xix *
                                     this->jacobian),
             static_cast<value_type>(static_cast<type_real>(-1.0) * this->xiy *
                                     this->jacobian),
             static_cast<value_type>(static_cast<type_real>(-1.0) * this->xiz *
                                     this->jacobian) };
  };

  specfem::datatype::VectorPointViewType<type_real, 3, UseSIMD>
  impl_compute_normal_right() const {
    return { static_cast<value_type>(this->xix * this->jacobian),
             static_cast<value_type>(this->xiy * this->jacobian),
             static_cast<value_type>(this->xiz * this->jacobian) };
  };

  specfem::datatype::VectorPointViewType<type_real, 3, UseSIMD>
  impl_compute_normal_front() const {
    return { static_cast<value_type>(this->etax * this->jacobian),
             static_cast<value_type>(this->etay * this->jacobian),
             static_cast<value_type>(this->etaz * this->jacobian) };
  };

  specfem::datatype::VectorPointViewType<type_real, 3, UseSIMD>
  impl_compute_normal_back() const {
    return { static_cast<value_type>(static_cast<type_real>(-1.0) * this->etax *
                                     this->jacobian),
             static_cast<value_type>(static_cast<type_real>(-1.0) * this->etay *
                                     this->jacobian),
             static_cast<value_type>(static_cast<type_real>(-1.0) * this->etaz *
                                     this->jacobian) };
  };
};

} // namespace point
} // namespace specfem

namespace Kokkos { // reduction identity must be defined in Kokkos namespace
template <typename T> struct reduction_identity {
  KOKKOS_FORCEINLINE_FUNCTION static std::enable_if_t<
      ((specfem::data_access::is_point<T>::value) &&
       (specfem::data_access::is_jacobian_matrix<T>::value)),
      T>
  sum() {
    return T();
  }
};
} // namespace Kokkos
