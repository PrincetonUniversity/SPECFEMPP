#pragma once

#include "specfem/data_access.hpp"
#include "specfem/enums.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem {
namespace point {

/**
 * @brief Store spatial derivatives of the basis functions at a quadrature point
 *
 * @tparam DimensionTag Dimension of the spectral element
 * @tparam StoreJacobian Boolean indicating whether to store the Jacobian
 * @tparam UseSIMD Boolean indicating whether to use SIMD
 */
template <specfem::element::dimension_tag DimensionTag, bool StoreJacobian,
          bool UseSIMD>
struct jacobian_matrix;

/**
 * @brief Template specialization for 2D spectral elements without storing the
 * Jacobian determinant
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD
 */
template <bool UseSIMD>
struct jacobian_matrix<specfem::element::dimension_tag::dim2, false, UseSIMD>
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::point,
          specfem::data_access::DataClassType::jacobian_matrix,
          specfem::element::dimension_tag::dim2, UseSIMD> {
private:
  using base_type = specfem::data_access::Accessor<
      specfem::datatype::AccessorType::point,
      specfem::data_access::DataClassType::jacobian_matrix,
      specfem::element::dimension_tag::dim2,
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
  using tensor_type = typename base_type::template tensor_type<type_real, 2, 2>;
  constexpr static bool store_jacobian = false;
  ///@}

private:
  // J = [ xix  gammax ]   (row = spatial coord x/z, col = reference coord ξ/γ)
  //     [ xiz  gammaz ]
  tensor_type _data; ///< Underlying 2x2 tensor storage

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
  jacobian_matrix() : _data() { return; }

  /**
   * @brief Constructor with values
   *
   * @param xix @f$ \partial \xi / \partial x @f$
   * @param gammax @f$ \partial \gamma / \partial x @f$
   * @param xiz @f$ \partial \xi / \partial z @f$
   * @param gammaz @f$ \partial \gamma / \partial z @f$
   */
  KOKKOS_FUNCTION
  jacobian_matrix(const value_type &xix, const value_type &gammax,
                  const value_type &xiz, const value_type &gammaz)
      : _data() {
    this->xix() = xix;
    this->gammax() = gammax;
    this->xiz() = xiz;
    this->gammaz() = gammaz;
  }

  /**
   * @brief Constructor with constant value
   *
   * @param constant Value to initialize all members to
   */
  KOKKOS_FUNCTION
  jacobian_matrix(const value_type constant) : _data(constant) {}

  /**
   * @brief Construct from a 2x2 tensor
   *
   * @param t Tensor to copy into the matrix
   */
  KOKKOS_FUNCTION
  jacobian_matrix(const tensor_type &t) : _data(t) {}
  ///@}

  /**
   * @name Component accessors
   *
   * The tensor is the sole storage; each component is addressed by its
   * position in it.
   */
  ///@{
  /**
   * @brief @f$ \partial \xi / \partial x @f$
   * @return Reference to the (0, 0) component
   */
  KOKKOS_FUNCTION value_type &xix() { return _data(0, 0); }
  KOKKOS_FUNCTION const value_type &xix() const { return _data(0, 0); }

  /**
   * @brief @f$ \partial \gamma / \partial x @f$
   * @return Reference to the (0, 1) component
   */
  KOKKOS_FUNCTION value_type &gammax() { return _data(0, 1); }
  KOKKOS_FUNCTION const value_type &gammax() const { return _data(0, 1); }

  /**
   * @brief @f$ \partial \xi / \partial z @f$
   * @return Reference to the (1, 0) component
   */
  KOKKOS_FUNCTION value_type &xiz() { return _data(1, 0); }
  KOKKOS_FUNCTION const value_type &xiz() const { return _data(1, 0); }

  /**
   * @brief @f$ \partial \gamma / \partial z @f$
   * @return Reference to the (1, 1) component
   */
  KOKKOS_FUNCTION value_type &gammaz() { return _data(1, 1); }
  KOKKOS_FUNCTION const value_type &gammaz() const { return _data(1, 1); }
  ///@}

  KOKKOS_FUNCTION
  void init() { _data = tensor_type(); }

  /**
   * @brief Access the underlying 2x2 transformation tensor
   */
  ///@{
  KOKKOS_FUNCTION tensor_type &tensor() { return _data; }
  KOKKOS_FUNCTION const tensor_type &tensor() const { return _data; }
  ///@}

  // operator+
  KOKKOS_FUNCTION jacobian_matrix operator+(const jacobian_matrix &rhs) const {
    return { xix() + rhs.xix(), gammax() + rhs.gammax(), xiz() + rhs.xiz(),
             gammaz() + rhs.gammaz() };
  }

  // operator+=
  KOKKOS_FUNCTION jacobian_matrix &operator+=(const jacobian_matrix &rhs) {
    this->xix() = this->xix() + rhs.xix();
    this->gammax() = this->gammax() + rhs.gammax();
    this->xiz() = this->xiz() + rhs.xiz();
    this->gammaz() = this->gammaz() + rhs.gammaz();
    return *this;
  }

  // operator*
  KOKKOS_FUNCTION jacobian_matrix operator*(const type_real &rhs) const {
    return { xix() * rhs, gammax() * rhs, xiz() * rhs, gammaz() * rhs };
  }

  // operator==
  KOKKOS_FUNCTION bool operator==(const jacobian_matrix &rhs) const {
    return (specfem::utilities::is_close(this->xix(), rhs.xix())) &&
           (specfem::utilities::is_close(this->gammax(), rhs.gammax())) &&
           (specfem::utilities::is_close(this->xiz(), rhs.xiz())) &&
           (specfem::utilities::is_close(this->gammaz(), rhs.gammaz()));
  }
};

// operator*
template <typename PointJacobianMatrixType>
KOKKOS_FUNCTION std::enable_if_t<
    !PointJacobianMatrixType::store_jacobian &&
        PointJacobianMatrixType::dimension_tag ==
            specfem::element::dimension_tag::dim2 &&
        specfem::data_access::is_point<PointJacobianMatrixType>::value &&
        specfem::data_access::is_jacobian_matrix<
            PointJacobianMatrixType>::value,
    PointJacobianMatrixType>
operator*(const type_real &lhs, const PointJacobianMatrixType &rhs) {
  return PointJacobianMatrixType(rhs.xix() * lhs, rhs.gammax() * lhs,
                                 rhs.xiz() * lhs, rhs.gammaz() * lhs);
}

/**
 * @brief Template specialization for 3D spectral elements without storing the
 * Jacobian determinant
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD
 */
template <bool UseSIMD>
struct jacobian_matrix<specfem::element::dimension_tag::dim3, false, UseSIMD>
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::point,
          specfem::data_access::DataClassType::jacobian_matrix,
          specfem::element::dimension_tag::dim3, UseSIMD> {
private:
  using base_type = specfem::data_access::Accessor<
      specfem::datatype::AccessorType::point,
      specfem::data_access::DataClassType::jacobian_matrix,
      specfem::element::dimension_tag::dim3,
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
  using tensor_type = typename base_type::template tensor_type<type_real, 3, 3>;
  constexpr static bool store_jacobian = false;
  ///@}

private:
  // J = [ xix  etax  gammax ]   (row = spatial coord x/y/z, col = ref coord
  // ξ/η/γ)
  //     [ xiy  etay  gammay ]
  //     [ xiz  etaz  gammaz ]
  tensor_type _data; ///< Underlying 3x3 tensor storage

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
  jacobian_matrix() : _data() { return; }

  /**
   * @brief Constructor with values
   *
   * @param xix @f$ \partial \xi / \partial x @f$
   * @param etax @f$ \partial \eta / \partial x @f$
   * @param gammax @f$ \partial \gamma / \partial x @f$
   * @param xiy @f$ \partial \xi / \partial y @f$
   * @param etay @f$ \partial \eta / \partial y @f$
   * @param gammay @f$ \partial \gamma / \partial y @f$
   * @param xiz @f$ \partial \xi / \partial z @f$
   * @param etaz @f$ \partial \eta / \partial z @f$
   * @param gammaz @f$ \partial \gamma / \partial z @f$
   */
  KOKKOS_FUNCTION
  jacobian_matrix(const value_type &xix, const value_type &etax,
                  const value_type &gammax, const value_type &xiy,
                  const value_type &etay, const value_type &gammay,
                  const value_type &xiz, const value_type &etaz,
                  const value_type &gammaz)
      : _data() {
    this->xix() = xix;
    this->etax() = etax;
    this->gammax() = gammax;
    this->xiy() = xiy;
    this->etay() = etay;
    this->gammay() = gammay;
    this->xiz() = xiz;
    this->etaz() = etaz;
    this->gammaz() = gammaz;
  }

  /**
   * @brief Constructor with constant value
   *
   * @param constant Value to initialize all members to
   */
  KOKKOS_FUNCTION
  jacobian_matrix(const value_type constant) : _data(constant) {}

  /**
   * @brief Construct from a 3x3 tensor
   *
   * @param t Tensor to copy into the matrix
   */
  KOKKOS_FUNCTION
  jacobian_matrix(const tensor_type &t) : _data(t) {}
  ///@}

  /**
   * @name Component accessors
   *
   * The tensor is the sole storage; each component is addressed by its
   * position in it.
   */
  ///@{
  /**
   * @brief @f$ \partial \xi / \partial x @f$
   * @return Reference to the (0, 0) component
   */
  KOKKOS_FUNCTION value_type &xix() { return _data(0, 0); }
  KOKKOS_FUNCTION const value_type &xix() const { return _data(0, 0); }

  /**
   * @brief @f$ \partial \eta / \partial x @f$
   * @return Reference to the (0, 1) component
   */
  KOKKOS_FUNCTION value_type &etax() { return _data(0, 1); }
  KOKKOS_FUNCTION const value_type &etax() const { return _data(0, 1); }

  /**
   * @brief @f$ \partial \gamma / \partial x @f$
   * @return Reference to the (0, 2) component
   */
  KOKKOS_FUNCTION value_type &gammax() { return _data(0, 2); }
  KOKKOS_FUNCTION const value_type &gammax() const { return _data(0, 2); }

  /**
   * @brief @f$ \partial \xi / \partial y @f$
   * @return Reference to the (1, 0) component
   */
  KOKKOS_FUNCTION value_type &xiy() { return _data(1, 0); }
  KOKKOS_FUNCTION const value_type &xiy() const { return _data(1, 0); }

  /**
   * @brief @f$ \partial \eta / \partial y @f$
   * @return Reference to the (1, 1) component
   */
  KOKKOS_FUNCTION value_type &etay() { return _data(1, 1); }
  KOKKOS_FUNCTION const value_type &etay() const { return _data(1, 1); }

  /**
   * @brief @f$ \partial \gamma / \partial y @f$
   * @return Reference to the (1, 2) component
   */
  KOKKOS_FUNCTION value_type &gammay() { return _data(1, 2); }
  KOKKOS_FUNCTION const value_type &gammay() const { return _data(1, 2); }

  /**
   * @brief @f$ \partial \xi / \partial z @f$
   * @return Reference to the (2, 0) component
   */
  KOKKOS_FUNCTION value_type &xiz() { return _data(2, 0); }
  KOKKOS_FUNCTION const value_type &xiz() const { return _data(2, 0); }

  /**
   * @brief @f$ \partial \eta / \partial z @f$
   * @return Reference to the (2, 1) component
   */
  KOKKOS_FUNCTION value_type &etaz() { return _data(2, 1); }
  KOKKOS_FUNCTION const value_type &etaz() const { return _data(2, 1); }

  /**
   * @brief @f$ \partial \gamma / \partial z @f$
   * @return Reference to the (2, 2) component
   */
  KOKKOS_FUNCTION value_type &gammaz() { return _data(2, 2); }
  KOKKOS_FUNCTION const value_type &gammaz() const { return _data(2, 2); }
  ///@}

  KOKKOS_FUNCTION
  void init() { _data = tensor_type(); }

  /**
   * @brief Access the underlying 3x3 transformation tensor
   */
  ///@{
  KOKKOS_FUNCTION tensor_type &tensor() { return _data; }
  KOKKOS_FUNCTION const tensor_type &tensor() const { return _data; }
  ///@}

  // operator+
  KOKKOS_FUNCTION jacobian_matrix operator+(const jacobian_matrix &rhs) const {
    return { xix() + rhs.xix(), etax() + rhs.etax(), gammax() + rhs.gammax(),
             xiy() + rhs.xiy(), etay() + rhs.etay(), gammay() + rhs.gammay(),
             xiz() + rhs.xiz(), etaz() + rhs.etaz(), gammaz() + rhs.gammaz() };
  }

  // operator+=
  KOKKOS_FUNCTION jacobian_matrix &operator+=(const jacobian_matrix &rhs) {
    this->xix() = this->xix() + rhs.xix();
    this->etax() = this->etax() + rhs.etax();
    this->gammax() = this->gammax() + rhs.gammax();
    this->xiy() = this->xiy() + rhs.xiy();
    this->etay() = this->etay() + rhs.etay();
    this->gammay() = this->gammay() + rhs.gammay();
    this->xiz() = this->xiz() + rhs.xiz();
    this->etaz() = this->etaz() + rhs.etaz();
    this->gammaz() = this->gammaz() + rhs.gammaz();
    return *this;
  }

  // operator*
  KOKKOS_FUNCTION jacobian_matrix operator*(const type_real &rhs) const {
    return { xix() * rhs, etax() * rhs, gammax() * rhs,
             xiy() * rhs, etay() * rhs, gammay() * rhs,
             xiz() * rhs, etaz() * rhs, gammaz() * rhs };
  }

  // operator==
  KOKKOS_FUNCTION bool operator==(const jacobian_matrix &rhs) const {
    return (specfem::utilities::is_close(this->xix(), rhs.xix())) &&
           (specfem::utilities::is_close(this->etax(), rhs.etax())) &&
           (specfem::utilities::is_close(this->gammax(), rhs.gammax())) &&
           (specfem::utilities::is_close(this->xiy(), rhs.xiy())) &&
           (specfem::utilities::is_close(this->etay(), rhs.etay())) &&
           (specfem::utilities::is_close(this->gammay(), rhs.gammay())) &&
           (specfem::utilities::is_close(this->xiz(), rhs.xiz())) &&
           (specfem::utilities::is_close(this->etaz(), rhs.etaz())) &&
           (specfem::utilities::is_close(this->gammaz(), rhs.gammaz()));
  }
};

// operator*
template <typename PointJacobianMatrixType,
          std::enable_if_t<
              !PointJacobianMatrixType::store_jacobian &&
                  PointJacobianMatrixType::dimension_tag ==
                      specfem::element::dimension_tag::dim3 &&
                  PointJacobianMatrixType::data_class ==
                      specfem::data_access::DataClassType::jacobian_matrix,
              int> = 0>
KOKKOS_FUNCTION PointJacobianMatrixType
operator*(const type_real &lhs, const PointJacobianMatrixType &rhs) {
  return PointJacobianMatrixType(
      rhs.xix() * lhs, rhs.etax() * lhs, rhs.gammax() * lhs, rhs.xiy() * lhs,
      rhs.etay() * lhs, rhs.gammay() * lhs, rhs.xiz() * lhs, rhs.etaz() * lhs,
      rhs.gammaz() * lhs);
}

/**
 * @brief Template specialization for 2D spectral elements with storing the
 * Jacobian determinant
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD
 */
template <bool UseSIMD>
struct jacobian_matrix<specfem::element::dimension_tag::dim2, true, UseSIMD>
    : public jacobian_matrix<specfem::element::dimension_tag::dim2, false,
                             UseSIMD> {
private:
  using base_type =
      jacobian_matrix<specfem::element::dimension_tag::dim2, false,
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
  using tensor_type = typename base_type::tensor_type;
  constexpr static bool store_jacobian = true;
  ///@}

private:
  value_type jacobian_; ///< Jacobian determinant @f$ J @f$

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
  jacobian_matrix() : base_type(), jacobian_(0.0) { return; }

  /**
   * @brief Constructor with values
   *
   * @param xix @f$ \partial \xi / \partial x @f$
   * @param gammax @f$ \partial \gamma / \partial x @f$
   * @param xiz @f$ \partial \xi / \partial z @f$
   * @param gammaz @f$ \partial \gamma / \partial z @f$
   * @param jacobian Jacobian determinant @f$ J @f$
   */
  KOKKOS_FUNCTION
  jacobian_matrix(const value_type &xix, const value_type &gammax,
                  const value_type &xiz, const value_type &gammaz,
                  const value_type &jacobian)
      : jacobian_matrix<specfem::element::dimension_tag::dim2, false, UseSIMD>(
            xix, gammax, xiz, gammaz),
        jacobian_(jacobian) {}

  /**
   * @brief Constructor with constant value
   *
   * @param constant Value to initialize all members to
   */
  KOKKOS_FUNCTION
  jacobian_matrix(const value_type constant)
      : jacobian_matrix<specfem::element::dimension_tag::dim2, false, UseSIMD>(
            constant),
        jacobian_(constant) {}

  /**
   * @brief Construct from a 2x2 tensor and Jacobian determinant
   *
   * @param t Tensor to copy into the matrix
   * @param jacobian Jacobian determinant @f$ J @f$
   */
  KOKKOS_FUNCTION
  jacobian_matrix(const tensor_type &t, const value_type &jacobian)
      : jacobian_matrix<specfem::element::dimension_tag::dim2, false, UseSIMD>(
            t),
        jacobian_(jacobian) {}
  ///@}

  /**
   * @brief Jacobian determinant @f$ J @f$
   * @return Reference to the stored Jacobian determinant
   */
  ///@{
  KOKKOS_FUNCTION value_type &jacobian() { return jacobian_; }
  KOKKOS_FUNCTION const value_type &jacobian() const { return jacobian_; }
  ///@}

  KOKKOS_FUNCTION
  void init() {
    base_type::init();
    this->jacobian_ = 0.0;
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
    return (static_cast<const base_type &>(*this) ==
            static_cast<const base_type &>(rhs)) &&
           (specfem::utilities::is_close(this->jacobian(), rhs.jacobian()));
  }

private:
  specfem::datatype::VectorPointViewType<type_real, 2, UseSIMD>
  impl_compute_normal_bottom() const {
    return { static_cast<value_type>(static_cast<type_real>(-1.0) *
                                     this->gammax() * this->jacobian()),
             static_cast<value_type>(static_cast<type_real>(-1.0) *
                                     this->gammaz() * this->jacobian()) };
  };

  specfem::datatype::VectorPointViewType<type_real, 2, UseSIMD>
  impl_compute_normal_top() const {
    return { static_cast<value_type>(this->gammax() * this->jacobian()),
             static_cast<value_type>(this->gammaz() * this->jacobian()) };
  };

  specfem::datatype::VectorPointViewType<type_real, 2, UseSIMD>
  impl_compute_normal_left() const {
    return { static_cast<value_type>(static_cast<type_real>(-1.0) *
                                     this->xix() * this->jacobian()),
             static_cast<value_type>(static_cast<type_real>(-1.0) *
                                     this->xiz() * this->jacobian()) };
  };

  specfem::datatype::VectorPointViewType<type_real, 2, UseSIMD>
  impl_compute_normal_right() const {
    return { static_cast<value_type>(this->xix() * this->jacobian()),
             static_cast<value_type>(this->xiz() * this->jacobian()) };
  };
};

/**
 * @brief Template specialization for 3D spectral elements with storing the
 * Jacobian determinant
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD
 */
template <bool UseSIMD>
struct jacobian_matrix<specfem::element::dimension_tag::dim3, true, UseSIMD>
    : public jacobian_matrix<specfem::element::dimension_tag::dim3, false,
                             UseSIMD> {
private:
  using base_type =
      jacobian_matrix<specfem::element::dimension_tag::dim3, false,
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
  using tensor_type = typename base_type::tensor_type;
  constexpr static bool store_jacobian = true;
  ///@}

private:
  value_type jacobian_; ///< Jacobian determinant @f$ J @f$

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
  jacobian_matrix() : base_type(), jacobian_(0.0) { return; }

  /**
   * @brief Constructor with values
   *
   * @param xix @f$ \partial \xi / \partial x @f$
   * @param etax @f$ \partial \eta / \partial x @f$
   * @param gammax @f$ \partial \gamma / \partial x @f$
   * @param xiy @f$ \partial \xi / \partial y @f$
   * @param etay @f$ \partial \eta / \partial y @f$
   * @param gammay @f$ \partial \gamma / \partial y @f$
   * @param xiz @f$ \partial \xi / \partial z @f$
   * @param etaz @f$ \partial \eta / \partial z @f$
   * @param gammaz @f$ \partial \gamma / \partial z @f$
   * @param jacobian Jacobian determinant @f$ J @f$
   */
  KOKKOS_FUNCTION
  jacobian_matrix(const value_type &xix, const value_type &etax,
                  const value_type &gammax, const value_type &xiy,
                  const value_type &etay, const value_type &gammay,
                  const value_type &xiz, const value_type &etaz,
                  const value_type &gammaz, const value_type &jacobian)
      : jacobian_matrix<specfem::element::dimension_tag::dim3, false, UseSIMD>(
            xix, etax, gammax, xiy, etay, gammay, xiz, etaz, gammaz),
        jacobian_(jacobian) {}

  /**
   * @brief Constructor with constant value
   *
   * @param constant Value to initialize all members to
   */
  KOKKOS_FUNCTION
  jacobian_matrix(const value_type constant)
      : jacobian_matrix<specfem::element::dimension_tag::dim3, false, UseSIMD>(
            constant),
        jacobian_(constant) {}

  /**
   * @brief Construct from a 3x3 tensor and Jacobian determinant
   *
   * @param t Tensor to copy into the matrix
   * @param jacobian Jacobian determinant @f$ J @f$
   */
  KOKKOS_FUNCTION
  jacobian_matrix(const tensor_type &t, const value_type &jacobian)
      : jacobian_matrix<specfem::element::dimension_tag::dim3, false, UseSIMD>(
            t),
        jacobian_(jacobian) {}
  ///@}

  /**
   * @brief Jacobian determinant @f$ J @f$
   * @return Reference to the stored Jacobian determinant
   */
  ///@{
  KOKKOS_FUNCTION value_type &jacobian() { return jacobian_; }
  KOKKOS_FUNCTION const value_type &jacobian() const { return jacobian_; }
  ///@}

  KOKKOS_FUNCTION
  void init() {
    base_type::init();
    this->jacobian_ = 0.0;
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
    return (static_cast<const base_type &>(*this) ==
            static_cast<const base_type &>(rhs)) &&
           (specfem::utilities::is_close(this->jacobian(), rhs.jacobian()));
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
                                     this->gammax() * this->jacobian()),
             static_cast<value_type>(static_cast<type_real>(-1.0) *
                                     this->gammay() * this->jacobian()),
             static_cast<value_type>(static_cast<type_real>(-1.0) *
                                     this->gammaz() * this->jacobian()) };
  };

  specfem::datatype::VectorPointViewType<type_real, 3, UseSIMD>
  impl_compute_normal_top() const {
    return { static_cast<value_type>(this->gammax() * this->jacobian()),
             static_cast<value_type>(this->gammay() * this->jacobian()),
             static_cast<value_type>(this->gammaz() * this->jacobian()) };
  };

  specfem::datatype::VectorPointViewType<type_real, 3, UseSIMD>
  impl_compute_normal_left() const {
    return { static_cast<value_type>(static_cast<type_real>(-1.0) *
                                     this->xix() * this->jacobian()),
             static_cast<value_type>(static_cast<type_real>(-1.0) *
                                     this->xiy() * this->jacobian()),
             static_cast<value_type>(static_cast<type_real>(-1.0) *
                                     this->xiz() * this->jacobian()) };
  };

  specfem::datatype::VectorPointViewType<type_real, 3, UseSIMD>
  impl_compute_normal_right() const {
    return { static_cast<value_type>(this->xix() * this->jacobian()),
             static_cast<value_type>(this->xiy() * this->jacobian()),
             static_cast<value_type>(this->xiz() * this->jacobian()) };
  };

  specfem::datatype::VectorPointViewType<type_real, 3, UseSIMD>
  impl_compute_normal_front() const {
    // front = eta=-1 face (iy=0); outward normal = -J*nabla(eta)
    return { static_cast<value_type>(static_cast<type_real>(-1.0) *
                                     this->etax() * this->jacobian()),
             static_cast<value_type>(static_cast<type_real>(-1.0) *
                                     this->etay() * this->jacobian()),
             static_cast<value_type>(static_cast<type_real>(-1.0) *
                                     this->etaz() * this->jacobian()) };
  };

  specfem::datatype::VectorPointViewType<type_real, 3, UseSIMD>
  impl_compute_normal_back() const {
    // back = eta=+1 face (iy=nglly-1); outward normal = +J*nabla(eta)
    return { static_cast<value_type>(this->etax() * this->jacobian()),
             static_cast<value_type>(this->etay() * this->jacobian()),
             static_cast<value_type>(this->etaz() * this->jacobian()) };
  };
};

// Kokkos bit-copies this type both as a `View` element and as the reduction
// scalar of `Kokkos::Sum<>` (see specfem::algorithms::interpolate_function).
// It must therefore have no self-pointers -- storing references into its own
// tensor made it dangle after a `memcpy`, which segfaulted under threaded
// backends while passing on Serial (issue #2008).
//
// The `store_jacobian = true` specializations derive from their
// `store_jacobian = false` counterparts, and a derived class cannot be
// trivially copyable unless its bases are, so checking the former covers both.
template <bool UseSIMD>
inline constexpr bool jacobian_matrix_is_relocatable_v =
    std::is_trivially_copyable_v<jacobian_matrix<
        specfem::element::dimension_tag::dim2, true, UseSIMD>> &&
    std::is_trivially_copyable_v<
        jacobian_matrix<specfem::element::dimension_tag::dim3, true, UseSIMD>>;

static_assert(jacobian_matrix_is_relocatable_v<false> &&
                  jacobian_matrix_is_relocatable_v<true>,
              "point::jacobian_matrix must be trivially copyable");

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
