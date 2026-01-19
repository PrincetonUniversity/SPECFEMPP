#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include "specfem/datatype.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {
/**
 * @brief Struct to store boundary conditions (and boundary related information)
 * associated with a quadrature point
 *
 * @tparam BoundaryTag Tag indicating the type of boundary condition
 * @DimensionTag Dimension of the spectral element where the quadrature point
 * is located
 * @tparam UseSIMD Boolean indicating whether to use SIMD instructions
 */
template <specfem::element::boundary_tag BoundaryTag,
          specfem::dimension::type DimensionTag, bool UseSIMD>
struct boundary;

/**
 * @brief Template specialization for no boundary condition
 *
 * @tparam DimensionTag Dimension of the spectral element where the quadrature
 * point is located
 * @tparam UseSIMD Boolean indicating whether to use SIMD instructions
 */
template <specfem::dimension::type DimensionTag, bool UseSIMD>
struct boundary<specfem::element::boundary_tag::none, DimensionTag, UseSIMD>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::boundary, DimensionTag,
          UseSIMD> {
private:
  // We use simd_like vector to store tags. Tags are stored as enums, so a simd
  // type is ill-defined for them. However, we use scalar array types of size
  // simd<type_real>::size() to store them. The goal of this approach is to use
  // tags to mask a type_real simd vector and perform SIMD operations on those
  // SIMD vectors.
  /**
   * @brief Datatype for storing values.
   *
   * Is a scalar if UseSIMD is false, otherwise is a SIMD like vector.
   */
  using value_type = typename specfem::datatype::simd_like<
      specfem::element::boundary_tag_container, type_real, UseSIMD>::datatype;

public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  /**
   * @brief SIMD data type.
   */
  using simd = specfem::datatype::simd<type_real, UseSIMD>;
  ///@}

  /**
   * @name Compile-time constants
   *
   */
  ///@{
  /**
   * @brief Tag indicating no boundary condition.
   */
  constexpr static auto boundary_tag = specfem::element::boundary_tag::none;
  ///@}

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
  boundary() = default;
  ///@}

  /**
   * @brief Tag indicating the type of boundary condition at the quadrature
   * point.
   */
  value_type tag;
};

/**
 * @brief Template specialization for acoustic free surface boundary condition
 *
 * @tparam DimensionTag Dimension of the spectral element where the quadrature
 * point is located
 * @tparam UseSIMD Boolean indicating whether to use SIMD instructions
 */
template <specfem::dimension::type DimensionTag, bool UseSIMD>
struct boundary<specfem::element::boundary_tag::acoustic_free_surface,
                DimensionTag, UseSIMD>
    : public boundary<specfem::element::boundary_tag::none, DimensionTag,
                      UseSIMD> {
public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  /**
   * @brief SIMD data type.
   */
  using simd = specfem::datatype::simd<type_real, UseSIMD>;
  ///@}
  /**
   * @name Compile-time constants
   *
   */
  ///@{
  /**
   * @brief Tag indicating acoustic free surface boundary condition.
   */
  constexpr static auto boundary_tag =
      specfem::element::boundary_tag::acoustic_free_surface;
  ///@}

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
  boundary() = default;

  /**
   * @brief Implicit conversion constructor from composite Stacey Dirichlet
   * boundary
   *
   * @param boundary Composite Stacey Dirichlet boundary
   */
  KOKKOS_FUNCTION
  boundary(const specfem::point::boundary<
           specfem::element::boundary_tag::composite_stacey_dirichlet,
           DimensionTag, UseSIMD> &boundary);
  ///@}
};

/**
 * @brief Template specialization for Stacey boundary condition
 *
 * @tparam DimensionTag Dimension of the spectral element where the quadrature
 * point is located
 * @tparam UseSIMD Boolean indicating whether to use SIMD instructions
 */
template <specfem::dimension::type DimensionTag, bool UseSIMD>
struct boundary<specfem::element::boundary_tag::stacey, DimensionTag, UseSIMD>
    : public boundary<specfem::element::boundary_tag::acoustic_free_surface,
                      DimensionTag, UseSIMD> {
private:
  /**
   * @brief Number of spatial dimensions.
   */
  constexpr static int num_dimensions =
      specfem::dimension::dimension<DimensionTag>::dim;
  /**
   * @name Private Typedefs
   *
   */
  ///@{
  /**
   * @brief View type to store the normal vector to the edge at the quadrature
   * point.
   */
  using NormalViewType =
      specfem::datatype::VectorPointViewType<type_real, num_dimensions,
                                             UseSIMD>;

  /**
   * @brief SIMD data type.
   */
  using datatype =
      typename specfem::datatype::simd<type_real, UseSIMD>::datatype;
  ///@}

public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  /**
   * @brief SIMD data type.
   */
  using simd = specfem::datatype::simd<type_real, UseSIMD>;
  ///@}

  /**
   * @name Compile-time constants
   *
   */
  ///@{
  /**
   * @brief Tag indicating Stacey boundary condition.
   */
  constexpr static auto boundary_tag = specfem::element::boundary_tag::stacey;
  ///@}

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
  boundary() = default;

  /**
   * @brief Implicit conversion constructor from composite Stacey Dirichlet
   * boundary
   *
   * @param boundary Composite Stacey Dirichlet boundary
   */
  KOKKOS_FUNCTION
  boundary(const specfem::point::boundary<
           specfem::element::boundary_tag::composite_stacey_dirichlet,
           DimensionTag, UseSIMD> &boundary);
  ///@}

  /**
   * @brief Integration weight associated with the edge at the quadrature point.
   */
  datatype edge_weight = static_cast<type_real>(0.0);

  /**
   * @brief Normal vector to the edge at the quadrature point.
   */
  NormalViewType edge_normal = { static_cast<type_real>(0.0),
                                 static_cast<type_real>(0.0) };
};

/**
 * @brief Template specialization for Stacey boundary condition in 3D
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD instructions
 */
template <bool UseSIMD>
struct boundary<specfem::element::boundary_tag::stacey,
                specfem::dimension::type::dim3, UseSIMD>
    : public boundary<specfem::element::boundary_tag::acoustic_free_surface,
                      specfem::dimension::type::dim3, UseSIMD> {
private:
  constexpr static int num_dimensions = 3;
  /**
   * @name Private Typedefs
   *
   */
  ///@{
  using NormalViewType =
      specfem::datatype::VectorPointViewType<type_real, num_dimensions,
                                             UseSIMD>; ///< View type to store
                                                       ///< the normal vector to
                                                       ///< the face at the
                                                       ///< quadrature point

  using datatype =
      typename specfem::datatype::simd<type_real, UseSIMD>::datatype; ///< SIMD
                                                                      ///< data
                                                                      ///< type
  ///@}

public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<type_real, UseSIMD>;
  ///@}

  /**
   * @name Compile-time constants
   *
   */
  ///@{
  constexpr static auto boundary_tag = specfem::element::boundary_tag::stacey;
  ///@}

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
  boundary() = default;

  /**
   * @brief Implicit conversion constructor from composite Stacey Dirichlet
   * boundary
   *
   * @param boundary Composite Stacey Dirichlet boundary
   */
  KOKKOS_FUNCTION
  boundary(const specfem::point::boundary<
           specfem::element::boundary_tag::composite_stacey_dirichlet,
           specfem::dimension::type::dim3, UseSIMD> &boundary);
  ///@}

  datatype face_weight =
      static_cast<type_real>(0.0); ///< Integration weight associated with the
                                   ///< face at the quadrature point
  NormalViewType face_normal = {
    static_cast<type_real>(0.0), static_cast<type_real>(0.0),
    static_cast<type_real>(0.0)
  }; ///< Normal vector to the face at
     ///< the quadrature point
};

/**
 * @brief Template specialization for composite Stacey Dirichlet boundary
 * condition
 *
 * @tparam DimensionTag Dimension of the spectral element where the quadrature
 * point is located
 * @tparam UseSIMD Boolean indicating whether to use SIMD instructions
 */
template <specfem::dimension::type DimensionTag, bool UseSIMD>
struct boundary<specfem::element::boundary_tag::composite_stacey_dirichlet,
                DimensionTag, UseSIMD>
    : public boundary<specfem::element::boundary_tag::stacey, DimensionTag,
                      UseSIMD> {
public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  /**
   * @brief SIMD data type.
   */
  using simd = specfem::datatype::simd<type_real, UseSIMD>;
  ///@}
  /**
   * @name Compile-time constants
   *
   */
  ///@{
  /**
   * @brief Tag indicating composite Stacey Dirichlet boundary condition.
   */
  constexpr static auto boundary_tag =
      specfem::element::boundary_tag::composite_stacey_dirichlet;
  ///@}

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
  boundary() = default;
  ///@}
};

template <specfem::dimension::type DimensionTag, bool UseSIMD>
KOKKOS_FUNCTION
specfem::point::boundary<specfem::element::boundary_tag::acoustic_free_surface,
                         DimensionTag, UseSIMD>::
    boundary(const specfem::point::boundary<
             specfem::element::boundary_tag::composite_stacey_dirichlet,
             DimensionTag, UseSIMD> &boundary) {
  this->tag = boundary.tag;
}

template <specfem::dimension::type DimensionTag, bool UseSIMD>
KOKKOS_FUNCTION specfem::point::boundary<specfem::element::boundary_tag::stacey,
                                         DimensionTag, UseSIMD>::
    boundary(const specfem::point::boundary<
             specfem::element::boundary_tag::composite_stacey_dirichlet,
             DimensionTag, UseSIMD> &boundary) {
  this->tag = boundary.tag;
  this->edge_weight = boundary.edge_weight;
  this->edge_normal = boundary.edge_normal;
}

template <bool UseSIMD>
KOKKOS_FUNCTION
specfem::point::boundary<specfem::element::boundary_tag::stacey,
                         specfem::dimension::type::dim3, UseSIMD>::
    boundary(const specfem::point::boundary<
             specfem::element::boundary_tag::composite_stacey_dirichlet,
             specfem::dimension::type::dim3, UseSIMD> &boundary) {
  this->tag = boundary.tag;
  this->face_weight = boundary.face_weight;
  this->face_normal = boundary.face_normal;
}

} // namespace point
} // namespace specfem
