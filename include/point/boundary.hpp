#ifndef _POINT_BOUNDARY_HPP
#define _POINT_BOUNDARY_HPP

#include "datatypes/point_view.hpp"
#include "datatypes/simd.hpp"
#include "enumerations/boundary.hpp"
#include "enumerations/dimension.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {
/**
 * @brief Struct to store boundary conditions (and boundary related information)
 * associated with a quadrature point
 *
 * @tparam BoundaryTag Tag indicating the type of boundary condition
 * @DimensionType Dimension of the spectral element where the quadrature point
 * is located
 * @tparam UseSIMD Boolean indicating whether to use SIMD instructions
 */
template <specfem::element::boundary_tag BoundaryTag,
          specfem::dimension::type DimensionType, bool UseSIMD>
struct boundary;

/**
 * @brief Template specialization for no boundary condition
 *
 * @tparam DimensionType Dimension of the spectral element where the quadrature
 * point is located
 * @tparam UseSIMD Boolean indicating whether to use SIMD instructions
 */
template <specfem::dimension::type DimensionType, bool UseSIMD>
struct boundary<specfem::element::boundary_tag::none, DimensionType, UseSIMD> {
private:
  // We use simd_like vector to store tags. Tags are stored as enums, so a simd
  // type is ill-defined for them. However, we use scalar array types of size
  // simd<type_real>::size() to store them. The goal of this approach is to use
  // tags to mask a type_real simd vector and perform SIMD operations on those
  // SIMD vectors.
  using value_type = typename specfem::datatype::simd_like<
      specfem::element::boundary_tag_container, type_real,
      UseSIMD>::datatype; ///< Datatype for storing values. Is a scalar if
                          ///< UseSIMD is false, otherwise is a SIMD like
                          ///< vector.

public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD data type
  ///@}

  /**
   * @name Compile-time constants
   *
   */
  ///@{
  constexpr static auto boundary_tag =
      specfem::element::boundary_tag::none; ///< Tag indicating no boundary
                                            ///< condition
  constexpr static bool isPointBoundaryType =
      true; ///< Flag indicating this is a point boundary type
  constexpr static auto dimension =
      DimensionType; ///< Dimension of the spectral element
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

  value_type tag; ///< Tag indicating the type of boundary condition at the
                  ///< quadrature point
};

/**
 * @brief Template specialization for acoustic free surface boundary condition
 *
 * @tparam DimensionType Dimension of the spectral element where the quadrature
 * point is located
 * @tparam UseSIMD Boolean indicating whether to use SIMD instructions
 */
template <specfem::dimension::type DimensionType, bool UseSIMD>
struct boundary<specfem::element::boundary_tag::acoustic_free_surface,
                DimensionType, UseSIMD>
    : public boundary<specfem::element::boundary_tag::none, DimensionType,
                      UseSIMD> {
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
           DimensionType, UseSIMD> &boundary);
  ///@}
};

/**
 * @brief Template specialization for Stacey boundary condition
 *
 * @tparam DimensionType Dimension of the spectral element where the quadrature
 * point is located
 * @tparam UseSIMD Boolean indicating whether to use SIMD instructions
 */
template <specfem::dimension::type DimensionType, bool UseSIMD>
struct boundary<specfem::element::boundary_tag::stacey, DimensionType, UseSIMD>
    : public boundary<specfem::element::boundary_tag::acoustic_free_surface,
                      DimensionType, UseSIMD> {
private:
  constexpr static int num_dimensions =
      specfem::dimension::dimension<DimensionType>::dim;
  /**
   * @name Private Typedefs
   *
   */
  ///@{
  using NormalViewType =
      specfem::datatype::ScalarPointViewType<type_real, num_dimensions,
                                             UseSIMD>; ///< View type to store
                                                       ///< the normal vector to
                                                       ///< the edge at the
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
           DimensionType, UseSIMD> &boundary);
  ///@}

  datatype edge_weight =
      static_cast<type_real>(0.0); ///< Integration weight associated with the
                                   ///< edge at the quadrature point
  NormalViewType edge_normal = {
    static_cast<type_real>(0.0), static_cast<type_real>(0.0)
  }; ///< Normal vector to the edge at
     ///< the quadrature point
};

/**
 * @brief Template specialization for composite Stacey Dirichlet boundary
 * condition
 *
 * @tparam DimensionType Dimension of the spectral element where the quadrature
 * point is located
 * @tparam UseSIMD Boolean indicating whether to use SIMD instructions
 */
template <specfem::dimension::type DimensionType, bool UseSIMD>
struct boundary<specfem::element::boundary_tag::composite_stacey_dirichlet,
                DimensionType, UseSIMD>
    : public boundary<specfem::element::boundary_tag::stacey, DimensionType,
                      UseSIMD> {
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

template <specfem::dimension::type DimensionType, bool UseSIMD>
KOKKOS_FUNCTION
specfem::point::boundary<specfem::element::boundary_tag::acoustic_free_surface,
                         DimensionType, UseSIMD>::
    boundary(const specfem::point::boundary<
             specfem::element::boundary_tag::composite_stacey_dirichlet,
             DimensionType, UseSIMD> &boundary) {
  this->tag = boundary.tag;
}

template <specfem::dimension::type DimensionType, bool UseSIMD>
KOKKOS_FUNCTION specfem::point::boundary<specfem::element::boundary_tag::stacey,
                                         DimensionType, UseSIMD>::
    boundary(const specfem::point::boundary<
             specfem::element::boundary_tag::composite_stacey_dirichlet,
             DimensionType, UseSIMD> &boundary) {
  this->tag = boundary.tag;
  this->edge_weight = boundary.edge_weight;
  this->edge_normal = boundary.edge_normal;
}

} // namespace point
} // namespace specfem
#endif
