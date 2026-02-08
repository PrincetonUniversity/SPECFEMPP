#pragma once

#include "specfem/element.hpp"
#include "specfem/data_access.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <cstddef>

namespace specfem {
namespace point {

/**
 * @brief Struct to store global coordinates associated with a quadrature point
 *
 *
 * @tparam DimensionTag Dimension of the element where the quadrature point is
 * located
 */
template <specfem::element::dimension_tag DimensionTag>
struct global_coordinates;

/**
 * @brief Euclidean distance between two global coordinates
 *
 * @tparam DimensionTag Dimension of the element where the quadrature point is
 * located
 * @param p1 Coordinates of the first point
 * @param p2 Coordinates of the second point
 * @return type_real Distance between the two points
 */
template <specfem::element::dimension_tag DimensionTag>
KOKKOS_FUNCTION type_real
distance(const specfem::point::global_coordinates<DimensionTag> &p1,
         const specfem::point::global_coordinates<DimensionTag> &p2);

//-------------------------- 2D Specializations ------------------------------//

/**
 * @brief 2D global coordinates
 *
 * Stores the physical coordinates (\f$x, z\f$) for a point in 2D space.
 */
template <> struct global_coordinates<specfem::element::dimension_tag::dim2> {
  type_real x; ///< Global coordinate \f$ x \f$
  type_real z; ///< Global coordinate \f$ z \f$

  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION
  global_coordinates() = default;

  /**
   * @brief Construct a new global coordinates object
   *
   * @param x Global coordinate \f$ x \f$
   * @param z Global coordinate \f$ z \f$
   */
  KOKKOS_FUNCTION
  global_coordinates(const type_real &x, const type_real &z) : x(x), z(z) {}

  /**
   * @brief Construct a new global coordinates object from Kokkos array
   *
   * @param coords Kokkos 1D array containing [x, z] coordinates
   */
  template <typename ViewType>
  KOKKOS_FUNCTION global_coordinates(const ViewType &coords)
      : x(coords[0]), z(coords[1]) {
    static_assert(ViewType::rank() == 1, "ViewType must be rank 1");
    static_assert(ViewType::static_extent(0) == 2,
                  "ViewType must have extent 2 for 2D coordinates");
  }

  /**
   * @brief Get coordinates as a Kokkos::Array
   *
   * @return Kokkos::Array<type_real, 2> containing [x, z] coordinates
   */
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<type_real, 2> coordinates() const {
    return Kokkos::Array<type_real, 2>{x, z};
  }
};

//-------------------------- 3D Specializations ------------------------------//

/**
 * @brief 3D global coordinates
 *
 * Stores the physical coordinates (\f$x, y, z\f$) for a point in 3D space.
 */
template <> struct global_coordinates<specfem::element::dimension_tag::dim3> {
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim3;
  constexpr static auto data_class =
      specfem::data_access::DataClassType::global_coordinates;

  type_real x; ///< Global coordinate \f$ x \f$
  type_real y; ///< Global coordinate \f$ y \f$
  type_real z; ///< Global coordinate \f$ z \f$

  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION
  global_coordinates() = default;

  /**
   * @brief Construct a new global coordinates object
   *
   * @param x Global coordinate \f$ x \f$
   * @param y Global coordinate \f$ y \f$
   * @param z Global coordinate \f$ z \f$
   */
  KOKKOS_FUNCTION
  global_coordinates(const type_real &x, const type_real &y, const type_real &z)
      : x(x), y(y), z(z) {}

  /**
   * @brief Construct a new global coordinates object from Kokkos array
   *
   * @param coords Kokkos 1D array containing [x, y, z] coordinates
   */
  template <typename ViewType>
  KOKKOS_FUNCTION global_coordinates(const ViewType &coords)
      : x(coords[0]), y(coords[1]), z(coords[2]) {
    static_assert(ViewType::rank() == 1, "ViewType must be rank 1");
    static_assert(ViewType::static_extent(0) == 3,
                  "ViewType must have extent 3 for 3D coordinates");
  }

  /**
   * @brief Get coordinates as a Kokkos::Array
   *
   * @return Kokkos::Array<type_real, 3> containing [x, y, z] coordinates
   */
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<type_real, 3> coordinates() const {
    return Kokkos::Array<type_real, 3>{x, y, z};
  }
};

} // namespace point
} // namespace specfem

template <specfem::element::dimension_tag Dimension>
std::ostream &
operator<<(std::ostream &s,
           const specfem::point::global_coordinates<Dimension> &point);
