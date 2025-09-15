#pragma once

#include "enumerations/dimension.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <cstddef>

namespace specfem {
namespace point {

/**
 * @brief Struct to store local coordinates associated with a quadrature point
 *
 * @tparam DimensionTag Dimension of the element where the quadrature point is
 * located
 */
template <specfem::dimension::type DimensionTag> struct local_coordinates;

/**
 * @brief Struct to store global coordinates associated with a quadrature point
 *
 *
 * @tparam DimensionTag Dimension of the element where the quadrature point is
 * located
 */
template <specfem::dimension::type DimensionTag> struct global_coordinates;

/**
 * @brief Euclidean distance between two global coordinates
 *
 * @tparam DimensionTag Dimension of the element where the quadrature point is
 * located
 * @param p1 Coordinates of the first point
 * @param p2 Coordinates of the second point
 * @return type_real Distance between the two points
 */
template <specfem::dimension::type DimensionTag>
KOKKOS_FUNCTION type_real
distance(const specfem::point::global_coordinates<DimensionTag> &p1,
         const specfem::point::global_coordinates<DimensionTag> &p2);

//-------------------------- 2D Specializations ------------------------------//

/**
 * @brief Template specialization for 2D elements
 *
 */
template <> struct local_coordinates<specfem::dimension::type::dim2> {
  int ispec;       ///< Index of the spectral element
  type_real xi;    ///< Local coordinate \f$ \xi \f$
  type_real gamma; ///< Local coordinate \f$ \gamma \f$

  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION
  local_coordinates() = default;

  /**
   * @brief Construct a new local coordinates object
   *
   * @param ispec Index of the spectral element
   * @param xi Local coordinate \f$ \xi \f$
   * @param gamma Local coordinate \f$ \gamma \f$
   */
  KOKKOS_FUNCTION
  local_coordinates(const int &ispec, const type_real &xi,
                    const type_real &gamma)
      : ispec(ispec), xi(xi), gamma(gamma) {}

  /**
   * @brief Construct a new local coordinates object from element index and
   * Kokkos array
   *
   * @param ispec Index of the spectral element
   * @param coords Kokkos 1D array containing [xi, gamma] coordinates
   */
  template <typename ViewType>
  KOKKOS_FUNCTION local_coordinates(const int &ispec, const ViewType &coords)
      : ispec(ispec), xi(coords[0]), gamma(coords[1]) {
    static_assert(ViewType::rank == 1, "ViewType must be rank 1");
    static_assert(ViewType::static_extent(0) == 2,
                  "ViewType must have extent 2 for 2D coordinates");
  }
};

/**
 * @brief Template specialization for 2D elements
 *
 */
template <> struct global_coordinates<specfem::dimension::type::dim2> {
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
    static_assert(ViewType::rank == 1, "ViewType must be rank 1");
    static_assert(ViewType::static_extent(0) == 2,
                  "ViewType must have extent 2 for 2D coordinates");
  }
};

//-------------------------- 3D Specializations ------------------------------//

/**
 * @brief Template specialization for 3D elements
 *
 */
template <> struct local_coordinates<specfem::dimension::type::dim3> {
  int ispec;       ///< Index of the spectral element
  type_real xi;    ///< Local coordinate \f$ \xi \f$
  type_real eta;   ///< Local coordinate \f$ \eta \f$
  type_real gamma; ///< Local coordinate \f$ \gamma \f$

  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION
  local_coordinates() = default;

  /**
   * @brief Construct a new local coordinates object
   *
   * @param ispec Index of the spectral element
   * @param xi Local coordinate \f$ \xi \f$
   * @param eta Local coordinate \f$ \eta \f$
   * @param gamma Local coordinate \f$ \gamma \f$
   */
  KOKKOS_FUNCTION
  local_coordinates(const int &ispec, const type_real &xi, const type_real &eta,
                    const type_real &gamma)
      : ispec(ispec), xi(xi), eta(eta), gamma(gamma) {}

  /**
   * @brief Construct a new local coordinates object from element index and
   * Kokkos array
   *
   * @param ispec Index of the spectral element
   * @param coords Kokkos 1D array containing [xi, eta, gamma] coordinates
   */
  template <typename ViewType>
  KOKKOS_FUNCTION local_coordinates(const int &ispec, const ViewType &coords)
      : ispec(ispec), xi(coords[0]), eta(coords[1]), gamma(coords[2]) {
    static_assert(ViewType::rank == 1, "ViewType must be rank 1");
    static_assert(ViewType::static_extent(0) == 3,
                  "ViewType must have extent 3 for 3D coordinates");
  }
};

/**
 * @brief Template specialization for 3D elements
 *
 */
template <> struct global_coordinates<specfem::dimension::type::dim3> {
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
    static_assert(ViewType::rank == 1, "ViewType must be rank 1");
    static_assert(ViewType::static_extent(0) == 3,
                  "ViewType must have extent 3 for 3D coordinates");
  }
};

} // namespace point
} // namespace specfem

template <specfem::dimension::type Dimension>
std::ostream &
operator<<(std::ostream &s,
           const specfem::point::global_coordinates<Dimension> &point);
