#pragma once

#include "specfem/element.hpp"
#include "specfem/setup.hpp"
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
template <specfem::element::dimension_tag DimensionTag> struct local_coordinates;

//-------------------------- 2D Specializations ------------------------------//

/**
 * @brief 2D local coordinates for spectral elements
 *
 * Stores the element index and local coordinates (\f$\xi, \gamma\f$)
 * for a point within a 2D spectral element.
 */
template <> struct local_coordinates<specfem::element::dimension_tag::dim2> {
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
    static_assert(ViewType::rank() == 1, "ViewType must be rank 1");
    static_assert(ViewType::static_extent(0) == 2,
                  "ViewType must have extent 2 for 2D coordinates");
  }
};

//-------------------------- 3D Specializations ------------------------------//

/**
 * @brief 3D local coordinates for spectral elements
 *
 * Stores the element index and local coordinates (\f$\xi, \eta, \gamma\f$)
 * for a point within a 3D spectral element.
 */
template <> struct local_coordinates<specfem::element::dimension_tag::dim3> {
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
    static_assert(ViewType::rank() == 1, "ViewType must be rank 1");
    static_assert(ViewType::static_extent(0) == 3,
                  "ViewType must have extent 3 for 3D coordinates");
  }
};

} // namespace point
} // namespace specfem
