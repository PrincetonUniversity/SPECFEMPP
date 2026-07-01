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
template <specfem::element::dimension_tag DimensionTag>
struct local_coordinates;

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

  /**
   * @brief Return a string representation of the local coordinates
   *
   * @return std::string
   */
  std::string print() const {
    std::ostringstream oss;
    oss << "ispec=" << ispec << ", xi=" << xi << ", gamma=" << gamma;
    return oss.str();
  }

  /**
   * @brief Whether the point lies inside the reference element.
   *
   * @return true if the element index is valid and every local coordinate
   * magnitude is within the reference element \f$ [-1, 1] \f$.
   */
  KOKKOS_INLINE_FUNCTION
  bool inside() const {
    return ispec >= 0 && Kokkos::abs(xi) <= type_real(1) &&
           Kokkos::abs(gamma) <= type_real(1);
  }

  /**
   * @brief Whether the point lies outside the reference element beyond @p
   * bound.
   *
   * Deliberately not the negation of @ref inside(): a located point in the
   * tolerance band (1 < |coord| <= bound) is neither inside nor outside. An
   * unlocated point (ispec < 0) is not in any element and therefore counts as
   * outside.
   *
   * @param bound Tolerance on the reference-element coordinate magnitude.
   * @return true if the element index is invalid (ispec < 0) or any local
   * coordinate magnitude exceeds @p bound.
   */
  KOKKOS_INLINE_FUNCTION
  bool outside(type_real bound) const {
    return ispec < 0 || Kokkos::abs(xi) > bound || Kokkos::abs(gamma) > bound;
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

  /**
   * @brief Return a string representation of the local coordinates
   *
   * @return std::string
   */
  std::string print() const {
    std::ostringstream oss;
    oss << "ispec=" << ispec << ", xi=" << xi << ", eta=" << eta
        << ", gamma=" << gamma;
    return oss.str();
  }

  /**
   * @brief Whether the point lies inside the reference element.
   *
   * @return true if the element index is valid and every local coordinate
   * magnitude is within the reference element \f$ [-1, 1] \f$.
   */
  KOKKOS_INLINE_FUNCTION
  bool inside() const {
    return ispec >= 0 && Kokkos::abs(xi) <= type_real(1) &&
           Kokkos::abs(eta) <= type_real(1) &&
           Kokkos::abs(gamma) <= type_real(1);
  }

  /**
   * @brief Whether the point lies outside the reference element beyond @p
   * bound.
   *
   * Deliberately not the negation of @ref inside(): a located point in the
   * tolerance band (1 < |coord| <= bound) is neither inside nor outside. An
   * unlocated point (ispec < 0) is not in any element and therefore counts as
   * outside.
   *
   * @param bound Tolerance on the reference-element coordinate magnitude.
   * @return true if the element index is invalid (ispec < 0) or any local
   * coordinate magnitude exceeds @p bound.
   */
  KOKKOS_INLINE_FUNCTION
  bool outside(type_real bound) const {
    return ispec < 0 || Kokkos::abs(xi) > bound || Kokkos::abs(eta) > bound ||
           Kokkos::abs(gamma) > bound;
  }
};

} // namespace point
} // namespace specfem
