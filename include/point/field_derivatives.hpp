#pragma once

#include "datatypes/point_view.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {

/**
 * @brief Store field derivatives for a quadrature point
 *
 * The field derivatives are given by:
 * \f$ du_{i,k} = \partial_i u_k \f$
 *
 * @tparam DimensionTag The dimension of the element where the quadrature point
 * is located
 * @tparam MediumTag The medium of the element where the quadrature point is
 * located
 * @tparam UseSIMD Use SIMD instructions
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
struct field_derivatives {

  /**
   * @name Compile time constants
   *
   */
  ///@{
  constexpr static bool is_point_field_derivatives = true;
  static constexpr int components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;
  constexpr static auto medium_tag = MediumTag; ///< Medium tag for the element
  constexpr static auto dimension = DimensionTag; ///< Dimension of the element
  constexpr static int num_dimensions =
      specfem::element::attributes<DimensionTag, MediumTag>::dimension;
  ///@}

  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type

  using ViewType =
      specfem::datatype::VectorPointViewType<type_real, components,
                                             num_dimensions,
                                             UseSIMD>; ///< Underlying view type
                                                       ///< to store the field
                                                       ///< derivatives
  ///@}

  ViewType du; ///< View to store the field derivatives.

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION field_derivatives() = default;

  /**
   * @brief Constructor
   *
   * @param du Field derivatives
   */
  KOKKOS_FUNCTION field_derivatives(const ViewType &du) : du(du) {}
  ///@}
};

} // namespace point
} // namespace specfem
