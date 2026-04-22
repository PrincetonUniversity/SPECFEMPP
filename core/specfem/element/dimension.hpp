#pragma once

#include "tags.hpp"
#include <string>

namespace specfem {
namespace element {
/**
 * @brief Compile-time dimension traits and utilities.
 *
 * Provides dimension-specific constants and string conversion.
 * Template specializations define dimension-dependent behavior.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 *
 * @code
 * // Get dimension info at compile time
 * using dim_2d =
 * specfem::element::dimension<specfem::element::dimension_tag::dim2>;
 * static_assert(dim_2d::dim == 2);
 * std::string name = dim_2d::to_string(); // "2D"
 * @endcode
 */
template <specfem::element::dimension_tag DimensionTag> class dimension;

/**
 * @brief 2D dimension specialization.
 *
 * Provides compile-time constants for 2D finite element computations.
 */
template <> class dimension<specfem::element::dimension_tag::dim2> {
public:
  static constexpr auto value =
      specfem::element::dimension_tag::dim2; ///< Dimension type tag
  static constexpr int dim = 2;              ///< Spatial dimension count
};

/**
 * @brief 3D dimension specialization.
 *
 * Provides compile-time constants for 3D finite element computations.
 */
template <> class dimension<specfem::element::dimension_tag::dim3> {
public:
  static constexpr auto value =
      specfem::element::dimension_tag::dim3; ///< Dimension type tag
  static constexpr int dim = 3;              ///< Spatial dimension count
};

} // namespace element
} // namespace specfem
