#pragma once

#include <string>

namespace specfem {
namespace dimension {

/**
 * @brief Spatial dimension types for finite element simulations.
 *
 * Used as template parameters throughout the codebase to distinguish
 * between 2D and 3D implementations.
 */
enum class type {
  dim2, ///< 2D spatial dimension
  dim3  ///< 3D spatial dimension
};

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
 * using dim_2d = specfem::dimension::dimension<specfem::dimension::type::dim2>;
 * static_assert(dim_2d::dim == 2);
 * std::string name = dim_2d::to_string(); // "2D"
 * @endcode
 */
template <specfem::dimension::type DimensionTag> class dimension;

/**
 * @brief 2D dimension specialization.
 *
 * Provides compile-time constants for 2D finite element computations.
 */
template <> class dimension<specfem::dimension::type::dim2> {
public:
  static constexpr auto value =
      specfem::dimension::type::dim2; ///< Dimension type tag
  static constexpr int dim = 2;       ///< Spatial dimension count

  /**
   * @brief Get dimension as string.
   * @return "2D" string representation
   */
  static std::string to_string() { return "2D"; }
};

/**
 * @brief 3D dimension specialization.
 *
 * Provides compile-time constants for 3D finite element computations.
 */
template <> class dimension<specfem::dimension::type::dim3> {
public:
  static constexpr auto value =
      specfem::dimension::type::dim3; ///< Dimension type tag
  static constexpr int dim = 3;       ///< Spatial dimension count

  /**
   * @brief Get dimension as string.
   * @return "3D" string representation
   */
  static std::string to_string() { return "3D"; }
};

} // namespace dimension
} // namespace specfem
