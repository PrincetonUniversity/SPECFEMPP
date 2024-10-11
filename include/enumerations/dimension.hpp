#pragma once

#include <string>

namespace specfem {
namespace dimension {

/**
 * @brief Dimension type enumeration
 *
 */
enum class type { dim2, dim3 };

/**
 * @brief Dimension
 *
 * @tparam DimensionType dimension type enumeration
 */
template <specfem::dimension::type DimensionType> class dimension;

/**
 * @brief 2D dimension specialization
 *
 */
template <> class dimension<specfem::dimension::type::dim2> {
public:
  static constexpr auto value =
      specfem::dimension::type::dim2; ///< dimension type
  static constexpr int dim = 2;       ///< Number of dimensions
  static std::string to_string() {
    return "2D";
  } ///< Convert dimension to string
};

/**
 * @brief 3D dimension specialization
 *
 */
template <> class dimension<specfem::dimension::type::dim3> {
public:
  static constexpr auto value =
      specfem::dimension::type::dim3; ///< dimension type
  static constexpr int dim = 3;       ///< Number of dimensions
  static std::string to_string() {
    return "3D";
  } ///< Convert dimension to string
};

} // namespace dimension
} // namespace specfem
