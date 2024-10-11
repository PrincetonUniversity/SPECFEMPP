#pragma once

#include "enumerations/dimension.hpp"

namespace specfem {
namespace element {

constexpr int ntypes = 2; ///< Number of element types

/**
 * @brief Medium tag enumeration
 *
 */
enum class medium_tag { elastic, acoustic, poroelastic };

/**
 * @brief Property tag enumeration
 *
 */
enum class property_tag { isotropic };
} // namespace element

namespace medium {

/**
 * @brief Medium
 *
 * @tparam Dimension dimension type
 * @tparam MediumTag medium tag
 * @tparam PropertyTag property tag
 */
template <specfem::dimension::type Dimension,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag =
              specfem::element::property_tag::isotropic>
class medium;

/**
 * @brief 2D Elastic, Isotropic medium specialization
 *
 */
template <>
class medium<specfem::dimension::type::dim2,
             specfem::element::medium_tag::elastic,
             specfem::element::property_tag::isotropic> {
public:
  static constexpr auto dimension =
      specfem::dimension::type::dim2; ///< dimension type
  static constexpr auto medium_tag =
      specfem::element::medium_tag::elastic; ///< medium tag
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic; ///< property tag
  static constexpr int components =
      2; ///< Number of components for the field inside the medium
  static std::string to_string() {
    return "Elastic, Isotropic";
  } ///< Convert medium to string
};

/**
 * @brief 2D Acoustic, Isotropic medium specialization
 *
 */
template <>
class medium<specfem::dimension::type::dim2,
             specfem::element::medium_tag::acoustic,
             specfem::element::property_tag::isotropic> {
public:
  static constexpr auto dimension =
      specfem::dimension::type::dim2; ///< dimension type
  static constexpr auto medium_tag =
      specfem::element::medium_tag::acoustic; ///< medium tag
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic; ///< property tag
  static constexpr int components =
      1; ///< Number of components for the field inside the medium
  static std::string to_string() {
    return "Acoustic, Isotropic";
  } ///< Convert medium to string
};

} // namespace medium
} // namespace specfem
