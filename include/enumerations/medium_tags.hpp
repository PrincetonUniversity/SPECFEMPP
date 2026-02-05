#pragma once

#include "specfem/element.hpp"
#include "specfem/macros.hpp"
#include <array>
#include <boost/preprocessor.hpp>
#include <tuple>

namespace specfem::element {
/**
 * @brief Generates a list of medium types within the simulation.
 *
 * Uses `MEDIUM_TAGS` to generate the list automatically.
 *
 * @tparam DimensionTag The dimension of the simulation.
 * @return A constexpr array of medium types.
 */
template <specfem::element::dimension_tag DimensionTag>
constexpr auto medium_types();

/**
 * @brief Generates a list of medium types for 2D simulations.
 *
 * @return A constexpr array of medium types for 2D.
 */
template <>
constexpr auto medium_types<specfem::element::dimension_tag::dim2>() {
  // Use boost preprocessor library to generate a list of medium
  // types
  constexpr int total_medium_types = BOOST_PP_SEQ_SIZE(MEDIUM_TAGS_DIM2);
  constexpr std::array<
      std::tuple<specfem::element::dimension_tag, specfem::element::medium_tag>,
      total_medium_types>
      medium_types{ _MAKE_CONSTEXPR_ARRAY(MEDIUM_TAGS_DIM2) };

  return medium_types;
}

/**
 * @brief Generates a list of medium types for 3D simulations.
 *
 * @return A constexpr array of medium types for 3D.
 */
template <>
constexpr auto medium_types<specfem::element::dimension_tag::dim3>() {
  // Use boost preprocessor library to generate a list of medium
  // types
  constexpr int total_medium_types = BOOST_PP_SEQ_SIZE(MEDIUM_TAGS_DIM3);
  constexpr std::array<
      std::tuple<specfem::element::dimension_tag, specfem::element::medium_tag>,
      total_medium_types>
      medium_types{ _MAKE_CONSTEXPR_ARRAY(MEDIUM_TAGS_DIM3) };

  return medium_types;
}

/**
 * @brief Generates a list of material systems within the simulation.
 *
 * Uses `MATERIAL_SYSTEMS` to generate the list automatically.
 *
 * @tparam DimensionTag The dimension of the simulation.
 * @return A constexpr array of material systems.
 */
template <specfem::element::dimension_tag DimensionTag>
constexpr auto material_systems();

/**
 * @brief Generates a list of material systems for 2D simulations.
 *
 * @return A constexpr array of material systems for 2D.
 */
template <>
constexpr auto material_systems<specfem::element::dimension_tag::dim2>() {
  // Use boost preprocessor library to generate a list of
  // material systems
  constexpr int total_material_systems =
      BOOST_PP_SEQ_SIZE(MATERIAL_SYSTEMS_DIM2);
  constexpr std::array<
      std::tuple<specfem::element::dimension_tag, specfem::element::medium_tag,
                 specfem::element::property_tag,
                 specfem::element::attenuation_tag>,
      total_material_systems>
      material_systems{ _MAKE_CONSTEXPR_ARRAY(MATERIAL_SYSTEMS_DIM2) };

  return material_systems;
}

/**
 * @brief Generates a list of material systems for 3D simulations.
 *
 * @return A constexpr array of material systems for 3D.
 */
template <>
constexpr auto material_systems<specfem::element::dimension_tag::dim3>() {
  // material systems
  constexpr int total_material_systems =
      BOOST_PP_SEQ_SIZE(MATERIAL_SYSTEMS_DIM3);
  constexpr std::array<
      std::tuple<specfem::element::dimension_tag, specfem::element::medium_tag,
                 specfem::element::property_tag,
                 specfem::element::attenuation_tag>,
      total_material_systems>
      material_systems{ _MAKE_CONSTEXPR_ARRAY(MATERIAL_SYSTEMS_DIM3) };

  return material_systems;
}

/**
 * @brief Generates a list of element types within the simulation.
 *
 * Uses `ELEMENT_TYPES` to generate the list automatically.
 *
 * @tparam DimensionTag The dimension of the simulation.
 * @return A constexpr array of element types.
 */
template <specfem::element::dimension_tag DimensionTag>
constexpr auto element_types();

/**
 * @brief Generates a list of element types for 2D simulations.
 *
 * @return A constexpr array of element types for 2D.
 */
template <>
constexpr auto element_types<specfem::element::dimension_tag::dim2>() {
  // material systems
  constexpr int total_element_types = BOOST_PP_SEQ_SIZE(ELEMENT_TYPES_DIM2);
  constexpr std::array<
      std::tuple<specfem::element::dimension_tag, specfem::element::medium_tag,
                 specfem::element::property_tag,
                 specfem::element::boundary_tag>,
      total_element_types>
      element_types{ _MAKE_CONSTEXPR_ARRAY(ELEMENT_TYPES_DIM2) };

  return element_types;
}

/**
 * @brief Generates a list of element types for 3D simulations.
 *
 * @return A constexpr array of element types for 3D.
 */
template <>
constexpr auto element_types<specfem::element::dimension_tag::dim3>() {
  // material systems
  constexpr int total_element_types = BOOST_PP_SEQ_SIZE(ELEMENT_TYPES_DIM3);
  constexpr std::array<
      std::tuple<specfem::element::dimension_tag, specfem::element::medium_tag,
                 specfem::element::property_tag,
                 specfem::element::boundary_tag>,
      total_element_types>
      element_types{ _MAKE_CONSTEXPR_ARRAY(ELEMENT_TYPES_DIM3) };

  return element_types;
}

} // namespace specfem::element
