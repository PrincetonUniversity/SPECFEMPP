#pragma once

#include "macros_impl/element_types.hpp"
#include "macros_impl/material_systems.hpp"
#include "macros_impl/medium_tags.hpp"
#include "macros_impl/utils.hpp"
#include "medium.hpp"
#include <boost/preprocessor.hpp>

/**
 * @name Element Tag macros
 *
 * @defgroup element_tags Element Tags
 *
 */
/// @{
#define DIMENSION_TAG_DIM2 (0, specfem::dimension::type::dim2, dim2)

#define MEDIUM_TAG_ELASTIC_PSV                                                 \
  (0, specfem::element::medium_tag::elastic_psv, elastic_psv)
#define MEDIUM_TAG_ELASTIC_SH                                                  \
  (1, specfem::element::medium_tag::elastic_sh, elastic_sh)
#define MEDIUM_TAG_ACOUSTIC                                                    \
  (2, specfem::element::medium_tag::acoustic, acoustic)
#define MEDIUM_TAG_POROELASTIC                                                 \
  (3, specfem::element::medium_tag::poroelastic, poroelastic)
#define MEDIUM_TAG_ELECTROMAGNETIC_TE                                          \
  (4, specfem::element::medium_tag::electromagnetic_te, electromagnetic_te)

#define PROPERTY_TAG_ISOTROPIC                                                 \
  (0, specfem::element::property_tag::isotropic, isotropic)
#define PROPERTY_TAG_ANISOTROPIC                                               \
  (1, specfem::element::property_tag::anisotropic, anisotropic)

#define BOUNDARY_TAG_NONE (0, specfem::element::boundary_tag::none, none)
#define BOUNDARY_TAG_STACEY (1, specfem::element::boundary_tag::stacey, stacey)
#define BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE                                     \
  (2, specfem::element::boundary_tag::acoustic_free_surface,                   \
   acoustic_free_surface)
#define BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET                                \
  (3, specfem::element::boundary_tag::composite_stacey_dirichlet,              \
   composite_stacey_dirichlet)

/**
 * @brief Macro to generate a list of medium types
 *
 */
#define MEDIUM_TAGS                                                            \
  ((DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_PSV))(                              \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_SH))(                            \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ACOUSTIC))(                              \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_POROELASTIC))(                           \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELECTROMAGNETIC_TE))

/**
 * @brief Macro to generate a list of material systems
 *
 */
#define MATERIAL_SYSTEMS                                                       \
  ((DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_PSV, PROPERTY_TAG_ISOTROPIC))(      \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_PSV, PROPERTY_TAG_ANISOTROPIC))( \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_SH, PROPERTY_TAG_ISOTROPIC))(    \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_SH, PROPERTY_TAG_ANISOTROPIC))(  \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ACOUSTIC, PROPERTY_TAG_ISOTROPIC))(      \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_POROELASTIC, PROPERTY_TAG_ISOTROPIC))(   \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELECTROMAGNETIC_TE,                      \
       PROPERTY_TAG_ISOTROPIC))

/**
 * @brief Macro to generate a list of element types
 *
 */
#define ELEMENT_TYPES                                                          \
  ((DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_PSV, PROPERTY_TAG_ISOTROPIC,        \
    BOUNDARY_TAG_NONE))((DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_PSV,           \
                         PROPERTY_TAG_ISOTROPIC, BOUNDARY_TAG_STACEY))(        \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_SH, PROPERTY_TAG_ISOTROPIC,      \
       BOUNDARY_TAG_NONE))((DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_SH,         \
                            PROPERTY_TAG_ISOTROPIC, BOUNDARY_TAG_STACEY))(     \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ACOUSTIC, PROPERTY_TAG_ISOTROPIC,        \
       BOUNDARY_TAG_NONE))((DIMENSION_TAG_DIM2, MEDIUM_TAG_ACOUSTIC,           \
                            PROPERTY_TAG_ISOTROPIC,                            \
                            BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE))(              \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ACOUSTIC, PROPERTY_TAG_ISOTROPIC,        \
       BOUNDARY_TAG_STACEY))((DIMENSION_TAG_DIM2, MEDIUM_TAG_ACOUSTIC,         \
                              PROPERTY_TAG_ISOTROPIC,                          \
                              BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))(       \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_PSV, PROPERTY_TAG_ANISOTROPIC,   \
       BOUNDARY_TAG_NONE))((DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_PSV,        \
                            PROPERTY_TAG_ANISOTROPIC, BOUNDARY_TAG_STACEY))(   \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_SH, PROPERTY_TAG_ANISOTROPIC,    \
       BOUNDARY_TAG_NONE))((DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_SH,         \
                            PROPERTY_TAG_ANISOTROPIC, BOUNDARY_TAG_STACEY))(   \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_POROELASTIC, PROPERTY_TAG_ISOTROPIC,     \
       BOUNDARY_TAG_NONE))((DIMENSION_TAG_DIM2, MEDIUM_TAG_ELECTROMAGNETIC_TE, \
                            PROPERTY_TAG_ISOTROPIC, BOUNDARY_TAG_NONE))

/**
 * @brief Filter sequence for different tags.
 *
 * This macro is to be only used in conjunction with @ref
 * CALL_MACRO_FOR_ALL_ELEMENT_TYPES or @ref CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS.
 *
 */
#define WHERE(...) (BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

/**
 * @brief Call a macro for all medium types
 *
 * Invoking CALL_MACRO_FOR_ALL_MEDIUM_TAGS(MACRO, seq) will call MACRO for all
 * medium types listed in macro sequence @ref MEDIUM_TAGS.
 *
 * @param MACRO The macro to be called. MACRO must have the following signature:
 * MACRO(DIMENSION_TAG, MEDIUM_TAG)
 *
 * @param seq A sequence filter for medium types. Sequences can be generated
 * using the @ref WHERE macro.
 *
 * @code
 *    #define CALL_FOO_ELASTIC(DIMENSION_TAG, MEDIUM_TAG)
 * \ foo<GET_TAG(DIMENTION_TAG), GET_TAG(MEDIUM_TAG)>();
 *
 *   CALL_MACRO_FOR_ALL_MEDIUM_TAGS(CALL_FOO, WHERE(DIMENSION_TAG_DIM2)
 * WHERE(MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH))
 * @endcode
 *
 *
 */
#define CALL_MACRO_FOR_ALL_MEDIUM_TAGS(MACRO, seq)                             \
  BOOST_PP_SEQ_FOR_EACH(_CALL_FOR_ONE_MEDIUM_TAG, MACRO,                       \
                        BOOST_PP_SEQ_FOR_EACH_PRODUCT(_CREATE_SEQ, seq))

/**
 * @brief Call a macro for all element types
 *
 * Invoking CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(MACRO, seq) will call MACRO for
 * all element types listed in macro sequence @ref MATERIAL_SYSTEMS.
 *
 * @param MACRO The macro to be called. MACRO must have the following signature:
 * MACRO(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, BOUNDARY_TAG)
 *
 * @param seq A sequence filter for element types. Sequences can be generated
 * using the @ref WHERE macro.
 *
 * @code
 *    #define CALL_FOO_ELASTIC(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG)
 * \ foo<GET_TAG(DIMENTION_TAG), GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG)>();
 *
 *   CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(CALL_FOO, WHERE(DIMENSION_TAG_DIM2)
 * WHERE(MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH)
 * WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC))
 * @endcode
 *
 *
 */
#define CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(MACRO, seq)                        \
  BOOST_PP_SEQ_FOR_EACH(_CALL_FOR_ONE_MATERIAL_SYSTEM, MACRO,                  \
                        BOOST_PP_SEQ_FOR_EACH_PRODUCT(_CREATE_SEQ, seq))

/**
 * @brief Call a macro for all element types
 *
 * Invoking CALL_MACRO_FOR_ALL_ELEMENT_TYPES(MACRO, seq) will call MACRO for all
 * element types listed in macro sequence @ref ELEMENT_TYPES.
 *
 * @param MACRO The macro to be called. MACRO must have the following signature:
 * MACRO(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, BOUNDARY_TAG)
 *
 * @param seq A sequence filter for element types. Sequences can be generated
 * using the @ref WHERE macro.
 *
 * @code
 *    #define CALL_FOO_ELASTIC(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG,
 * BOUNDARY_TAG) \ foo<GET_TAG(DIMENTION_TAG), GET_TAG(MEDIUM_TAG),
 * GET_TAG(PROPERTY_TAG), GET_TAG(BOUNDARY_TAG)>();
 *
 *   CALL_MACRO_FOR_ALL_ELEMENT_TYPES(CALL_FOO, WHERE(DIMENSION_TAG_DIM2)
 * WHERE(MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH)
 * WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC)
 * WHERE(BOUNDARY_TAG_NONE, BOUNDARY_TAG_STACEY))
 * @endcode
 *
 *
 */
#define CALL_MACRO_FOR_ALL_ELEMENT_TYPES(MACRO, seq)                           \
  BOOST_PP_SEQ_FOR_EACH(_CALL_FOR_ONE_ELEMENT_TYPE, MACRO,                     \
                        BOOST_PP_SEQ_FOR_EACH_PRODUCT(_CREATE_SEQ, seq))

namespace specfem {
namespace element {

/**
 * @brief A constexpr function to generate a list of medium types within the
 * simulation.
 *
 * This macro uses @ref MEDIUM_TAGS to generate a list of medium types
 * automatically.
 *
 * @return constexpr auto list of medium types
 */
constexpr auto medium_types() {
  // Use boost preprocessor library to generate a list of medium
  // types
  constexpr int total_medium_types = BOOST_PP_SEQ_SIZE(MEDIUM_TAGS);
  constexpr std::array<std::tuple<specfem::dimension::type, medium_tag>,
                       total_medium_types>
      medium_types{ _MAKE_CONSTEXPR_ARRAY(MEDIUM_TAGS,
                                          _MAKE_ARRAY_ELEM_MEDIUM) };

  return medium_types;
}

/**
 * @brief A constexpr function to generate a list of material systems within the
 * simulation
 *
 * This macro uses @ref MATERIAL_SYSTEMS to generate a list of material systems
 * automatically.
 *
 * @return constexpr auto list of material systems
 */
constexpr auto material_systems() {
  // Use boost preprocessor library to generate a list of
  // material systems
  constexpr int total_material_systems = BOOST_PP_SEQ_SIZE(MATERIAL_SYSTEMS);
  constexpr std::array<
      std::tuple<specfem::dimension::type, specfem::element::medium_tag,
                 specfem::element::property_tag>,
      total_material_systems>
      material_systems{ _MAKE_CONSTEXPR_ARRAY(MATERIAL_SYSTEMS,
                                              _MAKE_ARRAY_ELEM_MAT_SYS) };

  return material_systems;
}

/**
 * @brief A constexpr function to generate a list of element types within the
 * simulation
 *
 * This macro uses @ref ELEMENT_TYPES to generate a list of element types
 * automatically.
 *
 * @return constexpr auto list of element types
 */
constexpr auto element_types() {
  // Use boost preprocessor library to generate a list of
  // material systems
  constexpr int total_element_types = BOOST_PP_SEQ_SIZE(ELEMENT_TYPES);
  constexpr std::array<
      std::tuple<specfem::dimension::type, specfem::element::medium_tag,
                 specfem::element::property_tag,
                 specfem::element::boundary_tag>,
      total_element_types>
      material_systems{ _MAKE_CONSTEXPR_ARRAY(ELEMENT_TYPES,
                                              _MAKE_ARRAY_ELEM_ELEM) };

  return material_systems;
}

} // namespace element
} // namespace specfem
