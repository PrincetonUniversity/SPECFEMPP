#pragma once

#include "medium.hpp"
#include <boost/preprocessor.hpp>

namespace specfem {
namespace element {

/**
 * @name Element Tag macros
 *
 * @defgroup element_tags Element Tags
 *
 */
/// @{
#define DIMENSION_TAG_DIM2 (0, specfem::dimension::type::dim2, dim2)

#define MEDIUM_TAG_ELASTIC_SV                                                  \
  (0, specfem::element::medium_tag::elastic_sv, elastic)
#define MEDIUM_TAG_ACOUSTIC                                                    \
  (1, specfem::element::medium_tag::acoustic, acoustic)

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

#define GET_ID(elem) BOOST_PP_TUPLE_ELEM(0, elem)
#define GET_TAG(elem) BOOST_PP_TUPLE_ELEM(1, elem)
#define GET_NAME(elem) BOOST_PP_TUPLE_ELEM(2, elem)

#define ADD_UNDERSCORE(s, data, elem) BOOST_PP_CAT(_, elem)

#define CREATE_VARIABLE_NAME(prefix, ...)                                      \
  BOOST_PP_SEQ_CAT((prefix)BOOST_PP_SEQ_TRANSFORM(                             \
      ADD_UNDERSCORE, _, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)))

/**
 * @brief Macro to generate a list of medium types
 *
 */
#define MEDIUM_TYPES                                                           \
  ((DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_SV))(                               \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ACOUSTIC))

#define MAKE_ARRAY_ELEM(s, data, elem)                                         \
  std::make_tuple(GET_TAG(BOOST_PP_TUPLE_ELEM(0, elem)),                       \
                  GET_TAG(BOOST_PP_TUPLE_ELEM(1, elem)))

#define MAKE_CONSTEXPR_ARRAY(seq)                                              \
  BOOST_PP_SEQ_ENUM(BOOST_PP_SEQ_TRANSFORM(MAKE_ARRAY_ELEM, _, seq))

/**
 * @brief A constexpr function to generate a list of medium types within the
 * simulation.
 *
 * This macro uses @ref MEDIUM_TYPES to generate a list of medium types
 * automatically.
 *
 * @return constexpr auto list of medium types
 */
constexpr auto medium_types() {
  // Use boost preprocessor library to generate a list of medium
  // types
  constexpr int total_medium_types = BOOST_PP_SEQ_SIZE(MEDIUM_TYPES);
  constexpr std::array<std::tuple<specfem::dimension::type, medium_tag>,
                       total_medium_types>
      medium_types{ MAKE_CONSTEXPR_ARRAY(MEDIUM_TYPES) };

  return medium_types;
}

#undef MAKE_ARRAY_ELEM

/**
 * @brief Macro to generate a list of material systems
 *
 */
#define MATERIAL_SYSTEMS                                                       \
  ((DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_SV, PROPERTY_TAG_ISOTROPIC))(       \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_SV, PROPERTY_TAG_ANISOTROPIC))(  \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ACOUSTIC, PROPERTY_TAG_ISOTROPIC))

#define MAKE_ARRAY_ELEM(s, data, elem)                                         \
  std::make_tuple(GET_TAG(BOOST_PP_TUPLE_ELEM(0, elem)),                       \
                  GET_TAG(BOOST_PP_TUPLE_ELEM(1, elem)),                       \
                  GET_TAG(BOOST_PP_TUPLE_ELEM(2, elem)))

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
      material_systems{ MAKE_CONSTEXPR_ARRAY(MATERIAL_SYSTEMS) };

  return material_systems;
}

#undef MAKE_ARRAY_ELEM

/**
 * @brief Macro to generate a list of element types
 *
 */
#define ELEMENT_TYPES                                                          \
  ((DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_SV, PROPERTY_TAG_ISOTROPIC,         \
    BOUNDARY_TAG_NONE))((DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_SV,            \
                         PROPERTY_TAG_ISOTROPIC, BOUNDARY_TAG_STACEY))(        \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ACOUSTIC, PROPERTY_TAG_ISOTROPIC,        \
       BOUNDARY_TAG_NONE))((DIMENSION_TAG_DIM2, MEDIUM_TAG_ACOUSTIC,           \
                            PROPERTY_TAG_ISOTROPIC,                            \
                            BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE))(              \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ACOUSTIC, PROPERTY_TAG_ISOTROPIC,        \
       BOUNDARY_TAG_STACEY))((DIMENSION_TAG_DIM2, MEDIUM_TAG_ACOUSTIC,         \
                              PROPERTY_TAG_ISOTROPIC,                          \
                              BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))(       \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_SV, PROPERTY_TAG_ANISOTROPIC,    \
       BOUNDARY_TAG_NONE))((DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_SV,         \
                            PROPERTY_TAG_ANISOTROPIC, BOUNDARY_TAG_STACEY))

#define MAKE_ARRAY_ELEM(s, data, elem)                                         \
  std::make_tuple(GET_TAG(BOOST_PP_TUPLE_ELEM(0, elem)),                       \
                  GET_TAG(BOOST_PP_TUPLE_ELEM(1, elem)),                       \
                  GET_TAG(BOOST_PP_TUPLE_ELEM(2, elem)),                       \
                  GET_TAG(BOOST_PP_TUPLE_ELEM(3, elem)))

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
      material_systems{ MAKE_CONSTEXPR_ARRAY(ELEMENT_TYPES) };

  return material_systems;
}

#undef MAKE_CONSTEXPR_ARRAY
#undef MAKE_ARRAY_ELEM

// Touch the following code at your own risk

#define MEDIUM_IN_TUPLE(s, elem, tuple)                                        \
  BOOST_PP_IF(                                                                 \
      BOOST_PP_EQUAL(BOOST_PP_TUPLE_SIZE(tuple), 2),                           \
      BOOST_PP_IF(                                                             \
          BOOST_PP_EQUAL(GET_ID(BOOST_PP_TUPLE_ELEM(0, tuple)),                \
                         GET_ID(BOOST_PP_TUPLE_ELEM(0, elem))),                \
          BOOST_PP_IF(BOOST_PP_EQUAL(GET_ID(BOOST_PP_TUPLE_ELEM(1, tuple)),    \
                                     GET_ID(BOOST_PP_TUPLE_ELEM(1, elem))),    \
                      1, 0),                                                   \
          0),                                                                  \
      0)

#define MAT_SYS_IN_TUPLE(s, elem, tuple)                                       \
  BOOST_PP_IF(                                                                 \
      BOOST_PP_EQUAL(BOOST_PP_TUPLE_SIZE(tuple), 3),                           \
      BOOST_PP_IF(                                                             \
          BOOST_PP_EQUAL(GET_ID(BOOST_PP_TUPLE_ELEM(0, tuple)),                \
                         GET_ID(BOOST_PP_TUPLE_ELEM(0, elem))),                \
          BOOST_PP_IF(BOOST_PP_EQUAL(GET_ID(BOOST_PP_TUPLE_ELEM(1, tuple)),    \
                                     GET_ID(BOOST_PP_TUPLE_ELEM(1, elem))),    \
                      BOOST_PP_IF(BOOST_PP_EQUAL(                              \
                                      GET_ID(BOOST_PP_TUPLE_ELEM(2, tuple)),   \
                                      GET_ID(BOOST_PP_TUPLE_ELEM(2, elem))),   \
                                  1, 0),                                       \
                      0),                                                      \
          0),                                                                  \
      0)

#define ELEM_IN_TUPLE(s, elem, tuple)                                          \
  BOOST_PP_IF(                                                                 \
      BOOST_PP_EQUAL(BOOST_PP_TUPLE_SIZE(tuple), 4),                           \
      BOOST_PP_IF(                                                             \
          BOOST_PP_EQUAL(GET_ID(BOOST_PP_TUPLE_ELEM(0, tuple)),                \
                         GET_ID(BOOST_PP_TUPLE_ELEM(0, elem))),                \
          BOOST_PP_IF(                                                         \
              BOOST_PP_EQUAL(GET_ID(BOOST_PP_TUPLE_ELEM(1, tuple)),            \
                             GET_ID(BOOST_PP_TUPLE_ELEM(1, elem))),            \
              BOOST_PP_IF(                                                     \
                  BOOST_PP_EQUAL(GET_ID(BOOST_PP_TUPLE_ELEM(2, tuple)),        \
                                 GET_ID(BOOST_PP_TUPLE_ELEM(2, elem))),        \
                  BOOST_PP_IF(                                                 \
                      BOOST_PP_EQUAL(GET_ID(BOOST_PP_TUPLE_ELEM(3, tuple)),    \
                                     GET_ID(BOOST_PP_TUPLE_ELEM(3, elem))),    \
                      1, 0),                                                   \
                  0),                                                          \
              0),                                                              \
          0),                                                                  \
      0)

#define OP_OR(s, state, elem) BOOST_PP_OR(state, elem)

#define MEDIUM_IN_SEQUENCE(elem)                                               \
  BOOST_PP_SEQ_FOLD_LEFT(                                                      \
      OP_OR, 0, BOOST_PP_SEQ_TRANSFORM(MEDIUM_IN_TUPLE, elem, MEDIUM_TYPES))

#define MAT_SYS_IN_SEQUENCE(elem)                                              \
  BOOST_PP_SEQ_FOLD_LEFT(                                                      \
      OP_OR, 0,                                                                \
      BOOST_PP_SEQ_TRANSFORM(MAT_SYS_IN_TUPLE, elem, MATERIAL_SYSTEMS))

#define ELEM_IN_SEQUENCE(elem)                                                 \
  BOOST_PP_SEQ_FOLD_LEFT(                                                      \
      OP_OR, 0, BOOST_PP_SEQ_TRANSFORM(ELEM_IN_TUPLE, elem, ELEMENT_TYPES))

/**
 * @brief Filter sequence for different tags.
 *
 * This macro is to be only used in conjunction with @ref
 * CALL_MACRO_FOR_ALL_ELEMENT_TYPES or @ref CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS.
 *
 */
#define WHERE(...) (BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define CALL_FOR_ONE_MEDIUM_TYPE(s, MACRO, elem)                               \
  BOOST_PP_IF(MEDIUM_IN_SEQUENCE((BOOST_PP_SEQ_ENUM(elem))), MACRO,            \
              EMPTY_MACRO)                                                     \
  (BOOST_PP_TUPLE_ELEM(0, (BOOST_PP_SEQ_ENUM(elem))),                          \
   BOOST_PP_TUPLE_ELEM(1, (BOOST_PP_SEQ_ENUM(elem))))

#define CALL_FOR_ONE_MATERIAL_SYSTEM(s, MACRO, elem)                           \
  BOOST_PP_IF(MAT_SYS_IN_SEQUENCE((BOOST_PP_SEQ_ENUM(elem))), MACRO,           \
              EMPTY_MACRO)                                                     \
  (BOOST_PP_TUPLE_ELEM(0, (BOOST_PP_SEQ_ENUM(elem))),                          \
   BOOST_PP_TUPLE_ELEM(1, (BOOST_PP_SEQ_ENUM(elem))),                          \
   BOOST_PP_TUPLE_ELEM(2, (BOOST_PP_SEQ_ENUM(elem))))

#define CALL_FOR_ONE_ELEMENT_TYPE(s, MACRO, elem)                              \
  BOOST_PP_IF(ELEM_IN_SEQUENCE((BOOST_PP_SEQ_ENUM(elem))), MACRO, EMPTY_MACRO) \
  (BOOST_PP_TUPLE_ELEM(0, (BOOST_PP_SEQ_ENUM(elem))),                          \
   BOOST_PP_TUPLE_ELEM(1, (BOOST_PP_SEQ_ENUM(elem))),                          \
   BOOST_PP_TUPLE_ELEM(2, (BOOST_PP_SEQ_ENUM(elem))),                          \
   BOOST_PP_TUPLE_ELEM(3, (BOOST_PP_SEQ_ENUM(elem))))

#define EMPTY_MACRO(...)

#define CREATE_SEQ(r, elem) (elem)

/**
 * @brief Call a macro for all medium types
 *
 * Invoking CALL_MACRO_FOR_ALL_MEDIUM_TAGS(MACRO, seq) will call MACRO for all
 * medium types listed in macro sequence @ref MEDIUM_TYPES.
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
 * WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC))
 * @endcode
 *
 *
 */
#define CALL_MACRO_FOR_ALL_MEDIUM_TAGS(MACRO, seq)                             \
  BOOST_PP_SEQ_FOR_EACH(CALL_FOR_ONE_MEDIUM_TYPE, MACRO,                       \
                        BOOST_PP_SEQ_FOR_EACH_PRODUCT(CREATE_SEQ, seq))

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
 * WHERE(MEDIUM_TAG_ELASTIC_SV) WHERE(PROPERTY_TAG_ISOTROPIC,
 * PROPERTY_TAG_ANISOTROPIC))
 * @endcode
 *
 *
 */
#define CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(MACRO, seq)                        \
  BOOST_PP_SEQ_FOR_EACH(CALL_FOR_ONE_MATERIAL_SYSTEM, MACRO,                   \
                        BOOST_PP_SEQ_FOR_EACH_PRODUCT(CREATE_SEQ, seq))

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
 * WHERE(MEDIUM_TAG_ELASTIC_SV) WHERE(PROPERTY_TAG_ISOTROPIC,
 * PROPERTY_TAG_ANISOTROPIC) WHERE(BOUNDARY_TAG_NONE, BOUNDARY_TAG_STACEY))
 * @endcode
 *
 *
 */
#define CALL_MACRO_FOR_ALL_ELEMENT_TYPES(MACRO, seq)                           \
  BOOST_PP_SEQ_FOR_EACH(CALL_FOR_ONE_ELEMENT_TYPE, MACRO,                      \
                        BOOST_PP_SEQ_FOR_EACH_PRODUCT(CREATE_SEQ, seq))

} // namespace element
} // namespace specfem
