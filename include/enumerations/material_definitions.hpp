#pragma once

#include "macros_impl/array.hpp"
#include "macros_impl/utils.hpp"

/**
 * @name Element Tag macros
 *
 * @defgroup element_tags Element Tags
 *
 */
/// @{
#define DIMENSION_TAG_DIM2 (0, specfem::dimension::type::dim2, dim2)
#define DIMENSION_TAG_DIM3 (1, specfem::dimension::type::dim3, dim3)

#define MEDIUM_TAG_ELASTIC_PSV                                                 \
  (0, specfem::element::medium_tag::elastic_psv, elastic_psv)
#define MEDIUM_TAG_ELASTIC_SH                                                  \
  (1, specfem::element::medium_tag::elastic_sh, elastic_sh)
#define MEDIUM_TAG_ELASTIC_PSV_T                                               \
  (2, specfem::element::medium_tag::elastic_psv_t, elastic_psv_t)
#define MEDIUM_TAG_ACOUSTIC                                                    \
  (3, specfem::element::medium_tag::acoustic, acoustic)
#define MEDIUM_TAG_POROELASTIC                                                 \
  (4, specfem::element::medium_tag::poroelastic, poroelastic)
#define MEDIUM_TAG_ELECTROMAGNETIC_TE                                          \
  (5, specfem::element::medium_tag::electromagnetic_te, electromagnetic_te)

#define PROPERTY_TAG_ISOTROPIC                                                 \
  (0, specfem::element::property_tag::isotropic, isotropic)
#define PROPERTY_TAG_ANISOTROPIC                                               \
  (1, specfem::element::property_tag::anisotropic, anisotropic)
#define PROPERTY_TAG_ISOTROPIC_COSSERAT                                        \
  (2, specfem::element::property_tag::isotropic_cosserat, isotropic_cosserat)

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
#define MEDIUM_TAGS_DIM2                                                       \
  ((DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_PSV))(                              \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_SH))(                            \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_PSV_T))(                         \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ACOUSTIC))(                              \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_POROELASTIC))(                           \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELECTROMAGNETIC_TE))

#define MEDIUM_TAGS_DIM3 ((DIMENSION_TAG_DIM3, MEDIUM_TAG_ELASTIC_PSV))

#define MEDIUM_TAGS MEDIUM_TAGS_DIM2 MEDIUM_TAGS_DIM3

/**
 * @brief Macro to generate a list of material systems
 *
 */
#define MATERIAL_SYSTEMS_DIM2                                                  \
  ((DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_PSV, PROPERTY_TAG_ISOTROPIC))(      \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_PSV, PROPERTY_TAG_ANISOTROPIC))( \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_SH, PROPERTY_TAG_ISOTROPIC))(    \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_SH, PROPERTY_TAG_ANISOTROPIC))(  \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_PSV_T,                           \
       PROPERTY_TAG_ISOTROPIC_COSSERAT))(                                      \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ACOUSTIC, PROPERTY_TAG_ISOTROPIC))(      \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_POROELASTIC, PROPERTY_TAG_ISOTROPIC))(   \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELECTROMAGNETIC_TE,                      \
       PROPERTY_TAG_ISOTROPIC))

#define MATERIAL_SYSTEMS_DIM3                                                  \
  ((DIMENSION_TAG_DIM3, MEDIUM_TAG_ELASTIC_PSV, PROPERTY_TAG_ISOTROPIC))

#define MATERIAL_SYSTEMS MATERIAL_SYSTEMS_DIM2 MATERIAL_SYSTEMS_DIM3

/**
 * @brief Macro to generate a list of element types
 *
 */
#define ELEMENT_TYPES_DIM2                                                     \
  ((DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_PSV, PROPERTY_TAG_ISOTROPIC,        \
    BOUNDARY_TAG_NONE))((DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_PSV,           \
                         PROPERTY_TAG_ISOTROPIC, BOUNDARY_TAG_STACEY))(        \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_SH, PROPERTY_TAG_ISOTROPIC,      \
       BOUNDARY_TAG_NONE))((DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_SH,         \
                            PROPERTY_TAG_ISOTROPIC, BOUNDARY_TAG_STACEY))(     \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_PSV_T,                           \
       PROPERTY_TAG_ISOTROPIC_COSSERAT, BOUNDARY_TAG_NONE))(                   \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_PSV_T,                           \
       PROPERTY_TAG_ISOTROPIC_COSSERAT,                                        \
       BOUNDARY_TAG_STACEY))((DIMENSION_TAG_DIM2, MEDIUM_TAG_ACOUSTIC,         \
                              PROPERTY_TAG_ISOTROPIC, BOUNDARY_TAG_NONE))(     \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ACOUSTIC, PROPERTY_TAG_ISOTROPIC,        \
       BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE))(                                   \
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
       BOUNDARY_TAG_NONE))((DIMENSION_TAG_DIM2, MEDIUM_TAG_POROELASTIC,        \
                            PROPERTY_TAG_ISOTROPIC, BOUNDARY_TAG_STACEY))(     \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELECTROMAGNETIC_TE,                      \
       PROPERTY_TAG_ISOTROPIC, BOUNDARY_TAG_NONE))

#define ELEMENT_TYPES_DIM3                                                     \
  ((DIMENSION_TAG_DIM3, MEDIUM_TAG_ELASTIC_PSV, PROPERTY_TAG_ISOTROPIC,        \
    BOUNDARY_TAG_NONE))

#define ELEMENT_TYPES ELEMENT_TYPES_DIM2 ELEMENT_TYPES_DIM3

/**
 * @brief Tag getters. The macros are intended to be used only in @ref DECLARE
 * and @ref INSTANTIATE.
 */
#define _DIMENSION_TAG_ BOOST_PP_SEQ_TO_LIST((0))
#define _MEDIUM_TAG_ BOOST_PP_SEQ_TO_LIST((1))
#define _PROPERTY_TAG_ BOOST_PP_SEQ_TO_LIST((2))
#define _BOUNDARY_TAG_ BOOST_PP_SEQ_TO_LIST((3))

/**
 * @brief Declare for each tag.
 *
 * This macro is to be only used in conjunction with @ref FOR_EACH_IN_PRODUCT
 *
 */
#define DECLARE(...) BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)
/**
 * @brief Instantiate templates for each tag.
 *
 * This macro is to be only used in conjunction with @ref FOR_EACH_IN_PRODUCT
 *
 */
#define INSTANTIATE(...)                                                       \
  BOOST_PP_SEQ_TRANSFORM(_TRANSFORM_INSTANTIATE, _,                            \
                         BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

/**
 * @brief Capture existing variables as reference in the code block.
 */
#define CAPTURE(...) BOOST_PP_VARIADIC_TO_TUPLE(BOOST_PP_EMPTY(), __VA_ARGS__),

/**
 * @brief Converts tag arguments to a sequence of tag tuples,
 * e.g. DIMENSION_TAG(DIM2) expands to DIMENSION_TAG_DIM2
 */
#define DIMENSION_TAG(...)                                                     \
  BOOST_PP_SEQ_TRANSFORM(_TRANSFORM_TAGS, DIMENSION_TAG_,                      \
                         BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define MEDIUM_TAG(...)                                                        \
  BOOST_PP_SEQ_TRANSFORM(_TRANSFORM_TAGS, MEDIUM_TAG_,                         \
                         BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define PROPERTY_TAG(...)                                                      \
  BOOST_PP_SEQ_TRANSFORM(_TRANSFORM_TAGS, PROPERTY_TAG_,                       \
                         BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define BOUNDARY_TAG(...)                                                      \
  BOOST_PP_SEQ_TRANSFORM(_TRANSFORM_TAGS, BOUNDARY_TAG_,                       \
                         BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

/**
 * @brief Declare variables or run code for all material systems
 * listed in macro sequence @ref MATERIAL_SYSTEMS.
 * @param seq A sequence filter for material systems.
 * @param ... To declare variabled, use DECLARE() as the first argument,
 * e.g. DECLARE((IndexViewType, elements),
 *  (IndexViewType::HostMirror, h_elements),
 *  ((properties)((_MEDIUM_TAG_, _PROPERTY_TAG_)), value))
 * To capture existing variables as reference in the code block, add a tuple as
 * argument, e.g. CAPTURE(value, elements, h_elements). The last argument is the
 * code block to be executed.
 */
#define FOR_EACH_IN_PRODUCT(seq, ...)                                          \
  BOOST_PP_SEQ_FOR_EACH(                                                       \
      _FOR_ONE_TAG_SEQ, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__),                 \
      BOOST_PP_SEQ_FOR_EACH_PRODUCT(_CREATE_SEQ, BOOST_PP_TUPLE_TO_SEQ(seq)))

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
      medium_types{ _MAKE_CONSTEXPR_ARRAY(MEDIUM_TAGS) };

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
template <specfem::dimension::type DimensionTag>
constexpr auto material_systems();

template <> constexpr auto material_systems<specfem::dimension::type::dim2>() {
  // Use boost preprocessor library to generate a list of
  // material systems
  constexpr int total_material_systems =
      BOOST_PP_SEQ_SIZE(MATERIAL_SYSTEMS_DIM2);
  constexpr std::array<
      std::tuple<specfem::dimension::type, specfem::element::medium_tag,
                 specfem::element::property_tag>,
      total_material_systems>
      material_systems{ _MAKE_CONSTEXPR_ARRAY(MATERIAL_SYSTEMS_DIM2) };

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
template <specfem::dimension::type DimensionTag> constexpr auto element_types();

template <> constexpr auto element_types<specfem::dimension::type::dim2>() {
  // Use boost preprocessor library to generate a list of
  // material systems
  constexpr int total_element_types = BOOST_PP_SEQ_SIZE(ELEMENT_TYPES_DIM2);
  constexpr std::array<
      std::tuple<specfem::dimension::type, specfem::element::medium_tag,
                 specfem::element::property_tag,
                 specfem::element::boundary_tag>,
      total_element_types>
      material_systems{ _MAKE_CONSTEXPR_ARRAY(ELEMENT_TYPES_DIM2) };

  return material_systems;
}

} // namespace element
} // namespace specfem
