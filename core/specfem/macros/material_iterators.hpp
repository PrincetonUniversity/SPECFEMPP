#pragma once

/**
 * @file material_iterators.hpp
 * @brief Macros for material and element definitions
 */

#include "enum_tags.hpp"
#include "macros_impl/array.hpp"
#include "macros_impl/utils.hpp"
#include "specfem/enums.hpp"

#include <boost/preprocessor.hpp>

/**
 * @defgroup dimension_tag_macros Dimension Tag Macros
 * @brief Macros for dimension tags.
 * @{
 */
/**
 * @brief Dimension tag for 2D
 */
#define DIMENSION_TAG_DIM2                                                     \
  (0, specfem::element::dimension_tag::dim2, dim2, _ENUM_ID_DIMENSION_TAG)

/**
 * @brief Dimension tag for 3D
 */
#define DIMENSION_TAG_DIM3                                                     \
  (1, specfem::element::dimension_tag::dim3, dim3, _ENUM_ID_DIMENSION_TAG)
/** @} */

/**
 * @defgroup medium_tag_macros Medium Tag Macros
 * @brief Macros for medium tags.
 * @{
 */
/**
 * @brief Medium tag for Elastic P-SV
 */
#define MEDIUM_TAG_ELASTIC_PSV                                                 \
  (0, specfem::element::medium_tag::elastic_psv, elastic_psv,                  \
   _ENUM_ID_MEDIUM_TAG)

/**
 * @brief Medium tag for Elastic SH
 */
#define MEDIUM_TAG_ELASTIC_SH                                                  \
  (1, specfem::element::medium_tag::elastic_sh, elastic_sh, _ENUM_ID_MEDIUM_TAG)

/**
 * @brief Medium tag for Elastic P-SV Transverse Isotropic
 */
#define MEDIUM_TAG_ELASTIC_PSV_T                                               \
  (2, specfem::element::medium_tag::elastic_psv_t, elastic_psv_t,              \
   _ENUM_ID_MEDIUM_TAG)

/**
 * @brief Medium tag for Acoustic
 */
#define MEDIUM_TAG_ACOUSTIC                                                    \
  (3, specfem::element::medium_tag::acoustic, acoustic, _ENUM_ID_MEDIUM_TAG)

/**
 * @brief Medium tag for Poroelastic
 */
#define MEDIUM_TAG_POROELASTIC                                                 \
  (4, specfem::element::medium_tag::poroelastic, poroelastic,                  \
   _ENUM_ID_MEDIUM_TAG)

/**
 * @brief Medium tag for Electromagnetic TE
 */
#define MEDIUM_TAG_ELECTROMAGNETIC_TE                                          \
  (5, specfem::element::medium_tag::electromagnetic_te, electromagnetic_te,    \
   _ENUM_ID_MEDIUM_TAG)

/**
 * @brief Medium tag for Elastic
 */
#define MEDIUM_TAG_ELASTIC                                                     \
  (6, specfem::element::medium_tag::elastic, elastic, _ENUM_ID_MEDIUM_TAG)

/**
 * @brief Medium tag for Elastic Spin
 */
#define MEDIUM_TAG_ELASTIC_SPIN                                                \
  (7, specfem::element::medium_tag::elastic_spin, elastic_spin,                \
   _ENUM_ID_MEDIUM_TAG)
/** @} */

/**
 * @defgroup property_tag_macros Property Tag Macros
 * @brief Macros for property tags.
 * @{
 */
/**
 * @brief Property tag for Isotropic
 */
#define PROPERTY_TAG_ISOTROPIC                                                 \
  (0, specfem::element::property_tag::isotropic, isotropic,                    \
   _ENUM_ID_PROPERTY_TAG)

/**
 * @brief Property tag for Anisotropic
 */
#define PROPERTY_TAG_ANISOTROPIC                                               \
  (1, specfem::element::property_tag::anisotropic, anisotropic,                \
   _ENUM_ID_PROPERTY_TAG)

/**
 * @brief Property tag for Isotropic Cosserat
 */
#define PROPERTY_TAG_ISOTROPIC_COSSERAT                                        \
  (2, specfem::element::property_tag::isotropic_cosserat, isotropic_cosserat,  \
   _ENUM_ID_PROPERTY_TAG)
/** @} */

#define ATTENUATION_TAG_NONE                                                   \
  (0, specfem::element::attenuation_tag::none, none, _ENUM_ID_ATTENUATION_TAG)

#define ATTENUATION_TAG_CONSTANT_ISOTROPIC                                     \
  (1, specfem::element::attenuation_tag::constant_isotropic,                   \
   constant_isotropic, _ENUM_ID_ATTENUATION_TAG)

/**
 * @defgroup boundary_tag_macros Boundary Tag Macros
 * @brief Macros for boundary tags.
 * @{
 */
/**
 * @brief Boundary tag for None
 */
#define BOUNDARY_TAG_NONE                                                      \
  (0, specfem::element::boundary_tag::none, none, _ENUM_ID_BOUNDARY_TAG)

/**
 * @brief Boundary tag for Stacey
 */
#define BOUNDARY_TAG_STACEY                                                    \
  (1, specfem::element::boundary_tag::stacey, stacey, _ENUM_ID_BOUNDARY_TAG)

/**
 * @brief Boundary tag for Acoustic Free Surface
 */
#define BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE                                     \
  (2, specfem::element::boundary_tag::acoustic_free_surface,                   \
   acoustic_free_surface, _ENUM_ID_BOUNDARY_TAG)

/**
 * @brief Boundary tag for Composite Stacey Dirichlet
 */
#define BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET                                \
  (3, specfem::element::boundary_tag::composite_stacey_dirichlet,              \
   composite_stacey_dirichlet, _ENUM_ID_BOUNDARY_TAG)
/** @} */

/**
 * @defgroup material_iterator_macros Material Iterator Macros
 * @brief Macros for defining material and element tags.
 * @{
 */
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

#define MEDIUM_TAGS_DIM3                                                       \
  ((DIMENSION_TAG_DIM3, MEDIUM_TAG_ELASTIC))(                                  \
      (DIMENSION_TAG_DIM3, MEDIUM_TAG_ACOUSTIC))(                              \
      (DIMENSION_TAG_DIM3, MEDIUM_TAG_ELASTIC_SPIN))

#define MEDIUM_TAGS MEDIUM_TAGS_DIM2 MEDIUM_TAGS_DIM3

/**
 * @brief Macro to generate a list of material systems
 *
 */
#define MATERIAL_SYSTEMS_DIM2                                                  \
  ((DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_PSV, PROPERTY_TAG_ISOTROPIC,        \
    ATTENUATION_TAG_NONE))((DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_PSV,        \
                            PROPERTY_TAG_ANISOTROPIC, ATTENUATION_TAG_NONE))(  \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_SH, PROPERTY_TAG_ISOTROPIC,      \
       ATTENUATION_TAG_NONE))(                                                 \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_SH, PROPERTY_TAG_ANISOTROPIC,    \
       ATTENUATION_TAG_NONE))((DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_PSV_T,   \
                               PROPERTY_TAG_ISOTROPIC_COSSERAT,                \
                               ATTENUATION_TAG_NONE))(                         \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ACOUSTIC, PROPERTY_TAG_ISOTROPIC,        \
       ATTENUATION_TAG_NONE))((DIMENSION_TAG_DIM2, MEDIUM_TAG_POROELASTIC,     \
                               PROPERTY_TAG_ISOTROPIC, ATTENUATION_TAG_NONE))( \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELECTROMAGNETIC_TE,                      \
       PROPERTY_TAG_ISOTROPIC, ATTENUATION_TAG_NONE))(                         \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_PSV, PROPERTY_TAG_ISOTROPIC,     \
       ATTENUATION_TAG_CONSTANT_ISOTROPIC))(                                   \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_PSV, PROPERTY_TAG_ANISOTROPIC,   \
       ATTENUATION_TAG_CONSTANT_ISOTROPIC))(                                   \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_SH, PROPERTY_TAG_ISOTROPIC,      \
       ATTENUATION_TAG_CONSTANT_ISOTROPIC))(                                   \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC_SH, PROPERTY_TAG_ANISOTROPIC,    \
       ATTENUATION_TAG_CONSTANT_ISOTROPIC))(                                   \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ACOUSTIC, PROPERTY_TAG_ISOTROPIC,        \
       ATTENUATION_TAG_CONSTANT_ISOTROPIC))(                                   \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_POROELASTIC, PROPERTY_TAG_ISOTROPIC,     \
       ATTENUATION_TAG_CONSTANT_ISOTROPIC))

#define MATERIAL_SYSTEMS_DIM3                                                  \
  ((DIMENSION_TAG_DIM3, MEDIUM_TAG_ELASTIC, PROPERTY_TAG_ISOTROPIC,            \
    ATTENUATION_TAG_NONE))((DIMENSION_TAG_DIM3, MEDIUM_TAG_ELASTIC,            \
                            PROPERTY_TAG_ANISOTROPIC, ATTENUATION_TAG_NONE))(  \
      (DIMENSION_TAG_DIM3, MEDIUM_TAG_ACOUSTIC, PROPERTY_TAG_ISOTROPIC,        \
       ATTENUATION_TAG_NONE))((DIMENSION_TAG_DIM3, MEDIUM_TAG_ELASTIC,         \
                               PROPERTY_TAG_ISOTROPIC,                         \
                               ATTENUATION_TAG_CONSTANT_ISOTROPIC))(           \
      (DIMENSION_TAG_DIM3, MEDIUM_TAG_ACOUSTIC, PROPERTY_TAG_ISOTROPIC,        \
       ATTENUATION_TAG_CONSTANT_ISOTROPIC))(                                   \
      (DIMENSION_TAG_DIM3, MEDIUM_TAG_ELASTIC_SPIN,                            \
       PROPERTY_TAG_ISOTROPIC_COSSERAT, ATTENUATION_TAG_NONE))

#define MATERIAL_SYSTEMS MATERIAL_SYSTEMS_DIM2 MATERIAL_SYSTEMS_DIM3

/**
 * @brief Macro to generate a list of medium-property systems (unique
 * combinations of dimension, medium, and property tags, without attenuation).
 * Derived from MATERIAL_SYSTEMS by removing the attenuation tag and
 * deduplicating.
 */
#define MEDIUM_PROPERTY_SYSTEMS_DIM2                                           \
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

#define MEDIUM_PROPERTY_SYSTEMS_DIM3                                           \
  ((DIMENSION_TAG_DIM3, MEDIUM_TAG_ELASTIC, PROPERTY_TAG_ISOTROPIC))(          \
      (DIMENSION_TAG_DIM3, MEDIUM_TAG_ELASTIC, PROPERTY_TAG_ANISOTROPIC))(     \
      (DIMENSION_TAG_DIM3, MEDIUM_TAG_ACOUSTIC, PROPERTY_TAG_ISOTROPIC))(      \
      (DIMENSION_TAG_DIM3, MEDIUM_TAG_ELASTIC_SPIN,                            \
       PROPERTY_TAG_ISOTROPIC_COSSERAT))

#define MEDIUM_PROPERTY_SYSTEMS                                                \
  MEDIUM_PROPERTY_SYSTEMS_DIM2 MEDIUM_PROPERTY_SYSTEMS_DIM3

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
  ((DIMENSION_TAG_DIM3, MEDIUM_TAG_ELASTIC, PROPERTY_TAG_ISOTROPIC,            \
    BOUNDARY_TAG_NONE))((DIMENSION_TAG_DIM3, MEDIUM_TAG_ACOUSTIC,              \
                         PROPERTY_TAG_ISOTROPIC, BOUNDARY_TAG_NONE))(          \
      (DIMENSION_TAG_DIM3, MEDIUM_TAG_ELASTIC_SPIN,                            \
       PROPERTY_TAG_ISOTROPIC_COSSERAT, BOUNDARY_TAG_NONE))

#define ELEMENT_TYPES ELEMENT_TYPES_DIM2 ELEMENT_TYPES_DIM3

/** @} */

/**
 * @brief Tag getters. The macros are intended to be used only in @ref DECLARE
 * and @ref INSTANTIATE.
 */
#define _DIMENSION_TAG_ BOOST_PP_SEQ_TO_LIST((0))
#define _MEDIUM_TAG_ BOOST_PP_SEQ_TO_LIST((1))
#define _PROPERTY_TAG_ BOOST_PP_SEQ_TO_LIST((2))
#define _ATTENUATION_TAG_ BOOST_PP_SEQ_TO_LIST((3))
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

#define ATTENUATION_TAG(...)                                                   \
  BOOST_PP_SEQ_TRANSFORM(_TRANSFORM_TAGS, ATTENUATION_TAG_,                    \
                         BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define FOR_EACH_IN_PRODUCT(seq, ...)                                          \
  BOOST_PP_SEQ_FOR_EACH(_FOR_ONE_TAG_SEQ,                                      \
                        (seq)BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__),            \
                        _GET_VALID_SEQS(BOOST_PP_TUPLE_SIZE(seq)))

#include "interface_iterators.hpp"
