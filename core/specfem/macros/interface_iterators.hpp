#pragma once

/**
 * @file interface_iterators.hpp
 * @brief Macros for interface and connection definitions
 */

#include "enum_tags.hpp"
#include "macros_impl/array.hpp"
#include "macros_impl/utils.hpp"
#include "specfem/enums.hpp"
#include <boost/preprocessor.hpp>

/**
 * @defgroup interface_iterator_macros Interface Iterator Macros
 * @brief Macros for defining interface and connection tags.
 * @{
 */

/**
 * @defgroup connection_tag_macros Connection Tag Macros
 * @brief Macros for connection tags.
 * @{
 */
/**
 * @brief Strongly conforming connection tag
 */
#define CONNECTION_TAG_STRONGLY_CONFORMING                                     \
  (0, specfem::element_connections::type::strongly_conforming,                 \
   strongly_conforming, _ENUM_ID_CONNECTION_TAG)

/**
 * @brief Weakly conforming connection tag
 */
#define CONNECTION_TAG_WEAKLY_CONFORMING                                       \
  (1, specfem::element_connections::type::weakly_conforming,                   \
   weakly_conforming, _ENUM_ID_CONNECTION_TAG)

/**
 * @brief Non-conforming connection tag
 */
#define CONNECTION_TAG_NONCONFORMING                                           \
  (2, specfem::element_connections::type::nonconforming, nonconforming,        \
   _ENUM_ID_CONNECTION_TAG)
/** @} */

/**
 * @defgroup interface_tag_macros Interface Tag Macros
 * @brief Macros for interface tags.
 * @{
 */
/**
 * @brief Elastic-Acoustic interface tag
 */
#define INTERFACE_TAG_ELASTIC_ACOUSTIC                                         \
  (0, specfem::element_coupling::interface_tag::elastic_acoustic,              \
   elastic_acoustic, _ENUM_ID_INTERFACE_TAG)

/**
 * @brief Acoustic-Elastic interface tag
 */
#define INTERFACE_TAG_ACOUSTIC_ELASTIC                                         \
  (1, specfem::element_coupling::interface_tag::acoustic_elastic,              \
   acoustic_elastic, _ENUM_ID_INTERFACE_TAG)
/** @} */

/**
 * @defgroup flux_scheme_tag_macros Flux Scheme Tag Macros
 * @brief Macros for flux schemes.
 * @{
 */
/**
 * @brief Natural flux scheme tag
 */
#define FLUX_SCHEME_TAG_NATURAL                                                \
  (0, specfem::element_coupling::flux_scheme_tag::natural, natural,            \
   _ENUM_ID_FLUX_SCHEME_TAG)

/**
 * @brief Symmetric interior penalty flux scheme tag
 */
#define FLUX_SCHEME_TAG_SYMMETRIC_INTERIOR_PENALTY                             \
  (1, specfem::element_coupling::flux_scheme_tag::symmetric_interior_penalty,  \
   symmetric_interior_penalty, _ENUM_ID_FLUX_SCHEME_TAG)
/** @} */

/// \cond
#define _MAKE_INTERFACE_TUPLE(r, product) BOOST_PP_SEQ_TO_TUPLE(product)

#define _GENERATE_INTERFACE(seqs)                                              \
  (BOOST_PP_SEQ_FOR_EACH_PRODUCT(_MAKE_INTERFACE_TUPLE, seqs))

/**
 * @brief Converts interface tag arguments to a sequence of tag tuples
 */
#define INTERFACE_TAG(...)                                                     \
  BOOST_PP_SEQ_TRANSFORM(_TRANSFORM_TAGS, INTERFACE_TAG_,                      \
                         BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

/**
 * @brief Converts connection tag arguments to a sequence of tag tuples
 */
#define CONNECTION_TAG(...)                                                    \
  BOOST_PP_SEQ_TRANSFORM(_TRANSFORM_TAGS, CONNECTION_TAG_,                     \
                         BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

/**
 * @brief Converts flux scheme tag arguments to a sequence of tag tuples
 */
#define FLUX_SCHEME_TAG(...)                                                   \
  BOOST_PP_SEQ_TRANSFORM(_TRANSFORM_TAGS, FLUX_SCHEME_TAG_,                    \
                         BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

/**
 * @brief Tag getters. The macros are intended to be used only in @ref DECLARE
 * and @ref INSTANTIATE.
 */
#define _CONNECTION_TAG_ BOOST_PP_SEQ_TO_LIST((1))
#define _INTERFACE_TAG_ BOOST_PP_SEQ_TO_LIST((2))
#define _FLUX_SCHEME_TAG_ BOOST_PP_SEQ_TO_LIST((4))

/**
 * @brief List of interface systems
 */
#define INTERFACE_SYSTEMS                                                      \
  ((DIMENSION_TAG_DIM2, CONNECTION_TAG_WEAKLY_CONFORMING,                      \
    INTERFACE_TAG_ELASTIC_ACOUSTIC))((DIMENSION_TAG_DIM2,                      \
                                      CONNECTION_TAG_WEAKLY_CONFORMING,        \
                                      INTERFACE_TAG_ACOUSTIC_ELASTIC))(        \
      (DIMENSION_TAG_DIM3, CONNECTION_TAG_WEAKLY_CONFORMING,                   \
       INTERFACE_TAG_ELASTIC_ACOUSTIC))((DIMENSION_TAG_DIM3,                   \
                                         CONNECTION_TAG_WEAKLY_CONFORMING,     \
                                         INTERFACE_TAG_ACOUSTIC_ELASTIC))

/**
 * @brief List of edges with interfaces
 */
#define EDGES                                                                  \
  ((DIMENSION_TAG_DIM2, CONNECTION_TAG_WEAKLY_CONFORMING,                      \
    INTERFACE_TAG_ELASTIC_ACOUSTIC, BOUNDARY_TAG_NONE))(                       \
      (DIMENSION_TAG_DIM2, CONNECTION_TAG_WEAKLY_CONFORMING,                   \
       INTERFACE_TAG_ELASTIC_ACOUSTIC, BOUNDARY_TAG_STACEY))(                  \
      (DIMENSION_TAG_DIM2, CONNECTION_TAG_WEAKLY_CONFORMING,                   \
       INTERFACE_TAG_ACOUSTIC_ELASTIC, BOUNDARY_TAG_NONE))(                    \
      (DIMENSION_TAG_DIM2, CONNECTION_TAG_WEAKLY_CONFORMING,                   \
       INTERFACE_TAG_ACOUSTIC_ELASTIC, BOUNDARY_TAG_STACEY))(                  \
      (DIMENSION_TAG_DIM2, CONNECTION_TAG_WEAKLY_CONFORMING,                   \
       INTERFACE_TAG_ACOUSTIC_ELASTIC, BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE))(   \
      (DIMENSION_TAG_DIM2, CONNECTION_TAG_WEAKLY_CONFORMING,                   \
       INTERFACE_TAG_ACOUSTIC_ELASTIC,                                         \
       BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))(                              \
      (DIMENSION_TAG_DIM2, CONNECTION_TAG_NONCONFORMING,                       \
       INTERFACE_TAG_ELASTIC_ACOUSTIC, BOUNDARY_TAG_NONE))(                    \
      (DIMENSION_TAG_DIM2, CONNECTION_TAG_NONCONFORMING,                       \
       INTERFACE_TAG_ELASTIC_ACOUSTIC, BOUNDARY_TAG_STACEY))(                  \
      (DIMENSION_TAG_DIM2, CONNECTION_TAG_NONCONFORMING,                       \
       INTERFACE_TAG_ACOUSTIC_ELASTIC, BOUNDARY_TAG_NONE))(                    \
      (DIMENSION_TAG_DIM2, CONNECTION_TAG_NONCONFORMING,                       \
       INTERFACE_TAG_ACOUSTIC_ELASTIC, BOUNDARY_TAG_STACEY))(                  \
      (DIMENSION_TAG_DIM2, CONNECTION_TAG_NONCONFORMING,                       \
       INTERFACE_TAG_ACOUSTIC_ELASTIC, BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE))(   \
      (DIMENSION_TAG_DIM2, CONNECTION_TAG_NONCONFORMING,                       \
       INTERFACE_TAG_ACOUSTIC_ELASTIC,                                         \
       BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))(                              \
      (DIMENSION_TAG_DIM3, CONNECTION_TAG_WEAKLY_CONFORMING,                   \
       INTERFACE_TAG_ELASTIC_ACOUSTIC, BOUNDARY_TAG_NONE))(                    \
      (DIMENSION_TAG_DIM3, CONNECTION_TAG_WEAKLY_CONFORMING,                   \
       INTERFACE_TAG_ACOUSTIC_ELASTIC, BOUNDARY_TAG_NONE))

/**
 * @brief List of edges with interfaces and flux scheme tag
 */
#define EDGES_AND_FLUX_SCHEME                                                  \
  ((DIMENSION_TAG_DIM2, CONNECTION_TAG_WEAKLY_CONFORMING,                      \
    INTERFACE_TAG_ELASTIC_ACOUSTIC, BOUNDARY_TAG_NONE,                         \
    FLUX_SCHEME_TAG_NATURAL))((DIMENSION_TAG_DIM2,                             \
                               CONNECTION_TAG_WEAKLY_CONFORMING,               \
                               INTERFACE_TAG_ELASTIC_ACOUSTIC,                 \
                               BOUNDARY_TAG_STACEY, FLUX_SCHEME_TAG_NATURAL))( \
      (DIMENSION_TAG_DIM2, CONNECTION_TAG_WEAKLY_CONFORMING,                   \
       INTERFACE_TAG_ACOUSTIC_ELASTIC, BOUNDARY_TAG_NONE,                      \
       FLUX_SCHEME_TAG_NATURAL))(                                              \
      (DIMENSION_TAG_DIM2, CONNECTION_TAG_WEAKLY_CONFORMING,                   \
       INTERFACE_TAG_ACOUSTIC_ELASTIC, BOUNDARY_TAG_STACEY,                    \
       FLUX_SCHEME_TAG_NATURAL))(                                              \
      (DIMENSION_TAG_DIM2, CONNECTION_TAG_WEAKLY_CONFORMING,                   \
       INTERFACE_TAG_ACOUSTIC_ELASTIC, BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,     \
       FLUX_SCHEME_TAG_NATURAL))(                                              \
      (DIMENSION_TAG_DIM2, CONNECTION_TAG_WEAKLY_CONFORMING,                   \
       INTERFACE_TAG_ACOUSTIC_ELASTIC,                                         \
       BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET, FLUX_SCHEME_TAG_NATURAL))(     \
      (DIMENSION_TAG_DIM2, CONNECTION_TAG_NONCONFORMING,                       \
       INTERFACE_TAG_ELASTIC_ACOUSTIC, BOUNDARY_TAG_NONE,                      \
       FLUX_SCHEME_TAG_NATURAL))(                                              \
      (DIMENSION_TAG_DIM2, CONNECTION_TAG_NONCONFORMING,                       \
       INTERFACE_TAG_ELASTIC_ACOUSTIC, BOUNDARY_TAG_STACEY,                    \
       FLUX_SCHEME_TAG_NATURAL))(                                              \
      (DIMENSION_TAG_DIM2, CONNECTION_TAG_NONCONFORMING,                       \
       INTERFACE_TAG_ACOUSTIC_ELASTIC, BOUNDARY_TAG_NONE,                      \
       FLUX_SCHEME_TAG_NATURAL))(                                              \
      (DIMENSION_TAG_DIM2, CONNECTION_TAG_NONCONFORMING,                       \
       INTERFACE_TAG_ACOUSTIC_ELASTIC, BOUNDARY_TAG_STACEY,                    \
       FLUX_SCHEME_TAG_NATURAL))(                                              \
      (DIMENSION_TAG_DIM2, CONNECTION_TAG_NONCONFORMING,                       \
       INTERFACE_TAG_ACOUSTIC_ELASTIC, BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,     \
       FLUX_SCHEME_TAG_NATURAL))(                                              \
      (DIMENSION_TAG_DIM2, CONNECTION_TAG_NONCONFORMING,                       \
       INTERFACE_TAG_ACOUSTIC_ELASTIC,                                         \
       BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET, FLUX_SCHEME_TAG_NATURAL))(     \
      (DIMENSION_TAG_DIM3, CONNECTION_TAG_WEAKLY_CONFORMING,                   \
       INTERFACE_TAG_ELASTIC_ACOUSTIC, BOUNDARY_TAG_NONE,                      \
       FLUX_SCHEME_TAG_NATURAL))(                                              \
      (DIMENSION_TAG_DIM3, CONNECTION_TAG_WEAKLY_CONFORMING,                   \
       INTERFACE_TAG_ACOUSTIC_ELASTIC, BOUNDARY_TAG_NONE,                      \
       FLUX_SCHEME_TAG_NATURAL))(                                              \
      (DIMENSION_TAG_DIM3, CONNECTION_TAG_NONCONFORMING,                       \
       INTERFACE_TAG_ELASTIC_ACOUSTIC, BOUNDARY_TAG_NONE,                      \
       FLUX_SCHEME_TAG_NATURAL))((DIMENSION_TAG_DIM3,                          \
                                  CONNECTION_TAG_NONCONFORMING,                \
                                  INTERFACE_TAG_ACOUSTIC_ELASTIC,              \
                                  BOUNDARY_TAG_NONE, FLUX_SCHEME_TAG_NATURAL))
/// \endcond
/** @} */
