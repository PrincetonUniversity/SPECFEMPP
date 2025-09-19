#pragma once

#include "enum_tags.hpp"
#include "macros_impl/array.hpp"
#include "macros_impl/utils.hpp"

#define CONNECTION_TAG_STRONGLY_CONFORMING                                     \
  (0, specfem::connections::type::strongly_conforming, strongly_conforming,    \
   _ENUM_ID_CONNECTION_TAG)

#define CONNECTION_TAG_WEAKLY_CONFORMING                                       \
  (1, specfem::connections::type::weakly_conforming, weakly_conforming,        \
   _ENUM_ID_CONNECTION_TAG)

#define INTERFACE_TAG_ELASTIC_ACOUSTIC                                         \
  (0, specfem::interface::interface_tag::elastic_acoustic, elastic_acoustic,   \
   _ENUM_ID_INTERFACE_TAG)
#define INTERFACE_TAG_ACOUSTIC_ELASTIC                                         \
  (1, specfem::interface::interface_tag::acoustic_elastic, acoustic_elastic,   \
   _ENUM_ID_INTERFACE_TAG)

#define _MAKE_INTERFACE_TUPLE(r, product) BOOST_PP_SEQ_TO_TUPLE(product)

#define _GENERATE_INTERFACE(seqs)                                              \
  (BOOST_PP_SEQ_FOR_EACH_PRODUCT(_MAKE_INTERFACE_TUPLE, seqs))

#define INTERFACE_TAG(...)                                                     \
  BOOST_PP_SEQ_TRANSFORM(_TRANSFORM_TAGS, INTERFACE_TAG_,                      \
                         BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define CONNECTION_TAG(...)                                                    \
  BOOST_PP_SEQ_TRANSFORM(_TRANSFORM_TAGS, CONNECTION_TAG_,                     \
                         BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

/**
 * @brief Tag getters. The macros are intended to be used only in @ref DECLARE
 * and @ref INSTANTIATE.
 */
#define _CONNECTION_TAG_ BOOST_PP_SEQ_TO_LIST((1))
#define _INTERFACE_TAG_ BOOST_PP_SEQ_TO_LIST((2))

#define INTERFACE_SYSTEMS                                                      \
  ((DIMENSION_TAG_DIM2, CONNECTION_TAG_WEAKLY_CONFORMING,                      \
    INTERFACE_TAG_ELASTIC_ACOUSTIC))((DIMENSION_TAG_DIM2,                      \
                                      CONNECTION_TAG_WEAKLY_CONFORMING,        \
                                      INTERFACE_TAG_ACOUSTIC_ELASTIC))

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
       BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

namespace specfem::interface {

template <specfem::dimension::type DimensionTag> constexpr auto edges();

template <> constexpr auto edges<specfem::dimension::type::dim2>() {
  constexpr int total_edges = BOOST_PP_SEQ_SIZE(EDGES);
  constexpr std::array<
      std::tuple<specfem::dimension::type, specfem::connections::type,
                 specfem::interface::interface_tag,
                 specfem::element::boundary_tag>,
      total_edges>
      edges{ _MAKE_CONSTEXPR_ARRAY(EDGES) };
  return edges;
}

} // namespace specfem::interface
