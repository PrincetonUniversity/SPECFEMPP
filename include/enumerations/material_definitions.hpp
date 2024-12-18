#pragma once

#include "medium.hpp"
#include <boost/preprocessor.hpp>

namespace specfem {
namespace element {

#define DIMENSION_TAG_DIM2 (0, specfem::dimension::type::dim2)

#define MEDIUM_TAG_ELASTIC (0, specfem::element::medium_tag::elastic)
#define MEDIUM_TAG_ACOUSTIC (1, specfem::element::medium_tag::acoustic)

#define PROPERTY_TAG_ISOTROPIC (0, specfem::element::property_tag::isotropic)
#define PROPERTY_TAG_ANISOTROPIC                                               \
  (1, specfem::element::property_tag::anisotropic)

#define BOUNDARY_TAG_NONE (0, specfem::element::boundary_tag::none)
#define BOUNDARY_TAG_STACEY (1, specfem::element::boundary_tag::stacey)
#define BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE                                     \
  (2, specfem::element::boundary_tag::acoustic_free_surface)
#define BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET                                \
  (3, specfem::element::boundary_tag::composite_stacey_dirichlet)

#define GET_ID(s, data, elem) BOOST_PP_TUPLE_ELEM(0, elem)
#define GET_TAG(s, data, elem) BOOST_PP_TUPLE_ELEM(1, elem)

#define MEDIUM_TYPES                                                           \
  ((DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC))(                                  \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ACOUSTIC))

#define MAKE_ARRAY_ELEM(s, data, elem)                                         \
  std::make_tuple(GET_TAG(s, data, BOOST_PP_TUPLE_ELEM(0, elem)),              \
                  GET_TAG(s, data, BOOST_PP_TUPLE_ELEM(1, elem)))

#define MAKE_CONSTEXPR_ARRAY(seq)                                              \
  BOOST_PP_SEQ_ENUM(BOOST_PP_SEQ_TRANSFORM(MAKE_ARRAY_ELEM, _, seq))

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

// Define your sequence
#define MATERIAL_SYSTEMS                                                       \
  ((DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC, PROPERTY_TAG_ISOTROPIC))(          \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC, PROPERTY_TAG_ANISOTROPIC))(     \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ACOUSTIC, PROPERTY_TAG_ISOTROPIC))

// Helper macro to transform each element in the sequence into a
// std::pair
#define MAKE_ARRAY_ELEM(s, data, elem)                                         \
  std::make_tuple(GET_TAG(s, data, BOOST_PP_TUPLE_ELEM(0, elem)),              \
                  GET_TAG(s, data, BOOST_PP_TUPLE_ELEM(1, elem)),              \
                  GET_TAG(s, data, BOOST_PP_TUPLE_ELEM(2, elem)))

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

#define ELEMENT_TYPES                                                          \
  ((DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC, PROPERTY_TAG_ISOTROPIC,            \
    BOUNDARY_TAG_NONE))((DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC,               \
                         PROPERTY_TAG_ISOTROPIC, BOUNDARY_TAG_STACEY))(        \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ACOUSTIC, PROPERTY_TAG_ISOTROPIC,        \
       BOUNDARY_TAG_NONE))((DIMENSION_TAG_DIM2, MEDIUM_TAG_ACOUSTIC,           \
                            PROPERTY_TAG_ISOTROPIC,                            \
                            BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE))(              \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ACOUSTIC, PROPERTY_TAG_ISOTROPIC,        \
       BOUNDARY_TAG_STACEY))((DIMENSION_TAG_DIM2, MEDIUM_TAG_ACOUSTIC,         \
                              PROPERTY_TAG_ISOTROPIC,                          \
                              BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))(       \
      (DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC, PROPERTY_TAG_ANISOTROPIC,       \
       BOUNDARY_TAG_NONE))((DIMENSION_TAG_DIM2, MEDIUM_TAG_ELASTIC,            \
                            PROPERTY_TAG_ANISOTROPIC, BOUNDARY_TAG_STACEY))

#define MAKE_ARRAY_ELEM(s, data, elem)                                         \
  std::make_tuple(GET_TAG(s, data, BOOST_PP_TUPLE_ELEM(0, elem)),              \
                  GET_TAG(s, data, BOOST_PP_TUPLE_ELEM(1, elem)),              \
                  GET_TAG(s, data, BOOST_PP_TUPLE_ELEM(2, elem)),              \
                  GET_TAG(s, data, BOOST_PP_TUPLE_ELEM(3, elem)))

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

#define ELEM_IN_TUPLE(s, elem, tuple)                                          \
  BOOST_PP_IF(                                                                 \
      BOOST_PP_EQUAL(BOOST_PP_TUPLE_SIZE(tuple), 4),                           \
      BOOST_PP_IF(                                                             \
          BOOST_PP_EQUAL(GET_ID(s, _, BOOST_PP_TUPLE_ELEM(0, tuple)),          \
                         GET_ID(s, _, BOOST_PP_TUPLE_ELEM(0, elem))),          \
          BOOST_PP_IF(                                                         \
              BOOST_PP_EQUAL(GET_ID(s, _, BOOST_PP_TUPLE_ELEM(1, tuple)),      \
                             GET_ID(s, _, BOOST_PP_TUPLE_ELEM(1, elem))),      \
              BOOST_PP_IF(                                                     \
                  BOOST_PP_EQUAL(GET_ID(s, _, BOOST_PP_TUPLE_ELEM(2, tuple)),  \
                                 GET_ID(s, _, BOOST_PP_TUPLE_ELEM(2, elem))),  \
                  BOOST_PP_IF(BOOST_PP_EQUAL(                                  \
                                  GET_ID(s, _, BOOST_PP_TUPLE_ELEM(3, tuple)), \
                                  GET_ID(s, _, BOOST_PP_TUPLE_ELEM(3, elem))), \
                              1, 0),                                           \
                  0),                                                          \
              0),                                                              \
          0),                                                                  \
      0)

#define OP_OR(s, state, elem) BOOST_PP_OR(state, elem)

#define ELEM_IN_SEQUENCE(elem)                                                 \
  BOOST_PP_SEQ_FOLD_LEFT(                                                      \
      OP_OR, 0, BOOST_PP_SEQ_TRANSFORM(ELEM_IN_TUPLE, elem, ELEMENT_TYPES))

#define WHERE(...) (BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define CALL_FOR_ONE_SEQUENCE(s, MACRO, elem)                                  \
  BOOST_PP_IF(ELEM_IN_SEQUENCE((BOOST_PP_SEQ_ENUM(elem))), MACRO, EMPTY_MACRO) \
  (GET_TAG(s, _, BOOST_PP_TUPLE_ELEM(0, (BOOST_PP_SEQ_ENUM(elem)))),           \
   GET_TAG(s, _, BOOST_PP_TUPLE_ELEM(1, (BOOST_PP_SEQ_ENUM(elem)))),           \
   GET_TAG(s, _, BOOST_PP_TUPLE_ELEM(2, (BOOST_PP_SEQ_ENUM(elem)))),           \
   GET_TAG(s, _, BOOST_PP_TUPLE_ELEM(3, (BOOST_PP_SEQ_ENUM(elem)))))

#define EMPTY_MACRO(...)

#define CREATE_SEQ(r, elem) (elem)

#define CALL_MACRO_FOR_ALL_ELEMENT_TYPES(MACRO, seq)                           \
  BOOST_PP_SEQ_FOR_EACH(CALL_FOR_ONE_SEQUENCE, MACRO,                          \
                        BOOST_PP_SEQ_FOR_EACH_PRODUCT(CREATE_SEQ, seq))

} // namespace element
} // namespace specfem
