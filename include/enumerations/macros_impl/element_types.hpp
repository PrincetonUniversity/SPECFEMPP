#pragma once

#include "utils.hpp"

/**
 * @brief Sequence transformation function for _MAKE_CONSTEXPR_ARRAY.
 */
#define _MAKE_ARRAY_ELEM_ELEM(s, data, elem)                                   \
  std::make_tuple(GET_TAG(BOOST_PP_TUPLE_ELEM(0, elem)),                       \
                  GET_TAG(BOOST_PP_TUPLE_ELEM(1, elem)),                       \
                  GET_TAG(BOOST_PP_TUPLE_ELEM(2, elem)),                       \
                  GET_TAG(BOOST_PP_TUPLE_ELEM(3, elem)))

/**
 * @brief Call a macro for a specific element type if it is in the list of
 * available element types defined by macro ELEMENT_TYPES.
 */
#define _CALL_FOR_ONE_ELEMENT_TYPE(s, MACRO, elem)                             \
  BOOST_PP_IF(_ELEM_IN_SEQUENCE((BOOST_PP_SEQ_ENUM(elem))), MACRO,             \
              _EMPTY_MACRO)                                                    \
  (BOOST_PP_TUPLE_ELEM(0, (BOOST_PP_SEQ_ENUM(elem))),                          \
   BOOST_PP_TUPLE_ELEM(1, (BOOST_PP_SEQ_ENUM(elem))),                          \
   BOOST_PP_TUPLE_ELEM(2, (BOOST_PP_SEQ_ENUM(elem))),                          \
   BOOST_PP_TUPLE_ELEM(3, (BOOST_PP_SEQ_ENUM(elem))))

// touch the following code at your own risk
#define _ELEM_IN_TUPLE(s, elem, tuple)                                         \
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

#define _ELEM_IN_SEQUENCE(elem)                                                \
  BOOST_PP_SEQ_FOLD_LEFT(                                                      \
      _OP_OR, 0, BOOST_PP_SEQ_TRANSFORM(_ELEM_IN_TUPLE, elem, ELEMENT_TYPES))
