#pragma once

#include "utils.hpp"

/**
 * @brief Sequence transformation function for _MAKE_CONSTEXPR_ARRAY.
 */
#define _MAKE_ARRAY_ELEM_MAT_SYS(s, data, elem)                                \
  std::make_tuple(GET_TAG(BOOST_PP_TUPLE_ELEM(0, elem)),                       \
                  GET_TAG(BOOST_PP_TUPLE_ELEM(1, elem)),                       \
                  GET_TAG(BOOST_PP_TUPLE_ELEM(2, elem)))

/**
 * @brief Call a macro for a specific material system if it is in the list of
 * available material systems defined by macro MATERIAL_SYSTEMS.
 */
#define _CALL_FOR_ONE_MATERIAL_SYSTEM(s, MACRO, elem)                          \
  BOOST_PP_IF(_MAT_SYS_IN_SEQUENCE((BOOST_PP_SEQ_ENUM(elem))), MACRO,          \
              _EMPTY_MACRO)                                                    \
  (BOOST_PP_TUPLE_ELEM(0, (BOOST_PP_SEQ_ENUM(elem))),                          \
   BOOST_PP_TUPLE_ELEM(1, (BOOST_PP_SEQ_ENUM(elem))),                          \
   BOOST_PP_TUPLE_ELEM(2, (BOOST_PP_SEQ_ENUM(elem))))

// touch the following code at your own risk
#define _MAT_SYS_IN_TUPLE(s, elem, tuple)                                      \
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

#define _MAT_SYS_IN_SEQUENCE(elem)                                             \
  BOOST_PP_SEQ_FOLD_LEFT(                                                      \
      _OP_OR, 0,                                                               \
      BOOST_PP_SEQ_TRANSFORM(_MAT_SYS_IN_TUPLE, elem, MATERIAL_SYSTEMS))
