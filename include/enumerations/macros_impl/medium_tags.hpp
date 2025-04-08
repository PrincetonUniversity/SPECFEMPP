#pragma once

#include "utils.hpp"

/**
 * @brief Call a macro for a specific medium tag if it is in the list of
 * available medium tags defined by macro MEDIUM_TAGS.
 */
#define _CALL_FOR_ONE_MEDIUM_TAG(s, MACRO, elem)                               \
  BOOST_PP_IF(_MEDIUM_IN_SEQUENCE((BOOST_PP_SEQ_ENUM(elem))), MACRO,           \
              _EMPTY_MACRO)                                                    \
  (BOOST_PP_TUPLE_ELEM(0, (BOOST_PP_SEQ_ENUM(elem))),                          \
   BOOST_PP_TUPLE_ELEM(1, (BOOST_PP_SEQ_ENUM(elem))))

// Touch the following code at your own risk
#define _MEDIUM_IN_TUPLE(s, elem, tuple)                                       \
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

#define _MEDIUM_IN_SEQUENCE(elem)                                              \
  BOOST_PP_SEQ_FOLD_LEFT(                                                      \
      _OP_OR, 0, BOOST_PP_SEQ_TRANSFORM(_MEDIUM_IN_TUPLE, elem, MEDIUM_TAGS))
