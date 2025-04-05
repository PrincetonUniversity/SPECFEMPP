#pragma once

#include "enumerations/interface.hpp"
#include "expand_seq.hpp"
#include <boost/preprocessor.hpp>
#include <boost/vmd/is_list.hpp>
#include <boost/vmd/is_seq.hpp>
#include <boost/vmd/is_tuple.hpp>

#define GET_ID(elem) BOOST_PP_TUPLE_ELEM(0, elem)
#define GET_TAG(elem) BOOST_PP_TUPLE_ELEM(1, elem)
#define GET_NAME(elem) BOOST_PP_TUPLE_ELEM(2, elem)

#define CREATE_VARIABLE_NAME(prefix, ...)                                      \
  BOOST_PP_SEQ_CAT((prefix)BOOST_PP_SEQ_TRANSFORM(                             \
      _ADD_UNDERSCORE, _, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)))

#define _ADD_UNDERSCORE(s, data, elem) BOOST_PP_CAT(_, elem)

#define _EMPTY_MACRO(...)

#define _CREATE_SEQ(r, elem) (elem)

#define _EXPAND_VARIADIC(s, data, elem)                                        \
  BOOST_PP_EXPAND(BOOST_PP_VARIADIC_TO_SEQ elem)

#define _REFLECT(elem) elem

#define _REFLECT_1(_, elem) elem

#define _EMPTY_SEQ(...) ()

#define _TRANSFORM_INSTANTIATE(s, data, elem) (elem, )

/**
 * @brief Replace tags _DIMENSION_TAG_, _MEDIUM_TAG_, _PROPERTY_TAG_,
 * _BOUNDARY_TAG_ with current element type.
 */
#define _REPLACE_TAGS(s, data, elem)                                           \
  BOOST_PP_IF(BOOST_VMD_IS_LIST(elem),                                         \
              GET_TAG(BOOST_PP_TUPLE_ELEM(BOOST_PP_LIST_AT(elem, 0), data)),   \
              elem)
/**
 * @brief Replace tuple inside type declaration sequence with angle brackets.
 */
#define _EXPAND_TEMPLATE(data, elem)                                           \
<BOOST_PP_SEQ_ENUM(BOOST_PP_SEQ_TRANSFORM(_REPLACE_TAGS, data, BOOST_PP_TUPLE_TO_SEQ(elem)))>

/**
 * @brief Obtain type of the variable to be declared from type declaration
 * tuple.
 * @param data dimension_tag, medium_tag, property_tag
 * @param elem type tuple, e.g. (value, property, (_MEDIUM_TAG_,
 * _PROPERTY_TAG_)) For input elem (value, property, (_MEDIUM_TAG_,
 * _PROPERTY_TAG_)), the output is property<elastic, isotropic>
 */
#define _DECLARE_TYPE(data, elem)                                              \
  BOOST_PP_IF(BOOST_VMD_IS_TUPLE(elem), _EXPAND_TEMPLATE, _REFLECT_1)(data,    \
                                                                      elem)

/**
 * @brief Create a sequence of list of variables to be captured from DECLARE()
 */
#define _GET_CAPTURE_FROM_DECLARE(s, data, elem) BOOST_PP_TUPLE_ELEM(1, elem)

/**
 * @brief Create a sequence of list of variables to be captured from both
 * DECLARE(...) and CAPTURE(...)
 * @param seq declaration tuple sequence from DECLARE(...)
 * @param code code block possibly starting with CAPTURE(...)
 */
#define _GET_CAPTURE_SEQ(seq, code)                                            \
  BOOST_PP_IF(BOOST_PP_SEQ_SIZE(seq), BOOST_PP_SEQ_TRANSFORM,                  \
              _EMPTY_MACRO)(_GET_CAPTURE_FROM_DECLARE, _, seq)                 \
      BOOST_PP_IF(BOOST_VMD_IS_TUPLE(BOOST_PP_SEQ_HEAD(code)),                 \
                  BOOST_PP_TUPLE_TO_SEQ,                                       \
                  _EMPTY_MACRO)(BOOST_PP_SEQ_HEAD(code))

/**
 * @brief Remove possible CAPTURE(...) at the begining of the code.
 */
#define _REMOVE_CAPTURE_FROM_CODE(code)                                        \
  BOOST_PP_IF(BOOST_VMD_IS_TUPLE(BOOST_PP_SEQ_HEAD(code)), BOOST_PP_SEQ_TAIL,  \
              _REFLECT)(code)

/**
 * @brief Remove the declaration tuple DECLARE(...) from code.
 */
#define _REMOVE_DECLARE_FROM_CODE(code)                                        \
  BOOST_PP_IF(BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(code)), BOOST_PP_SEQ_TAIL,        \
              _EMPTY_SEQ)(code)

/**
 * @brief Macro to create a constexpr array from a sequence
 * Used by medium_types(), material_systems() and element_types()
 */
#define _MAKE_CONSTEXPR_ARRAY(seq, macro)                                      \
  BOOST_PP_SEQ_ENUM(BOOST_PP_SEQ_TRANSFORM(macro, _, seq))

// touch the following code at your own risk
#define _OP_OR(s, state, elem) BOOST_PP_OR(state, elem)
