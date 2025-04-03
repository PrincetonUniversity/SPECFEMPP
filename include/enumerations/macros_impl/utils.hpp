#pragma once

#include "enumerations/medium.hpp"
#include "expand_declare.hpp"
#include "seq_space.hpp"
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

#define _EMPTY_SEQ(code) ()

#define _REFLECT(code) code

/**
 * @brief Right and left hand side of type declaration sequence tuple.
 * For declaration property<elastic, isotropic> value, the tuple is
 * declared as (((property<elastic)(isotropic>), value))
 */
#define _WRITE_PAIRWISE_NAME(elem) elem))
#define _WRITE_PAIRWISE_TYPE(elem) ((elem,

/**
 * @brief Macros to group declaration sequence pairwise.
 * A declaration sequence wrapped in DECLARE(...) has form
 * (type1)(name1)(type2)(name2) ... This macro groups the sequence into sequence
 * of (type, name) tuples, e.g. ((type1, name1))((type2, name2)).
 */
#define _PAIRWISE_GROUP_HELPER(s, seq, i, elem)                                \
  BOOST_PP_IF(BOOST_PP_MOD(i, 2), _WRITE_PAIRWISE_NAME,                        \
              _WRITE_PAIRWISE_TYPE)(elem)

#define _PAIRWISE_GROUP(seq)                                                   \
  BOOST_PP_SEQ_FOR_EACH_I(_PAIRWISE_GROUP_HELPER, seq, seq)

/**
 * @brief Obtain type of the variable to be declared from type declaration
 * tuple.
 * @param elem type tuple, e.g. ((property<elastic)(isotropic>), value)
 * For input elem ((property<elastic)(isotropic>), value), the output is
 * property<elastic, isotropic>
 */
#define _DECLARE_TYPE(elem) BOOST_PP_SEQ_ENUM(BOOST_PP_TUPLE_ELEM(0, elem))

/**
 * @brief Create a sequence of list of variables to be captured from DECLARE()
 */
#define _GET_CAPTURE_FROM_DECLARE(s, data, elem) BOOST_PP_TUPLE_ELEM(1, elem)

#define _GET_CAPTURE_SEQ_FROM_DECLARE(seq)                                     \
  BOOST_PP_SEQ_TRANSFORM(_GET_CAPTURE_FROM_DECLARE, _, seq)

/**
 * @brief Create a sequence of list of variables to be captured from both
 * DECLARE(...) and CAPTURE(...)
 * @param seq declaration tuple sequence from DECLARE(...)
 * @param code code block possibly starting with CAPTURE(...)
 */
#define _GET_CAPTURE_SEQ(seq, code)                                            \
  BOOST_PP_IF(BOOST_PP_SEQ_SIZE(seq), _GET_CAPTURE_SEQ_FROM_DECLARE,           \
              _EMPTY_MACRO)(seq)                                               \
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
 * @brief Create a reference to the variable to be captured inside the code
 * block to be written. e.g. auto &value = value_dim2_elastic_isotropic;
 */
#define _GET_CAPTURE_DECLARATION(s, postfix, prefix)                           \
  BOOST_PP_IF(BOOST_PP_IS_EMPTY(prefix), BOOST_PP_EMPTY(),                     \
              auto &prefix = prefix##_##postfix;)

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
