#pragma once

#define GET_ID(elem) BOOST_PP_TUPLE_ELEM(0, elem)
#define GET_TAG(elem) BOOST_PP_TUPLE_ELEM(1, elem)
#define GET_NAME(elem) BOOST_PP_TUPLE_ELEM(2, elem)

#define _ADD_UNDERSCORE(s, data, elem) BOOST_PP_CAT(_, elem)

#define CREATE_VARIABLE_NAME(prefix, ...)                                      \
  BOOST_PP_SEQ_CAT((prefix)BOOST_PP_SEQ_TRANSFORM(                             \
      _ADD_UNDERSCORE, _, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)))

#define _EMPTY_MACRO(...)

#define _CREATE_SEQ(r, elem) (elem)

/**
 * @brief Macro to create a constexpr array from a sequence
 * Used by medium_types(), material_systems() and element_types()
 */
#define _MAKE_CONSTEXPR_ARRAY(seq, macro)                                      \
  BOOST_PP_SEQ_ENUM(BOOST_PP_SEQ_TRANSFORM(macro, _, seq))

// touch the following code at your own risk
#define _OP_OR(s, state, elem) BOOST_PP_OR(state, elem)
