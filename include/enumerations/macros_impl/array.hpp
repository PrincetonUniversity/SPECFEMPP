/**
 * @brief Macro to create a constexpr array from a sequence
 * Used by medium_types(), material_systems() and element_types()
 */
#define _MAKE_CONSTEXPR_ARRAY(seq, macro)                                      \
  BOOST_PP_SEQ_ENUM(BOOST_PP_SEQ_TRANSFORM(macro, _, seq))

/**
 * @brief Sequence transformation function for _MAKE_CONSTEXPR_ARRAY.
 */
#define _MAKE_ARRAY_ELEM_MEDIUM(s, data, elem)                                 \
  std::make_tuple(GET_TAG(BOOST_PP_TUPLE_ELEM(0, elem)),                       \
                  GET_TAG(BOOST_PP_TUPLE_ELEM(1, elem)))

/**
 * @brief Sequence transformation function for _MAKE_CONSTEXPR_ARRAY.
 */
#define _MAKE_ARRAY_ELEM_MAT_SYS(s, data, elem)                                \
  std::make_tuple(GET_TAG(BOOST_PP_TUPLE_ELEM(0, elem)),                       \
                  GET_TAG(BOOST_PP_TUPLE_ELEM(1, elem)),                       \
                  GET_TAG(BOOST_PP_TUPLE_ELEM(2, elem)))

/**
 * @brief Sequence transformation function for _MAKE_CONSTEXPR_ARRAY.
 */
#define _MAKE_ARRAY_ELEM_ELEM(s, data, elem)                                   \
  std::make_tuple(GET_TAG(BOOST_PP_TUPLE_ELEM(0, elem)),                       \
                  GET_TAG(BOOST_PP_TUPLE_ELEM(1, elem)),                       \
                  GET_TAG(BOOST_PP_TUPLE_ELEM(2, elem)),                       \
                  GET_TAG(BOOST_PP_TUPLE_ELEM(3, elem)))
