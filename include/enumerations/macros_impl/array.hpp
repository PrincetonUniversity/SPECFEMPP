/**
 * @brief Macro to create a constexpr array from a sequence
 * Used by medium_types(), material_systems() and element_types()
 */
#define _MAKE_CONSTEXPR_ARRAY(seq)                                             \
  BOOST_PP_SEQ_ENUM(BOOST_PP_SEQ_TRANSFORM(_MAKE_ARRAY, _, seq))

/**
 * @brief Sequence transformation macros for _MAKE_CONSTEXPR_ARRAY.
 */
#define _MAKE_ARRAY(s, data, elem)                                             \
  BOOST_PP_CAT(_MAKE_ARRAY_, BOOST_PP_TUPLE_SIZE(elem))(elem)

#define _MAKE_ARRAY_2(elem)                                                    \
  std::make_tuple(_GET_TAG(BOOST_PP_TUPLE_ELEM(0, elem)),                      \
                  _GET_TAG(BOOST_PP_TUPLE_ELEM(1, elem)))

#define _MAKE_ARRAY_3(elem)                                                    \
  std::make_tuple(_GET_TAG(BOOST_PP_TUPLE_ELEM(0, elem)),                      \
                  _GET_TAG(BOOST_PP_TUPLE_ELEM(1, elem)),                      \
                  _GET_TAG(BOOST_PP_TUPLE_ELEM(2, elem)))

#define _MAKE_ARRAY_4(elem)                                                    \
  std::make_tuple(_GET_TAG(BOOST_PP_TUPLE_ELEM(0, elem)),                      \
                  _GET_TAG(BOOST_PP_TUPLE_ELEM(1, elem)),                      \
                  _GET_TAG(BOOST_PP_TUPLE_ELEM(2, elem)),                      \
                  _GET_TAG(BOOST_PP_TUPLE_ELEM(3, elem)))
