/**
 * @brief Call a macro for each element in the sequence.
 *  Built-in macros like BOOST_PP_SEQ_FOR_EACH_R cannot be used here because
 * they cannot correctly expand angle brackets with comma, like
 * property<elastic, isotropic>.
 * @param MACRO The macro to be called
 * @param data data to be passed to the macro
 * @param seq sequence to be expanded
 * @param n size of the sequence
 */
#define _EXPAND_SEQ(MACRO, data, seq)                                          \
  BOOST_PP_CAT(_EXPAND_SEQ_, BOOST_PP_SEQ_SIZE(seq))(MACRO, data, seq)

// clang-format off
#define _EXPAND_SEQ_1(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(0, seq))
#define _EXPAND_SEQ_2(MACRO, data , seq) _EXPAND_SEQ_1(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(1, seq))
#define _EXPAND_SEQ_3(MACRO, data , seq) _EXPAND_SEQ_2(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(2, seq))
#define _EXPAND_SEQ_4(MACRO, data , seq) _EXPAND_SEQ_3(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(3, seq))
#define _EXPAND_SEQ_5(MACRO, data , seq) _EXPAND_SEQ_4(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(4, seq))
#define _EXPAND_SEQ_6(MACRO, data , seq) _EXPAND_SEQ_5(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(5, seq))
#define _EXPAND_SEQ_7(MACRO, data , seq) _EXPAND_SEQ_6(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(6, seq))
#define _EXPAND_SEQ_8(MACRO, data , seq) _EXPAND_SEQ_7(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(7, seq))
#define _EXPAND_SEQ_9(MACRO, data , seq) _EXPAND_SEQ_8(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(8, seq))
#define _EXPAND_SEQ_10(MACRO, data , seq) _EXPAND_SEQ_9(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(9, seq))
#define _EXPAND_SEQ_11(MACRO, data , seq) _EXPAND_SEQ_10(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(10, seq))
#define _EXPAND_SEQ_12(MACRO, data , seq) _EXPAND_SEQ_11(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(11, seq))
#define _EXPAND_SEQ_13(MACRO, data , seq) _EXPAND_SEQ_12(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(12, seq))
#define _EXPAND_SEQ_14(MACRO, data , seq) _EXPAND_SEQ_13(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(13, seq))
#define _EXPAND_SEQ_15(MACRO, data , seq) _EXPAND_SEQ_14(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(14, seq))
#define _EXPAND_SEQ_16(MACRO, data , seq) _EXPAND_SEQ_15(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(15, seq))
#define _EXPAND_SEQ_17(MACRO, data , seq) _EXPAND_SEQ_16(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(16, seq))
#define _EXPAND_SEQ_18(MACRO, data , seq) _EXPAND_SEQ_17(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(17, seq))
#define _EXPAND_SEQ_19(MACRO, data , seq) _EXPAND_SEQ_18(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(18, seq))
#define _EXPAND_SEQ_20(MACRO, data , seq) _EXPAND_SEQ_19(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(19, seq))
#define _EXPAND_SEQ_21(MACRO, data , seq) _EXPAND_SEQ_20(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(20, seq))
#define _EXPAND_SEQ_22(MACRO, data , seq) _EXPAND_SEQ_21(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(21, seq))
#define _EXPAND_SEQ_23(MACRO, data , seq) _EXPAND_SEQ_22(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(22, seq))
#define _EXPAND_SEQ_24(MACRO, data , seq) _EXPAND_SEQ_23(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(23, seq))
#define _EXPAND_SEQ_25(MACRO, data , seq) _EXPAND_SEQ_24(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(24, seq))
#define _EXPAND_SEQ_26(MACRO, data , seq) _EXPAND_SEQ_25(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(25, seq))
#define _EXPAND_SEQ_27(MACRO, data , seq) _EXPAND_SEQ_26(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(26, seq))
#define _EXPAND_SEQ_28(MACRO, data , seq) _EXPAND_SEQ_27(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(27, seq))
#define _EXPAND_SEQ_29(MACRO, data , seq) _EXPAND_SEQ_28(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(28, seq))
#define _EXPAND_SEQ_30(MACRO, data , seq) _EXPAND_SEQ_29(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(29, seq))
#define _EXPAND_SEQ_31(MACRO, data , seq) _EXPAND_SEQ_30(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(30, seq))
#define _EXPAND_SEQ_32(MACRO, data , seq) _EXPAND_SEQ_31(MACRO, data , seq) MACRO(data, BOOST_PP_SEQ_ELEM(31, seq))
// clang-format on

/**
 * @brief Similar as _EXPAND_SEQ, but defined separately to enable nested macro
 * call from _EXPAND_SEQ.
 */
#define _EXPAND_TUPLE(MACRO, data, seq)                                        \
  BOOST_PP_CAT(_EXPAND_TUPLE_, BOOST_PP_TUPLE_SIZE(seq))(MACRO, data, seq)

// clang-format off
#define _EXPAND_TUPLE_1(MACRO, data , seq) MACRO(data, BOOST_PP_TUPLE_ELEM(0, seq))
#define _EXPAND_TUPLE_2(MACRO, data , seq) _EXPAND_TUPLE_1(MACRO, data , seq) MACRO(data, BOOST_PP_TUPLE_ELEM(1, seq))
#define _EXPAND_TUPLE_3(MACRO, data , seq) _EXPAND_TUPLE_2(MACRO, data , seq) MACRO(data, BOOST_PP_TUPLE_ELEM(2, seq))
#define _EXPAND_TUPLE_4(MACRO, data , seq) _EXPAND_TUPLE_3(MACRO, data , seq) MACRO(data, BOOST_PP_TUPLE_ELEM(3, seq))
#define _EXPAND_TUPLE_5(MACRO, data , seq) _EXPAND_TUPLE_4(MACRO, data , seq) MACRO(data, BOOST_PP_TUPLE_ELEM(4, seq))
#define _EXPAND_TUPLE_6(MACRO, data , seq) _EXPAND_TUPLE_5(MACRO, data , seq) MACRO(data, BOOST_PP_TUPLE_ELEM(5, seq))
#define _EXPAND_TUPLE_7(MACRO, data , seq) _EXPAND_TUPLE_6(MACRO, data , seq) MACRO(data, BOOST_PP_TUPLE_ELEM(6, seq))
#define _EXPAND_TUPLE_8(MACRO, data , seq) _EXPAND_TUPLE_7(MACRO, data , seq) MACRO(data, BOOST_PP_TUPLE_ELEM(7, seq))
#define _EXPAND_TUPLE_9(MACRO, data , seq) _EXPAND_TUPLE_8(MACRO, data , seq) MACRO(data, BOOST_PP_TUPLE_ELEM(8, seq))
#define _EXPAND_TUPLE_10(MACRO, data , seq) _EXPAND_TUPLE_9(MACRO, data , seq) MACRO(data, BOOST_PP_TUPLE_ELEM(9, seq))
#define _EXPAND_TUPLE_11(MACRO, data , seq) _EXPAND_TUPLE_10(MACRO, data , seq) MACRO(data, BOOST_PP_TUPLE_ELEM(10, seq))
#define _EXPAND_TUPLE_12(MACRO, data , seq) _EXPAND_TUPLE_11(MACRO, data , seq) MACRO(data, BOOST_PP_TUPLE_ELEM(11, seq))
#define _EXPAND_TUPLE_13(MACRO, data , seq) _EXPAND_TUPLE_12(MACRO, data , seq) MACRO(data, BOOST_PP_TUPLE_ELEM(12, seq))
#define _EXPAND_TUPLE_14(MACRO, data , seq) _EXPAND_TUPLE_13(MACRO, data , seq) MACRO(data, BOOST_PP_TUPLE_ELEM(13, seq))
#define _EXPAND_TUPLE_15(MACRO, data , seq) _EXPAND_TUPLE_14(MACRO, data , seq) MACRO(data, BOOST_PP_TUPLE_ELEM(14, seq))
#define _EXPAND_TUPLE_16(MACRO, data , seq) _EXPAND_TUPLE_15(MACRO, data , seq) MACRO(data, BOOST_PP_TUPLE_ELEM(15, seq))
// clang-format on
