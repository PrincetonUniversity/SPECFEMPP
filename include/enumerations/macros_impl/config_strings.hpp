#pragma once

#include <boost/preprocessor.hpp>

#define _SEQ_CAT_WITH_OR(seq)                                                  \
  BOOST_PP_CAT(_SEQ_CAT_WITH_OR_, BOOST_PP_SEQ_SIZE(seq)) seq

#define _SEQ_CAT_WITH_OR_1(x) x
#define _SEQ_CAT_WITH_OR_2(x) x || _SEQ_CAT_WITH_OR_1
#define _SEQ_CAT_WITH_OR_3(x) x || _SEQ_CAT_WITH_OR_2
#define _SEQ_CAT_WITH_OR_4(x) x || _SEQ_CAT_WITH_OR_3
#define _SEQ_CAT_WITH_OR_5(x) x || _SEQ_CAT_WITH_OR_4
#define _SEQ_CAT_WITH_OR_6(x) x || _SEQ_CAT_WITH_OR_5
#define _SEQ_CAT_WITH_OR_7(x) x || _SEQ_CAT_WITH_OR_6
#define _SEQ_CAT_WITH_OR_8(x) x || _SEQ_CAT_WITH_OR_7
#define _SEQ_CAT_WITH_OR_9(x) x || _SEQ_CAT_WITH_OR_8
#define _SEQ_CAT_WITH_OR_10(x) x || _SEQ_CAT_WITH_OR_9
#define _SEQ_CAT_WITH_OR_11(x) x || _SEQ_CAT_WITH_OR_10
#define _SEQ_CAT_WITH_OR_12(x) x || _SEQ_CAT_WITH_OR_11
#define _SEQ_CAT_WITH_OR_13(x) x || _SEQ_CAT_WITH_OR_12
#define _SEQ_CAT_WITH_OR_14(x) x || _SEQ_CAT_WITH_OR_13
#define _SEQ_CAT_WITH_OR_15(x) x || _SEQ_CAT_WITH_OR_14
#define _SEQ_CAT_WITH_OR_16(x) x || _SEQ_CAT_WITH_OR_15

#define _TEST_CONFIG_STRING(s, data, elem) str_lower == #elem

#define _DEFINE_CONFIG_STRING_FUNCTIONS(s, data, elem)                         \
  bool BOOST_PP_CAT(BOOST_PP_CAT(is_, BOOST_PP_TUPLE_ELEM(0, elem)),           \
                    _string)(const std::string &str) {                         \
    const auto str_lower = to_lower(str);                                      \
    return _SEQ_CAT_WITH_OR(BOOST_PP_SEQ_TRANSFORM(                            \
        _TEST_CONFIG_STRING, _, BOOST_PP_TUPLE_TO_SEQ(elem)));                 \
  }

#define _DECLARE_CONFIG_STRING_FUNCTIONS(s, data, elem)                        \
  bool BOOST_PP_CAT(BOOST_PP_CAT(is_, BOOST_PP_TUPLE_ELEM(0, elem)),           \
                    _string)(const std::string &str);
