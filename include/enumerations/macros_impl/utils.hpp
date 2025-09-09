#pragma once

#include "enumerations/interface.hpp"
#include "expand_seq.hpp"
#include <boost/preprocessor.hpp>
#include <boost/vmd/vmd.hpp>

#define _GET_ID(elem) BOOST_PP_TUPLE_ELEM(0, elem)

#define _GET_TAG(elem) BOOST_PP_TUPLE_ELEM(1, elem)

#define _GET_NAME(elem) BOOST_PP_TUPLE_ELEM(2, elem)

#define _GET_ENUM_ID(elem) BOOST_PP_TUPLE_ELEM(3, elem)

#define _EMPTY_MACRO(...)

#define _EMPTY_SEQ(...) ()

#define _CREATE_SEQ(r, elem) (elem)

#define _REFLECT(elem) elem

#define _REFLECT_1(data, elem) elem

#define _REFLECT_2(s, data, elem) elem

#define _TRANSFORM_TAGS(s, data, elem) BOOST_PP_CAT(data, elem)

#define _TRANSFORM_INSTANTIATE(s, data, elem) (elem, )

#define _OP_OR(s, state, elem) BOOST_PP_OR(state, elem)

#define _SEQ_FOR_TAGS_2 MEDIUM_TAGS

#define _SEQ_FOR_TAGS_3 MATERIAL_SYSTEMS

#define _SEQ_FOR_TAGS_4 ELEMENT_TYPES EDGES

/**
 * @brief Declare a variable or instantiante a template based on the type
 * declaration tuple.
 * @param data possibly dimension_tag, medium_tag, property_tag, boundary_tag
 * @param elem Type tuple and optionally variable name. If its first element is
 * a tuple, e.g. ((property, (_MEDIUM_TAG_, _PROPERTY_TAG_)), value), it expands
 * to property<elastic, isotropic>. If its first element is not a tuple, e.g.
 * IndexViewType, value, it expands to the same IndexViewType. If the second
 * element exists, the expression declares a variable, e.g. property<elastic,
 * isotropic> value_dim2_elastic_isotropic; Otherwise, it instantiates a
 * template, e.g. template property<elastic, isotropic>;
 */
#define _WRITE_DECLARE(data, elem)                                             \
  BOOST_PP_IF(BOOST_VMD_IS_TUPLE(BOOST_PP_TUPLE_ELEM(0, elem)), _EXPAND_TUPLE, \
              _REFLECT_2)                                                      \
  (_DECLARE_TYPE, data, BOOST_PP_TUPLE_ELEM(0, elem)) BOOST_PP_IF(             \
      BOOST_PP_IS_EMPTY(BOOST_PP_TUPLE_ELEM(1, elem)), BOOST_PP_EMPTY(),       \
      _VAR_NAME_FROM_TAGS(data, BOOST_PP_TUPLE_ELEM(1, elem));)

/**
 * @brief Create a reference to the variable to be captured inside the code
 * block. e.g. auto &value = value_dim2_elastic_isotropic;
 * If elem is a tuple, the first element is used as the variable name and the
 * second element is used as the variable to be captured, e.g. for (value,
 * rhs.value), the output is auto &value = rhs.value_dim2_elastic_isotropic;
 */
#define _REFERENCE_LEFT(elem)                                                  \
  BOOST_PP_IF(BOOST_VMD_IS_TUPLE(elem), BOOST_PP_TUPLE_ELEM(0, elem), elem)

#define _REFERENCE_RIGHT(elem)                                                 \
  BOOST_PP_IF(BOOST_VMD_IS_TUPLE(elem), BOOST_PP_TUPLE_ELEM(1, elem), elem)

#define _WRITE_CAPTURE(data, elem)                                             \
  BOOST_PP_IF(                                                                 \
      BOOST_PP_IS_EMPTY(elem), BOOST_PP_EMPTY(),                               \
      [[maybe_unused]] auto &BOOST_PP_SEQ_CAT((_)(_REFERENCE_LEFT(elem))(_)) = \
          _VAR_NAME_FROM_TAGS(data, _REFERENCE_RIGHT(elem));)

/**
 * @brief Obtain Name of the variable to be declared from type declaration
 * tuple.
 * @param data possibly dimension_tag, medium_tag, property_tag, boundary_tag
 * @param elem type tuple, e.g. (value, property, (_MEDIUM_TAG_,
 * _PROPERTY_TAG_)) For input elem value and input data (_DIMENSION_TAG_,
 * _MEDIUM_TAG_, _PROPERTY_TAG_)), the output is value_dim2_elastic_isotropic
 */

#define _TRANSFORM_TAG_DATA(s, data, elem) BOOST_PP_CAT(_, _GET_NAME(elem))

#define _VAR_NAME_FROM_TAGS(data, elem)                                        \
  BOOST_PP_SEQ_CAT((elem)BOOST_PP_SEQ_TRANSFORM(_TRANSFORM_TAG_DATA, _,        \
                                                BOOST_PP_TUPLE_TO_SEQ(data)))

/**
 * @brief Replace tags _DIMENSION_TAG_, _MEDIUM_TAG_, _PROPERTY_TAG_,
 * _BOUNDARY_TAG_ with current element type.
 */
#define _REPLACE_TAGS(s, data, elem)                                           \
  BOOST_PP_IF(BOOST_VMD_IS_LIST(elem),                                         \
              _GET_TAG(BOOST_PP_TUPLE_ELEM(BOOST_PP_LIST_AT(elem, 0), data)),  \
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
 * @brief Create reference to tags for dimension, medium, property and boundary
 * in code block.
 */
#define _WRITE_TAG(tag, data, i)                                               \
  [[maybe_unused]] constexpr auto tag = _GET_TAG(BOOST_PP_TUPLE_ELEM(i, data));

// clang-format off
#define _WRITE_TAGS_1(data) _WRITE_TAG(_dimension_tag_, data, 0)
#define _WRITE_TAGS_2(data) _WRITE_TAGS_1(data) _WRITE_TAG(_medium_tag_, data, 1) _WRITE_TAG(_connection_tag_, data, 1)
#define _WRITE_TAGS_3(data) _WRITE_TAGS_2(data) _WRITE_TAG(_property_tag_, data, 2) _WRITE_TAG(_interface_tag_, data, 2)
#define _WRITE_TAGS_4(data) _WRITE_TAGS_3(data) _WRITE_TAG(_boundary_tag_, data, 3)
// clang-format on

/**
 * @brief Create a code block and write the code for all material systems in the
 * sequence.
 * @param seq sequence of variables to be captured by reference inside the
 * block.
 * @param code code block to be executed for each material system.
 */
#define _WRITE_BLOCK(data, seq, code)                                          \
  { BOOST_PP_CAT(_WRITE_TAGS_, BOOST_PP_TUPLE_SIZE(data))(data)                \
        _EXPAND_SEQ(_WRITE_CAPTURE, data, _GET_CAPTURE_SEQ(seq, code))         \
            BOOST_PP_SEQ_ENUM(_REMOVE_CAPTURE_FROM_CODE(code)) }

/**
 * @brief Write both variable declaration and code block for all material
 * systems in the sequence.
 * @param seq first item of the sequence is a sequence of declaration tuples for
 * variable declaration. other items are code tokens to be executed for each
 * material system.
 */
#define _WRITE_DECLARE_AND_BLOCK(data, _, code)                                \
  _EXPAND_SEQ(_WRITE_DECLARE, data, BOOST_PP_SEQ_HEAD(code))                   \
  BOOST_PP_IF(BOOST_PP_LESS(1, BOOST_PP_SEQ_SIZE(code)), _WRITE_BLOCK,         \
              _EMPTY_MACRO)                                                    \
  (data, BOOST_PP_SEQ_HEAD(code), _REMOVE_DECLARE_FROM_CODE(code))

/**
 * @brief Check if the first element of the sequence is a list of variable
 * names. If it is, then write both variable declaration and code block. If it
 * is not, then write only the code block.
 */
#define _CHECK_DECLARE(data, code)                                             \
  BOOST_PP_IF(BOOST_VMD_IS_SEQ(BOOST_PP_SEQ_HEAD(code)),                       \
              _WRITE_DECLARE_AND_BLOCK, _WRITE_BLOCK)(data, (), code)

#define _CHECK_ENUM(enum1, enum2)                                              \
  BOOST_PP_IF(                                                                 \
      BOOST_PP_EQUAL(_GET_ENUM_ID(enum1), _GET_ENUM_ID(enum2)),                \
      BOOST_PP_IF(BOOST_PP_EQUAL(_GET_ID(enum1), _GET_ID(enum2)), 1, 0), 0)

/**
 * @brief Compare each item in the sequence for a sequence pair of length 2, 3
 * and 4.
 */
#define _IN_TUPLE_2(s, elem, tuple)                                            \
  BOOST_PP_IF(                                                                 \
      BOOST_PP_EQUAL(BOOST_PP_TUPLE_SIZE(tuple), 2),                           \
      BOOST_PP_IF(_CHECK_ENUM(BOOST_PP_TUPLE_ELEM(0, tuple),                   \
                              BOOST_PP_TUPLE_ELEM(0, elem)),                   \
                  BOOST_PP_IF(_CHECK_ENUM(BOOST_PP_TUPLE_ELEM(1, tuple),       \
                                          BOOST_PP_TUPLE_ELEM(1, elem)),       \
                              1, 0),                                           \
                  0),                                                          \
      0)

#define _IN_TUPLE_3(s, elem, tuple)                                            \
  BOOST_PP_IF(                                                                 \
      BOOST_PP_EQUAL(BOOST_PP_TUPLE_SIZE(tuple), 3),                           \
      BOOST_PP_IF(                                                             \
          _CHECK_ENUM(BOOST_PP_TUPLE_ELEM(0, tuple),                           \
                      BOOST_PP_TUPLE_ELEM(0, elem)),                           \
          BOOST_PP_IF(_CHECK_ENUM(BOOST_PP_TUPLE_ELEM(1, tuple),               \
                                  BOOST_PP_TUPLE_ELEM(1, elem)),               \
                      BOOST_PP_IF(_CHECK_ENUM(BOOST_PP_TUPLE_ELEM(2, tuple),   \
                                              BOOST_PP_TUPLE_ELEM(2, elem)),   \
                                  1, 0),                                       \
                      0),                                                      \
          0),                                                                  \
      0)

#define _IN_TUPLE_4(s, elem, tuple)                                            \
  BOOST_PP_IF(                                                                 \
      BOOST_PP_EQUAL(BOOST_PP_TUPLE_SIZE(tuple), 4),                           \
      BOOST_PP_IF(                                                             \
          _CHECK_ENUM(BOOST_PP_TUPLE_ELEM(0, tuple),                           \
                      BOOST_PP_TUPLE_ELEM(0, elem)),                           \
          BOOST_PP_IF(                                                         \
              _CHECK_ENUM(BOOST_PP_TUPLE_ELEM(1, tuple),                       \
                          BOOST_PP_TUPLE_ELEM(1, elem)),                       \
              BOOST_PP_IF(                                                     \
                  _CHECK_ENUM(BOOST_PP_TUPLE_ELEM(2, tuple),                   \
                              BOOST_PP_TUPLE_ELEM(2, elem)),                   \
                  BOOST_PP_IF(_CHECK_ENUM(BOOST_PP_TUPLE_ELEM(3, tuple),       \
                                          BOOST_PP_TUPLE_ELEM(3, elem)),       \
                              1, 0),                                           \
                  0),                                                          \
              0),                                                              \
          0),                                                                  \
      0)

/**
 * @brief Check if a given tag sequence is in the list of available tag
 * sequences.
 */
#define _IN_SEQUENCE(n, elem)                                                  \
  BOOST_PP_SEQ_FOLD_LEFT(                                                      \
      _OP_OR, 0,                                                               \
      BOOST_PP_SEQ_TRANSFORM(BOOST_PP_CAT(_IN_TUPLE_, n), elem,                \
                             BOOST_PP_CAT(_SEQ_FOR_TAGS_, n)))

/**
 * Check if a given tag sequence is in the list of available tag sequences,
 * write declaration and code block for the sequence if it is in the list.
 */
#define _FOR_ONE_TAG_SEQ(s, code, elem)                                        \
  BOOST_PP_IF(                                                                 \
      _IN_SEQUENCE(BOOST_PP_SEQ_SIZE(elem), BOOST_PP_SEQ_TO_TUPLE(elem)),      \
      _CHECK_DECLARE, _EMPTY_MACRO)                                            \
  (BOOST_PP_SEQ_TO_TUPLE(elem), code)
