#pragma once

#include "utils.hpp"

/**
 * @brief Obtain Name of the variable to be declared from type declaration
 * tuple.
 * @param data dimension_tag, medium_tag, property_tag
 * @param elem type tuple, e.g. ((property) ((_MEDIUM_TAG_, _PROPERTY_TAG_)),
 * value), which expands to value_dim2_elastic_isotropic
 */
#define _VAR_NAME_FOR_MATERIAL_SYSTEM(data, elem)                              \
  BOOST_PP_SEQ_CAT((elem)(_)(GET_NAME(BOOST_PP_TUPLE_ELEM(0, data)))(          \
      _)(GET_NAME(BOOST_PP_TUPLE_ELEM(1, data)))(                              \
      _)(GET_NAME(BOOST_PP_TUPLE_ELEM(2, data))))

/**
 * @brief Declare a variable based on the type declaration tuple.
 * @param data dimension_tag, medium_tag, property_tag
 * @param elem type tuple, e.g. ((property) ((_MEDIUM_TAG_, _PROPERTY_TAG_)),
 * value), For input data (dim2, elastic, isotropic), this expands to
 * property<elastic, isotropic> value_dim2_elastic_isotropic;
 */
#define _WRITE_DECLARE_FOR_MATERIAL_SYSTEM(data, elem)                         \
  BOOST_PP_IF(BOOST_VMD_IS_SEQ(BOOST_PP_TUPLE_ELEM(0, elem)), _EXPAND_SEQ2,    \
              _REFLECT_2)                                                      \
  (_DECLARE_TYPE, data, BOOST_PP_TUPLE_ELEM(0, elem))                          \
      _VAR_NAME_FOR_MATERIAL_SYSTEM(data, BOOST_PP_TUPLE_ELEM(1, elem));

/**
 * @brief Create a reference to the variable to be captured inside the code
 * block. e.g. auto &value = value_dim2_elastic_isotropic;
 */
#define _WRITE_CAPTURE_FOR_MATERIAL_SYSTEM(data, elem)                         \
  BOOST_PP_IF(BOOST_PP_IS_EMPTY(elem), BOOST_PP_EMPTY(),                       \
              auto &BOOST_PP_SEQ_CAT((_)(elem)(_)) =                           \
                  _VAR_NAME_FOR_MATERIAL_SYSTEM(data, elem);)

/**
 * @brief Create a code block and write the code for all material systems in the
 * sequence.
 * @param seq sequence of variables to be captured by reference inside the
 * block.
 * @param code code block to be executed for each material system.
 */
#define _WRITE_BLOCK_FOR_MATERIAL_SYSTEM(DIMENSION_TAG, MEDIUM_TAG,            \
                                         PROPERTY_TAG, seq, code)              \
  {                                                                            \
    constexpr auto _dimension_tag_ = GET_TAG(DIMENSION_TAG);                   \
    constexpr auto _medium_tag_ = GET_TAG(MEDIUM_TAG);                         \
    constexpr auto _property_tag_ = GET_TAG(PROPERTY_TAG);                     \
    _EXPAND_SEQ(_WRITE_CAPTURE_FOR_MATERIAL_SYSTEM,                            \
                (DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG),                     \
                _GET_CAPTURE_SEQ(seq, code))                                   \
    BOOST_PP_SEQ_ENUM(_REMOVE_CAPTURE_FROM_CODE(code))                         \
  }

/**
 * @brief Write both variable declaration and code block for all material
 * systems in the sequence.
 * @param seq first item of the sequence is a sequence of declaration tuples for
 * variable declaration. other items are code tokens to be executed for each
 * material system.
 */
#define _WRITE_DECLARE_AND_BLOCK_FOR_MATERIAL_SYSTEM(                          \
    DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, _, code)                          \
  _EXPAND_SEQ(_WRITE_DECLARE_FOR_MATERIAL_SYSTEM,                              \
              (DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG),                       \
              BOOST_PP_SEQ_HEAD(code))                                         \
  BOOST_PP_IF(BOOST_PP_LESS(1, BOOST_PP_SEQ_SIZE(code)),                       \
              _WRITE_BLOCK_FOR_MATERIAL_SYSTEM, _EMPTY_MACRO)                  \
  (DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, BOOST_PP_SEQ_HEAD(code),           \
   _REMOVE_DECLARE_FROM_CODE(code))

/**
 * @brief Check if the first element of the sequence is a list of variable
 * names. If it is, then write both variable declaration and code block. If it
 * is not, then write only the code block.
 */
#define _CHECK_DECLARE_FOR_MATERIAL_SYSTEM(DIMENSION_TAG, MEDIUM_TAG,          \
                                           PROPERTY_TAG, code)                 \
  BOOST_PP_IF(BOOST_VMD_IS_SEQ(BOOST_PP_SEQ_HEAD(code)),                       \
              _WRITE_DECLARE_AND_BLOCK_FOR_MATERIAL_SYSTEM,                    \
              _WRITE_BLOCK_FOR_MATERIAL_SYSTEM)(DIMENSION_TAG, MEDIUM_TAG,     \
                                                PROPERTY_TAG, (), code)

/**
 * Check if a give material system is in the list of available material systems,
 * write declaration and code block for the material system if it is in the
 * list.
 */
#define _FOR_ONE_MATERIAL_SYSTEM(s, code, elem)                                \
  BOOST_PP_IF(_MAT_SYS_IN_SEQUENCE((BOOST_PP_SEQ_ENUM(elem))),                 \
              _CHECK_DECLARE_FOR_MATERIAL_SYSTEM, _EMPTY_MACRO)(               \
      BOOST_PP_TUPLE_ELEM(0, (BOOST_PP_SEQ_ENUM(elem))),                       \
      BOOST_PP_TUPLE_ELEM(1, (BOOST_PP_SEQ_ENUM(elem))),                       \
      BOOST_PP_TUPLE_ELEM(2, (BOOST_PP_SEQ_ENUM(elem))), code)

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
