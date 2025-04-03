#pragma once

#include "utils.hpp"

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

/**
 * @brief Obtain Name of the variable to be declared from type declaration
 * tuple.
 * @param data dimension_tag, medium_tag, property_tag
 * @param elem type tuple, e.g. ((property<elastic)(isotropic>), value)
 * For input elem ((property<elastic)(isotropic>), value), the output is
 * value_dim2_elastic_isotropic
 */
#define _DECLARE_NAME_FOR_MATERIAL_SYSTEM(data, elem)                          \
  BOOST_PP_SEQ_CAT((BOOST_PP_TUPLE_ELEM(1, elem))(                             \
      _)(GET_NAME(BOOST_PP_TUPLE_ELEM(0, data)))(                              \
      _)(GET_NAME(BOOST_PP_TUPLE_ELEM(1, data)))(                              \
      _)(GET_NAME(BOOST_PP_TUPLE_ELEM(2, data))))

/**
 * @brief Declare a variable based on the type declaration tuple.
 * @param data dimension_tag, medium_tag, property_tag
 * @param elem type tuple, e.g. ((property<elastic)(isotropic>), value)
 * For input data (dim2, elastic, isotropic) and elem
 * ((property<elastic)(isotropic>), value), the output is auto
 * value_dim2_elastic_isotropic = [] (const auto &_dimension_tag_, const auto
 * &_medium_tag_, const auto &_property_tag_) { property<elastic, isotropic>
 * _var_; return _var_; }(dim2, elastic, isotropic);
 */
#define _TRANSFORM_DECLARE_FOR_MATERIAL_SYSTEM(s, data, elem)                  \
  (auto _DECLARE_NAME_FOR_MATERIAL_SYSTEM(data, elem) =                        \
       [](const auto &_dimension_tag_, const auto &_medium_tag_,               \
          const auto &_property_tag_) {                                        \
         _DECLARE_TYPE(elem) _var_;                                            \
         return _var_;                                                         \
       }(GET_TAG(BOOST_PP_TUPLE_ELEM(0, data)),                                \
         GET_TAG(BOOST_PP_TUPLE_ELEM(1, data)),                                \
         GET_TAG(BOOST_PP_TUPLE_ELEM(2, data)));)

/**
 * @brief Declare variables for all material systems in the sequence.
 * @param seq sequence of declaration tuples created by macro DECLARE
 */
#define _WRITE_DECLARE_FOR_MATERIAL_SYSTEM(DIMENSION_TAG, MEDIUM_TAG,          \
                                           PROPERTY_TAG, seq)                  \
  _EXPAND_DECLARE(                                                             \
      BOOST_PP_SEQ_TRANSFORM(_TRANSFORM_DECLARE_FOR_MATERIAL_SYSTEM,           \
                             (DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG), seq))

/**
 * @brief Convert the sequence of variables to be captured to lines of reference
 * declaration.
 * @param seq sequence of variable name e.g. (value)(elements)(h_elements).
 * The corresponding output will be
 * ```
 * auto &value = value_dim2_elastic_isotropic;
 * auto &elements = elements_dim2_elastic_isotropic;
 * auto &h_elements = h_elements_dim2_elastic_isotropic;
 * ```
 */
#define _WRITE_CAPTURE_FOR_MATERIAL_SYSTEM(DIMENSION_TAG, MEDIUM_TAG,          \
                                           PROPERTY_TAG, seq)                  \
  _SEQ_SPACE(BOOST_PP_SEQ_TRANSFORM(                                           \
      _GET_CAPTURE_DECLARATION,                                                \
      CREATE_VARIABLE_NAME(GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),      \
                           GET_NAME(PROPERTY_TAG)),                            \
      seq))

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
    _WRITE_CAPTURE_FOR_MATERIAL_SYSTEM(                                        \
        DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, _GET_CAPTURE_SEQ(seq, code))  \
    BOOST_PP_SEQ_ENUM(_REMOVE_CAPTURE_FROM_CODE(code))                         \
  }

/**
 * @brief Write both variable declaration and code block for all material
 * systems in the sequence.
 * @param seq first item of the sequence is a sequence of declaration tuples for
 * variable declaration. other items are code tokens to be executed for each
 * material system.
 */
#define _WRITE_WITH_DECLARE_FOR_MATERIAL_SYSTEM(DIMENSION_TAG, MEDIUM_TAG,     \
                                                PROPERTY_TAG, _, code)         \
  _WRITE_DECLARE_FOR_MATERIAL_SYSTEM(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG,  \
                                     BOOST_PP_SEQ_HEAD(code))                  \
  BOOST_PP_IF(BOOST_PP_SEQ_SIZE(code), _WRITE_BLOCK_FOR_MATERIAL_SYSTEM,       \
              _EMPTY_MACRO)                                                    \
  (DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, BOOST_PP_SEQ_HEAD(code),           \
   _REMOVE_DECLARE_FROM_CODE(code))

/**
 * @brief Check if the first element of the sequence is a list of variable
 * names. If it is, then call variable declaration macro.
 */
#define _CHECK_DECLARE_FOR_MATERIAL_SYSTEM(DIMENSION_TAG, MEDIUM_TAG,          \
                                           PROPERTY_TAG, code)                 \
  BOOST_PP_IF(BOOST_VMD_IS_SEQ(BOOST_PP_SEQ_HEAD(code)),                       \
              _WRITE_WITH_DECLARE_FOR_MATERIAL_SYSTEM,                         \
              _WRITE_BLOCK_FOR_MATERIAL_SYSTEM)(DIMENSION_TAG, MEDIUM_TAG,     \
                                                PROPERTY_TAG, (), code)

/**
 * Check if a give material system is in the list of available material systems.
 */
#define _FOR_ONE_MATERIAL_SYSTEM(s, code, elem)                                \
  BOOST_PP_IF(_MAT_SYS_IN_SEQUENCE((BOOST_PP_SEQ_ENUM(elem))),                 \
              _CHECK_DECLARE_FOR_MATERIAL_SYSTEM, _EMPTY_MACRO)(               \
      BOOST_PP_TUPLE_ELEM(0, (BOOST_PP_SEQ_ENUM(elem))),                       \
      BOOST_PP_TUPLE_ELEM(1, (BOOST_PP_SEQ_ENUM(elem))),                       \
      BOOST_PP_TUPLE_ELEM(2, (BOOST_PP_SEQ_ENUM(elem))), code)

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
