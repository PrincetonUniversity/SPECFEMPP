#pragma once

// tag_macros.hpp — type-form shorthand macros for tag-set types
//
//   DIMENSION_T(dim2)
//     =>
//     specfem::tag_dispatch::dimension_set<specfem::element::dimension_tag::dim2>
//
//   MEDIUM_T(acoustic, elastic_psv)
//     =>
//     specfem::tag_dispatch::medium_set<specfem::element::medium_tag::acoustic,
//                                           specfem::element::medium_tag::elastic_psv>
//
// Usage: for_each_in_product<DIMENSION_T(dim2), MEDIUM_T(...)>(lambda);
//        using ET =
//        specfem::tag_dispatch::element_combinations<DIMENSION_T(dim2),
//        MEDIUM_T(...)>;
//
// BOOST_PP provides the variadic-argument map needed to prefix each element
// with its fully-qualified enum namespace.

#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/transform.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>

// ── Per-tag namespace-prefix helpers (SEQ_TRANSFORM callbacks) ───────────────

#define SPECFEM_PP_DIM_VAL(s, _, v) specfem::element::dimension_tag::v
#define SPECFEM_PP_MED_VAL(s, _, v) specfem::element::medium_tag::v
#define SPECFEM_PP_PROP_VAL(s, _, v) specfem::element::property_tag::v
#define SPECFEM_PP_ATT_VAL(s, _, v) specfem::element::attenuation_tag::v
#define SPECFEM_PP_BND_VAL(s, _, v) specfem::element::boundary_tag::v

// Internal: map op over __VA_ARGS__ and produce a comma-separated list
#define _SPECFEM_TAG_ENUM(op, ...)                                             \
  BOOST_PP_SEQ_ENUM(                                                           \
      BOOST_PP_SEQ_TRANSFORM(op, ~, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)))

// ── Public macros
// ─────────────────────────────────────────────────────────────

#define DIMENSION_T(...)                                                       \
  specfem::tag_dispatch::dimension_set<_SPECFEM_TAG_ENUM(SPECFEM_PP_DIM_VAL,   \
                                                         __VA_ARGS__)>
#define MEDIUM_T(...)                                                          \
  specfem::tag_dispatch::medium_set<_SPECFEM_TAG_ENUM(SPECFEM_PP_MED_VAL,      \
                                                      __VA_ARGS__)>
#define PROPERTY_T(...)                                                        \
  specfem::tag_dispatch::property_set<_SPECFEM_TAG_ENUM(SPECFEM_PP_PROP_VAL,   \
                                                        __VA_ARGS__)>
#define ATTENUATION_T(...)                                                     \
  specfem::tag_dispatch::attenuation_set<_SPECFEM_TAG_ENUM(SPECFEM_PP_ATT_VAL, \
                                                           __VA_ARGS__)>
#define BOUNDARY_T(...)                                                        \
  specfem::tag_dispatch::boundary_set<_SPECFEM_TAG_ENUM(SPECFEM_PP_BND_VAL,    \
                                                        __VA_ARGS__)>
