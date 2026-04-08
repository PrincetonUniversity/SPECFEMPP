#pragma once

// tag_macros.hpp — type-form shorthand macros for tag-set types
//
//   DIMENSION_SET(dim2)
//     =>
//     specfem::tag_dispatch::dimension_set<specfem::element::dimension_tag::dim2>
//
//   MEDIUM_SET(acoustic, elastic_psv)
//     =>
//     specfem::tag_dispatch::medium_set<specfem::element::medium_tag::acoustic,
//                                           specfem::element::medium_tag::elastic_psv>
//
// Usage: for_each_in_product<DIMENSION_SET(dim2), MEDIUM_SET(...)>(lambda);
//        using ET =
//        specfem::tag_dispatch::element_combinations<DIMENSION_SET(dim2),
//        MEDIUM_SET(...)>;
//
// BOOST_PP provides the variadic-argument map needed to prefix each element
// with its fully-qualified enum namespace.

#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/transform.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>

// ── Per-tag namespace-prefix helpers (SEQ_TRANSFORM callbacks) ───────────────

#define _SPECFEM_DIM_VAL(s, _, v) specfem::element::dimension_tag::v
#define _SPECFEM_MED_VAL(s, _, v) specfem::element::medium_tag::v
#define _SPECFEM_PROP_VAL(s, _, v) specfem::element::property_tag::v
#define _SPECFEM_ATT_VAL(s, _, v) specfem::element::attenuation_tag::v
#define _SPECFEM_BND_VAL(s, _, v) specfem::element::boundary_tag::v

// Internal: map op over __VA_ARGS__ and produce a comma-separated list
#define _SPECFEM_TAG_ENUM(op, ...)                                             \
  BOOST_PP_SEQ_ENUM(                                                           \
      BOOST_PP_SEQ_TRANSFORM(op, ~, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)))

// ── Public macros
// ─────────────────────────────────────────────────────────────

#define DIMENSION_SET(...)                                                     \
  specfem::tag_dispatch::dimension_set<_SPECFEM_TAG_ENUM(_SPECFEM_DIM_VAL,     \
                                                         __VA_ARGS__)> {}
#define MEDIUM_SET(...)                                                        \
  specfem::tag_dispatch::medium_set<_SPECFEM_TAG_ENUM(_SPECFEM_MED_VAL,        \
                                                      __VA_ARGS__)> {}
#define PROPERTY_SET(...)                                                      \
  specfem::tag_dispatch::property_set<_SPECFEM_TAG_ENUM(_SPECFEM_PROP_VAL,     \
                                                        __VA_ARGS__)> {}
#define ATTENUATION_SET(...)                                                   \
  specfem::tag_dispatch::attenuation_set<_SPECFEM_TAG_ENUM(_SPECFEM_ATT_VAL,   \
                                                           __VA_ARGS__)> {}
#define BOUNDARY_SET(...)                                                      \
  specfem::tag_dispatch::boundary_set<_SPECFEM_TAG_ENUM(_SPECFEM_BND_VAL,      \
                                                        __VA_ARGS__)> {}

#define _SPECFEM_CONN_VAL(s, _, v) specfem::element_connections::type::v
#define _SPECFEM_IFACE_VAL(s, _, v) specfem::element_coupling::interface_tag::v
#define _SPECFEM_FLUX_VAL(s, _, v) specfem::element_coupling::flux_scheme_tag::v

#define CONNECTION_SET(...)                                                    \
  specfem::tag_dispatch::connection_set<_SPECFEM_TAG_ENUM(_SPECFEM_CONN_VAL,   \
                                                          __VA_ARGS__)> {}
#define INTERFACE_SET(...)                                                     \
  specfem::tag_dispatch::interface_set<_SPECFEM_TAG_ENUM(_SPECFEM_IFACE_VAL,   \
                                                         __VA_ARGS__)> {}
#define FLUX_SCHEME_SET(...)                                                   \
  specfem::tag_dispatch::flux_scheme_set<_SPECFEM_TAG_ENUM(_SPECFEM_FLUX_VAL,  \
                                                           __VA_ARGS__)> {}
