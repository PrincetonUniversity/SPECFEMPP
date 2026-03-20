#pragma once
// !! Do NOT include this from specfem/macros.hpp !!
// Include only in explicit-instantiation .cpp files.

// ── Signature helpers
// ───────────────────────────────────────────────────────── Callers must have
// the following using-declarations in scope:
//   using specfem::element::dimension_tag;
//   using specfem::element::medium_tag;
//   using specfem::simulation::field_type;

// void(const assembly &)                  — compute_coupling,
// divide_mass_matrix
#define INSTANTIATE_COMPUTE_FUNCTION(FN, NGLL, DIM, WF, MED)                   \
  template void                                                                \
  specfem::compute::impl::FN<NGLL, specfem::tags::Tags<DIM, WF, MED> >(        \
      const specfem::assembly::assembly<DIM> &);

// void(assembly &, const int)             — compute_source_interaction
#define INSTANTIATE_COMPUTE_FUNCTION_VOID_NONCONST_INT(FN, NGLL, DIM, WF, MED) \
  template void                                                                \
  specfem::compute::impl::FN<NGLL, specfem::tags::Tags<DIM, WF, MED> >(        \
      specfem::assembly::assembly<DIM> &, const int);

// int(const assembly &, const int)        — compute_stiffness_interaction
#define INSTANTIATE_COMPUTE_FUNCTION_INT_CONST_INT(FN, NGLL, DIM, WF, MED)     \
  template int                                                                 \
  specfem::compute::impl::FN<NGLL, specfem::tags::Tags<DIM, WF, MED> >(        \
      const specfem::assembly::assembly<DIM> &, const int);

// int(assembly &, const int)              — update_wavefields
#define INSTANTIATE_COMPUTE_FUNCTION_INT_NONCONST_INT(FN, NGLL, DIM, WF, MED)  \
  template int specfem::compute::FN<NGLL, specfem::tags::Tags<DIM, WF, MED> >( \
      specfem::assembly::assembly<DIM> &,                                      \
      const int); // update_wavefields stays in compute::

// ── Combination list (X-macro)
// ──────────────────────────────────────────────── All (NGLL, DIM, WF, MED)
// tuples used by time_marching instantiations. Requires dimension_tag /
// field_type / medium_tag to be in scope (see above).
#define SPECFEM_COMPUTE_COMBINATIONS(MACRO)                                    \
  /* dim2, NGLL=5, forward */                                                  \
  MACRO(5, dimension_tag::dim2, field_type::forward, medium_tag::acoustic)     \
  MACRO(5, dimension_tag::dim2, field_type::forward, medium_tag::elastic)      \
  MACRO(5, dimension_tag::dim2, field_type::forward, medium_tag::elastic_psv)  \
  MACRO(5, dimension_tag::dim2, field_type::forward, medium_tag::elastic_sh)   \
  MACRO(5, dimension_tag::dim2, field_type::forward,                           \
        medium_tag::elastic_psv_t)                                             \
  MACRO(5, dimension_tag::dim2, field_type::forward, medium_tag::poroelastic)  \
  /* dim2, NGLL=8, forward */                                                  \
  MACRO(8, dimension_tag::dim2, field_type::forward, medium_tag::acoustic)     \
  MACRO(8, dimension_tag::dim2, field_type::forward, medium_tag::elastic)      \
  MACRO(8, dimension_tag::dim2, field_type::forward, medium_tag::elastic_psv)  \
  MACRO(8, dimension_tag::dim2, field_type::forward, medium_tag::elastic_sh)   \
  MACRO(8, dimension_tag::dim2, field_type::forward,                           \
        medium_tag::elastic_psv_t)                                             \
  MACRO(8, dimension_tag::dim2, field_type::forward, medium_tag::poroelastic)  \
  /* dim2, NGLL=5, adjoint */                                                  \
  MACRO(5, dimension_tag::dim2, field_type::adjoint, medium_tag::acoustic)     \
  MACRO(5, dimension_tag::dim2, field_type::adjoint, medium_tag::elastic)      \
  MACRO(5, dimension_tag::dim2, field_type::adjoint, medium_tag::elastic_psv)  \
  MACRO(5, dimension_tag::dim2, field_type::adjoint, medium_tag::elastic_sh)   \
  MACRO(5, dimension_tag::dim2, field_type::adjoint, medium_tag::poroelastic)  \
  /* dim2, NGLL=5, backward */                                                 \
  MACRO(5, dimension_tag::dim2, field_type::backward, medium_tag::acoustic)    \
  MACRO(5, dimension_tag::dim2, field_type::backward, medium_tag::elastic)     \
  MACRO(5, dimension_tag::dim2, field_type::backward, medium_tag::elastic_psv) \
  MACRO(5, dimension_tag::dim2, field_type::backward, medium_tag::elastic_sh)  \
  MACRO(5, dimension_tag::dim2, field_type::backward, medium_tag::poroelastic) \
  /* dim2, NGLL=8, adjoint */                                                  \
  MACRO(8, dimension_tag::dim2, field_type::adjoint, medium_tag::acoustic)     \
  MACRO(8, dimension_tag::dim2, field_type::adjoint, medium_tag::elastic)      \
  MACRO(8, dimension_tag::dim2, field_type::adjoint, medium_tag::elastic_psv)  \
  MACRO(8, dimension_tag::dim2, field_type::adjoint, medium_tag::elastic_sh)   \
  MACRO(8, dimension_tag::dim2, field_type::adjoint, medium_tag::poroelastic)  \
  /* dim2, NGLL=8, backward */                                                 \
  MACRO(8, dimension_tag::dim2, field_type::backward, medium_tag::acoustic)    \
  MACRO(8, dimension_tag::dim2, field_type::backward, medium_tag::elastic)     \
  MACRO(8, dimension_tag::dim2, field_type::backward, medium_tag::elastic_psv) \
  MACRO(8, dimension_tag::dim2, field_type::backward, medium_tag::elastic_sh)  \
  MACRO(8, dimension_tag::dim2, field_type::backward, medium_tag::poroelastic) \
  /* dim3, NGLL=5, forward */                                                  \
  MACRO(5, dimension_tag::dim3, field_type::forward, medium_tag::acoustic)     \
  MACRO(5, dimension_tag::dim3, field_type::forward, medium_tag::elastic)      \
  MACRO(5, dimension_tag::dim3, field_type::forward, medium_tag::elastic_psv)  \
  MACRO(5, dimension_tag::dim3, field_type::forward, medium_tag::elastic_sh)   \
  MACRO(5, dimension_tag::dim3, field_type::forward,                           \
        medium_tag::elastic_psv_t)                                             \
  MACRO(5, dimension_tag::dim3, field_type::forward, medium_tag::poroelastic)
