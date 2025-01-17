#include "kernels/impl/compute_source_interaction.hpp"
#include "enumerations/material_definitions.hpp"
#include "kernels/impl/compute_source_interaction.tpp"

#define INSTANTIATION_MACRO(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG,           \
                            BOUNDARY_TAG)                                      \
  /** instantiation for NGLL = 5     */                                        \
  template void specfem::kernels::impl::compute_source_interaction<            \
      GET_TAG(DIMENSION_TAG), specfem::wavefield::simulation_field::forward,   \
      5, GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG), GET_TAG(BOUNDARY_TAG)>(   \
      specfem::compute::assembly &, const int &);                              \
  template void specfem::kernels::impl::compute_source_interaction<            \
      GET_TAG(DIMENSION_TAG), specfem::wavefield::simulation_field::backward,  \
      5, GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG), GET_TAG(BOUNDARY_TAG)>(   \
      specfem::compute::assembly &, const int &);                              \
  template void specfem::kernels::impl::compute_source_interaction<            \
      GET_TAG(DIMENSION_TAG), specfem::wavefield::simulation_field::adjoint,   \
      5, GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG), GET_TAG(BOUNDARY_TAG)>(   \
      specfem::compute::assembly &, const int &);                              \
  /** instantiation for NGLL = 8     */                                        \
  template void specfem::kernels::impl::compute_source_interaction<            \
      GET_TAG(DIMENSION_TAG), specfem::wavefield::simulation_field::forward,   \
      8, GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG), GET_TAG(BOUNDARY_TAG)>(   \
      specfem::compute::assembly &, const int &);                              \
  template void specfem::kernels::impl::compute_source_interaction<            \
      GET_TAG(DIMENSION_TAG), specfem::wavefield::simulation_field::backward,  \
      8, GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG), GET_TAG(BOUNDARY_TAG)>(   \
      specfem::compute::assembly &, const int &);                              \
  template void specfem::kernels::impl::compute_source_interaction<            \
      GET_TAG(DIMENSION_TAG), specfem::wavefield::simulation_field::adjoint,   \
      8, GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG), GET_TAG(BOUNDARY_TAG)>(   \
      specfem::compute::assembly &, const int &);

CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
    INSTANTIATION_MACRO,
    WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC)
        WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC)
            WHERE(BOUNDARY_TAG_STACEY, BOUNDARY_TAG_NONE,
                  BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
                  BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef INSTANTIATION_MACRO
