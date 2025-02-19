#include "medium/compute_mass_matrix.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/material_definitions.hpp"
#include "enumerations/medium.hpp"

// This file only contains explicit template instantiations

#define INSTANTIATION_MACRO(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG)           \
  template specfem::point::field<GET_TAG(DIMENSION_TAG), GET_TAG(MEDIUM_TAG),  \
                                 false, false, false, true, false>             \
  specfem::medium::impl_mass_matrix_component(                                 \
      const specfem::point::properties<GET_TAG(DIMENSION_TAG),                 \
                                       GET_TAG(MEDIUM_TAG),                    \
                                       GET_TAG(PROPERTY_TAG), false> &,        \
      const specfem::point::partial_derivatives<GET_TAG(DIMENSION_TAG), true,  \
                                                false> &);                     \
                                                                               \
  template specfem::point::field<GET_TAG(DIMENSION_TAG), GET_TAG(MEDIUM_TAG),  \
                                 false, false, false, true, true>              \
  specfem::medium::impl_mass_matrix_component(                                 \
      const specfem::point::properties<GET_TAG(DIMENSION_TAG),                 \
                                       GET_TAG(MEDIUM_TAG),                    \
                                       GET_TAG(PROPERTY_TAG), true> &,         \
      const specfem::point::partial_derivatives<GET_TAG(DIMENSION_TAG), true,  \
                                                true> &);

CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(INSTANTIATION_MACRO,
                                    WHERE(DIMENSION_TAG_DIM2)
                                        WHERE(MEDIUM_TAG_ELASTIC_SV)
                                            WHERE(PROPERTY_TAG_ISOTROPIC,
                                                  PROPERTY_TAG_ANISOTROPIC))

#undef INSTANTIATION_MACRO
