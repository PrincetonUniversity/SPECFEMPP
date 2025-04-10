#include "kokkos_kernels/impl/compute_energy.hpp"
#include "enumerations/material_definitions.hpp"
#include "kokkos_kernels/impl/compute_energy.tpp"

#define INSTANTIATION_MACRO(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG)           \
  template type_real specfem::kokkos_kernels::impl::compute_energy<            \
      GET_TAG(DIMENSION_TAG), specfem::wavefield::simulation_field::forward,   \
      5, GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG)>(                          \
      const specfem::compute::assembly &);                                     \
  template type_real specfem::kokkos_kernels::impl::compute_energy<            \
      GET_TAG(DIMENSION_TAG), specfem::wavefield::simulation_field::backward,  \
      5, GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG)>(                          \
      const specfem::compute::assembly &);                                     \
  template type_real specfem::kokkos_kernels::impl::compute_energy<            \
      GET_TAG(DIMENSION_TAG), specfem::wavefield::simulation_field::adjoint,   \
      5, GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG)>(                          \
      const specfem::compute::assembly &);                                     \
  template type_real specfem::kokkos_kernels::impl::compute_energy<            \
      GET_TAG(DIMENSION_TAG), specfem::wavefield::simulation_field::forward,   \
      8, GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG)>(                          \
      const specfem::compute::assembly &);                                     \
  template type_real specfem::kokkos_kernels::impl::compute_energy<            \
      GET_TAG(DIMENSION_TAG), specfem::wavefield::simulation_field::backward,  \
      8, GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG)>(                          \
      const specfem::compute::assembly &);                                     \
  template type_real specfem::kokkos_kernels::impl::compute_energy<            \
      GET_TAG(DIMENSION_TAG), specfem::wavefield::simulation_field::adjoint,   \
      8, GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG)>(                          \
      const specfem::compute::assembly &);

CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
    INSTANTIATION_MACRO,
    WHERE(DIMENSION_TAG_DIM2)
        WHERE(MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH,
              MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC)
            WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC))

#undef INSTANTIATION_MACRO
