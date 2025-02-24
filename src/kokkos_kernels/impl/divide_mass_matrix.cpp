#include "kokkos_kernels/impl/divide_mass_matrix.hpp"
#include "kokkos_kernels/impl/divide_mass_matrix.tpp"

#define INSTANTIATION_MACRO(DIMENSION_TAG, MEDIUM_TAG)                         \
  template void specfem::kokkos_kernels::impl::divide_mass_matrix<             \
      GET_TAG(DIMENSION_TAG), specfem::wavefield::simulation_field::forward,   \
      GET_TAG(MEDIUM_TAG)>(const specfem::compute::assembly &);                \
  template void specfem::kokkos_kernels::impl::divide_mass_matrix<             \
      GET_TAG(DIMENSION_TAG), specfem::wavefield::simulation_field::backward,  \
      GET_TAG(MEDIUM_TAG)>(const specfem::compute::assembly &);                \
  template void specfem::kokkos_kernels::impl::divide_mass_matrix<             \
      GET_TAG(DIMENSION_TAG), specfem::wavefield::simulation_field::adjoint,   \
      GET_TAG(MEDIUM_TAG)>(const specfem::compute::assembly &);

CALL_MACRO_FOR_ALL_MEDIUM_TAGS(INSTANTIATION_MACRO,
                               WHERE(DIMENSION_TAG_DIM2)
                                   WHERE(MEDIUM_TAG_ELASTIC_SV,
                                         MEDIUM_TAG_ACOUSTIC))

#undef INSTANTIATION_MACRO
