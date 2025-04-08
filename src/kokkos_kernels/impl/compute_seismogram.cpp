#include "kokkos_kernels/impl/compute_seismogram.hpp"
#include "enumerations/material_definitions.hpp"
#include "kokkos_kernels/impl/compute_seismogram.tpp"

#define INSTANTIATION_MACRO(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG)           \
  /** instantiation for NGLL = 5     */                                        \
  template void specfem::kokkos_kernels::impl::compute_seismograms<            \
      GET_TAG(DIMENSION_TAG), specfem::wavefield::simulation_field::forward,   \
      5, GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG)>(                          \
      specfem::compute::assembly &, const int &);                              \
  template void specfem::kokkos_kernels::impl::compute_seismograms<            \
      GET_TAG(DIMENSION_TAG), specfem::wavefield::simulation_field::backward,  \
      5, GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG)>(                          \
      specfem::compute::assembly &, const int &);                              \
  template void specfem::kokkos_kernels::impl::compute_seismograms<            \
      GET_TAG(DIMENSION_TAG), specfem::wavefield::simulation_field::adjoint,   \
      5, GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG)>(                          \
      specfem::compute::assembly &, const int &);                              \
  /** instantiation for NGLL = 8     */                                        \
  template void specfem::kokkos_kernels::impl::compute_seismograms<            \
      GET_TAG(DIMENSION_TAG), specfem::wavefield::simulation_field::forward,   \
      8, GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG)>(                          \
      specfem::compute::assembly &, const int &);                              \
  template void specfem::kokkos_kernels::impl::compute_seismograms<            \
      GET_TAG(DIMENSION_TAG), specfem::wavefield::simulation_field::backward,  \
      8, GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG)>(                          \
      specfem::compute::assembly &, const int &);                              \
  template void specfem::kokkos_kernels::impl::compute_seismograms<            \
      GET_TAG(DIMENSION_TAG), specfem::wavefield::simulation_field::adjoint,   \
      8, GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG)>(                          \
      specfem::compute::assembly &, const int &);

CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
    INSTANTIATION_MACRO,
    WHERE(DIMENSION_TAG_DIM2)
        WHERE(MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH,
              MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC,
              MEDIUM_TAG_ELASTIC_PSV_T)
            WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC,
                  PROPERTY_TAG_ISOTROPIC_COSSERAT))

#undef INSTANTIATION_MACRO
