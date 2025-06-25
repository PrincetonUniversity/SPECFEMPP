#include "kokkos_kernels/impl/invert_mass_matrix.hpp"
#include "kokkos_kernels/impl/invert_mass_matrix.tpp"

FOR_EACH_IN_PRODUCT(
    (DIMENSION_TAG(DIM2),
     MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC, ELASTIC_PSV_T)),
    INSTANTIATE(
        (template void specfem::kokkos_kernels::impl::invert_mass_matrix,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::forward,
          _MEDIUM_TAG_),
         (const specfem::compute::assembly &);),
        (template void specfem::kokkos_kernels::impl::invert_mass_matrix,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::backward,
          _MEDIUM_TAG_),
         (const specfem::compute::assembly &);),
        (template void specfem::kokkos_kernels::impl::invert_mass_matrix,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::adjoint,
          _MEDIUM_TAG_),
         (const specfem::compute::assembly &);)))
