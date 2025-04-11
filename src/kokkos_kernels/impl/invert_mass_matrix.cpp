#include "kokkos_kernels/impl/invert_mass_matrix.hpp"
#include "kokkos_kernels/impl/invert_mass_matrix.tpp"

FOR_EACH(IN_PRODUCT((DIMENSION_TAG_DIM2),
                    (MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH,
                     MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC)),
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
