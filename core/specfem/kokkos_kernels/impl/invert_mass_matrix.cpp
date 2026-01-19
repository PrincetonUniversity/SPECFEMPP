#include "invert_mass_matrix.hpp"
#include "enumerations/interface.hpp"
#include "invert_mass_matrix.tpp"
#include "specfem/assembly.hpp"

FOR_EACH_IN_PRODUCT(
    (DIMENSION_TAG(DIM2),
     MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC, ELASTIC_PSV_T)),
    INSTANTIATE(
        (template void specfem::kokkos_kernels::impl::invert_mass_matrix,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::forward,
          _MEDIUM_TAG_),
         (const specfem::assembly::assembly<specfem::dimension::type::dim2>
              &);),
        (template void specfem::kokkos_kernels::impl::invert_mass_matrix,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::backward,
          _MEDIUM_TAG_),
         (const specfem::assembly::assembly<specfem::dimension::type::dim2>
              &);),
        (template void specfem::kokkos_kernels::impl::invert_mass_matrix,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::adjoint,
          _MEDIUM_TAG_),
         (const specfem::assembly::assembly<specfem::dimension::type::dim2>
              &);)))

FOR_EACH_IN_PRODUCT(
    (DIMENSION_TAG(DIM3), MEDIUM_TAG(ACOUSTIC, ELASTIC)),
    INSTANTIATE(
        (template void specfem::kokkos_kernels::impl::invert_mass_matrix,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::forward,
          _MEDIUM_TAG_),
         (const specfem::assembly::assembly<specfem::dimension::type::dim3>
              &);),
        (template void specfem::kokkos_kernels::impl::invert_mass_matrix,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::backward,
          _MEDIUM_TAG_),
         (const specfem::assembly::assembly<specfem::dimension::type::dim3>
              &);),
        (template void specfem::kokkos_kernels::impl::invert_mass_matrix,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::adjoint,
          _MEDIUM_TAG_),
         (const specfem::assembly::assembly<specfem::dimension::type::dim3>
              &);)))
