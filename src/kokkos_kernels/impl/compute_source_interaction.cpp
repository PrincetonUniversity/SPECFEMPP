#include "kokkos_kernels/impl/compute_source_interaction.hpp"
#include "enumerations/material_definitions.hpp"
#include "kokkos_kernels/impl/compute_source_interaction.tpp"

FOR_EACH(IN_PRODUCT((DIMENSION_TAG_DIM2),
                    (MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH,
                     MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC),
                    (PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC),
                    (BOUNDARY_TAG_NONE, BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
                     BOUNDARY_TAG_STACEY,
                     BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET)),
         INSTANTIATE(
             /** instantiation for NGLL = 5     */
             (template void
                  specfem::kokkos_kernels::impl::compute_source_interaction,
              (_DIMENSION_TAG_, specfem::wavefield::simulation_field::forward,
               5, _MEDIUM_TAG_, _PROPERTY_TAG_, _BOUNDARY_TAG_),
              (specfem::compute::assembly &, const int &);),
             (template void
                  specfem::kokkos_kernels::impl::compute_source_interaction,
              (_DIMENSION_TAG_, specfem::wavefield::simulation_field::backward,
               5, _MEDIUM_TAG_, _PROPERTY_TAG_, _BOUNDARY_TAG_),
              (specfem::compute::assembly &, const int &);),
             (template void
                  specfem::kokkos_kernels::impl::compute_source_interaction,
              (_DIMENSION_TAG_, specfem::wavefield::simulation_field::adjoint,
               5, _MEDIUM_TAG_, _PROPERTY_TAG_, _BOUNDARY_TAG_),
              (specfem::compute::assembly &, const int &);),
             /** instantiation for NGLL = 8     */
             (template void
                  specfem::kokkos_kernels::impl::compute_source_interaction,
              (_DIMENSION_TAG_, specfem::wavefield::simulation_field::forward,
               8, _MEDIUM_TAG_, _PROPERTY_TAG_, _BOUNDARY_TAG_),
              (specfem::compute::assembly &, const int &);),
             (template void
                  specfem::kokkos_kernels::impl::compute_source_interaction,
              (_DIMENSION_TAG_, specfem::wavefield::simulation_field::backward,
               8, _MEDIUM_TAG_, _PROPERTY_TAG_, _BOUNDARY_TAG_),
              (specfem::compute::assembly &, const int &);),
             (template void
                  specfem::kokkos_kernels::impl::compute_source_interaction,
              (_DIMENSION_TAG_, specfem::wavefield::simulation_field::adjoint,
               8, _MEDIUM_TAG_, _PROPERTY_TAG_, _BOUNDARY_TAG_),
              (specfem::compute::assembly &, const int &);)))
