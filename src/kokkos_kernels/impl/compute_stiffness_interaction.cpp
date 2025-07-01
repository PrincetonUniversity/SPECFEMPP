#include "kokkos_kernels/impl/compute_stiffness_interaction.hpp"
#include "enumerations/material_definitions.hpp"
#include "kokkos_kernels/impl/compute_stiffness_interaction.tpp"

FOR_EACH_IN_PRODUCT(
    (DIMENSION_TAG(DIM2),
     MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC, ELASTIC_PSV_T),
     PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT),
     BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                  COMPOSITE_STACEY_DIRICHLET)),
    INSTANTIATE(
        /** instantiation for NGLL = 5     */
        (template int
             specfem::kokkos_kernels::impl::compute_stiffness_interaction,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::forward, 5,
          _MEDIUM_TAG_, _PROPERTY_TAG_, _BOUNDARY_TAG_),
         (const specfem::assembly::assembly &, const int &);),
        (template int
             specfem::kokkos_kernels::impl::compute_stiffness_interaction,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::backward, 5,
          _MEDIUM_TAG_, _PROPERTY_TAG_, _BOUNDARY_TAG_),
         (const specfem::assembly::assembly &, const int &);),
        (template int
             specfem::kokkos_kernels::impl::compute_stiffness_interaction,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::adjoint, 5,
          _MEDIUM_TAG_, _PROPERTY_TAG_, _BOUNDARY_TAG_),
         (const specfem::assembly::assembly &, const int &);),
        /** instantiation for NGLL = 8     */
        (template int
             specfem::kokkos_kernels::impl::compute_stiffness_interaction,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::forward, 8,
          _MEDIUM_TAG_, _PROPERTY_TAG_, _BOUNDARY_TAG_),
         (const specfem::assembly::assembly &, const int &);),
        (template int
             specfem::kokkos_kernels::impl::compute_stiffness_interaction,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::backward, 8,
          _MEDIUM_TAG_, _PROPERTY_TAG_, _BOUNDARY_TAG_),
         (const specfem::assembly::assembly &, const int &);),
        (template int
             specfem::kokkos_kernels::impl::compute_stiffness_interaction,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::adjoint, 8,
          _MEDIUM_TAG_, _PROPERTY_TAG_, _BOUNDARY_TAG_),
         (const specfem::assembly::assembly &, const int &);)))
