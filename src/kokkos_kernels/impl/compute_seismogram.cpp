#include "kokkos_kernels/impl/compute_seismogram.hpp"
#include "enumerations/material_definitions.hpp"
#include "kokkos_kernels/impl/compute_seismogram.tpp"

FOR_EACH_IN_PRODUCT(
    (DIMENSION_TAG(DIM2),
     MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC),
     PROPERTY_TAG(ISOTROPIC, ANISOTROPIC)),
    INSTANTIATE(
        /** instantiation for NGLL = 5     */
        (template void specfem::kokkos_kernels::impl::compute_seismograms,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::forward, 5,
          _MEDIUM_TAG_, _PROPERTY_TAG_),
         (specfem::compute::assembly &, const int &);),
        (template void specfem::kokkos_kernels::impl::compute_seismograms,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::backward, 5,
          _MEDIUM_TAG_, _PROPERTY_TAG_),
         (specfem::compute::assembly &, const int &);),
        (template void specfem::kokkos_kernels::impl::compute_seismograms,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::adjoint, 5,
          _MEDIUM_TAG_, _PROPERTY_TAG_),
         (specfem::compute::assembly &, const int &);),
        /** instantiation for NGLL = 8     */
        (template void specfem::kokkos_kernels::impl::compute_seismograms,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::forward, 8,
          _MEDIUM_TAG_, _PROPERTY_TAG_),
         (specfem::compute::assembly &, const int &);),
        (template void specfem::kokkos_kernels::impl::compute_seismograms,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::backward, 8,
          _MEDIUM_TAG_, _PROPERTY_TAG_),
         (specfem::compute::assembly &, const int &);),
        (template void specfem::kokkos_kernels::impl::compute_seismograms,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::adjoint, 8,
          _MEDIUM_TAG_, _PROPERTY_TAG_),
         (specfem::compute::assembly &, const int &);)))
