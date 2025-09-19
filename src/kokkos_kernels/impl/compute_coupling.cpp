#include "kokkos_kernels/impl/compute_coupling.hpp"
#include "enumerations/interface.hpp"
#include "enumerations/material_definitions.hpp"
#include "kokkos_kernels/impl/compute_coupling.tpp"
#include <type_traits>

FOR_EACH_IN_PRODUCT(
    (DIMENSION_TAG(DIM2), CONNECTION_TAG(WEAKLY_CONFORMING),
     INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
     BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                  COMPOSITE_STACEY_DIRICHLET)),
    INSTANTIATE(
        (template void specfem::kokkos_kernels::impl::compute_coupling,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::forward,
          _INTERFACE_TAG_, _BOUNDARY_TAG_),
         (const std::integral_constant<
              specfem::connections::type,
              specfem::connections::type::weakly_conforming> /*unused*/,
          const specfem::assembly::assembly<specfem::dimension::type::dim2>
              &);),
        (template void specfem::kokkos_kernels::impl::compute_coupling,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::backward,
          _INTERFACE_TAG_, _BOUNDARY_TAG_),
         (const std::integral_constant<
              specfem::connections::type,
              specfem::connections::type::weakly_conforming> /*unused*/,
          const specfem::assembly::assembly<specfem::dimension::type::dim2>
              &);),
        (template void specfem::kokkos_kernels::impl::compute_coupling,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::adjoint,
          _INTERFACE_TAG_, _BOUNDARY_TAG_),
         (const std::integral_constant<
              specfem::connections::type,
              specfem::connections::type::weakly_conforming> /*unused*/,
          const specfem::assembly::assembly<specfem::dimension::type::dim2>
              &);)))
