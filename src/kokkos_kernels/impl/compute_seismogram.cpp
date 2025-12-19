#include "kokkos_kernels/impl/compute_seismogram.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_kernels/impl/compute_seismogram.tpp"
#include "specfem/assembly.hpp"
#include "specfem/macros.hpp"

FOR_EACH_IN_PRODUCT(
    (DIMENSION_TAG(DIM2),
     MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC, ELASTIC_PSV_T),
     PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
    INSTANTIATE(
        /** instantiation for NGLL = 5     */
        (template void specfem::kokkos_kernels::impl::compute_seismograms,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::forward, 5,
          _MEDIUM_TAG_, _PROPERTY_TAG_),
         (specfem::assembly::assembly<specfem::dimension::type::dim2> &,
          const int &);),
        (template void specfem::kokkos_kernels::impl::compute_seismograms,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::backward, 5,
          _MEDIUM_TAG_, _PROPERTY_TAG_),
         (specfem::assembly::assembly<specfem::dimension::type::dim2> &,
          const int &);),
        (template void specfem::kokkos_kernels::impl::compute_seismograms,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::adjoint, 5,
          _MEDIUM_TAG_, _PROPERTY_TAG_),
         (specfem::assembly::assembly<specfem::dimension::type::dim2> &,
          const int &);),
        /** instantiation for NGLL = 8     */
        (template void specfem::kokkos_kernels::impl::compute_seismograms,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::forward, 8,
          _MEDIUM_TAG_, _PROPERTY_TAG_),
         (specfem::assembly::assembly<specfem::dimension::type::dim2> &,
          const int &);),
        (template void specfem::kokkos_kernels::impl::compute_seismograms,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::backward, 8,
          _MEDIUM_TAG_, _PROPERTY_TAG_),
         (specfem::assembly::assembly<specfem::dimension::type::dim2> &,
          const int &);),
        (template void specfem::kokkos_kernels::impl::compute_seismograms,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::adjoint, 8,
          _MEDIUM_TAG_, _PROPERTY_TAG_),
         (specfem::assembly::assembly<specfem::dimension::type::dim2> &,
          const int &);)))

FOR_EACH_IN_PRODUCT(
    (DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC, ACOUSTIC),
     PROPERTY_TAG(ISOTROPIC)),
    INSTANTIATE(
        /** instantiation for NGLL = 5     */
        (template void specfem::kokkos_kernels::impl::compute_seismograms,
         (_DIMENSION_TAG_, specfem::wavefield::simulation_field::forward, 5,
          _MEDIUM_TAG_, _PROPERTY_TAG_),
         (specfem::assembly::assembly<specfem::dimension::type::dim3> &,
          const int &);), ))
