#include "kokkos_kernels/impl/compute_material_derivatives.hpp"
#include "kokkos_kernels/impl/compute_material_derivatives.tpp"

FOR_EACH_MATERIAL_SYSTEM(
    WHERE2((DIMENSION_TAG_DIM2),
           (MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH, MEDIUM_TAG_ACOUSTIC,
            MEDIUM_TAG_POROELASTIC),
           (PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC)),
    INSTANTIATE(
        /** instantiation for NGLL = 5     */
        (template void
             specfem::kokkos_kernels::impl::compute_material_derivatives,
         (_DIMENSION_TAG_, 5, _MEDIUM_TAG_, _PROPERTY_TAG_),
         (const specfem::compute::assembly &, const type_real &);),
        /** instantiation for NGLL = 8     */
        (template void
             specfem::kokkos_kernels::impl::compute_material_derivatives,
         (_DIMENSION_TAG_, 8, _MEDIUM_TAG_, _PROPERTY_TAG_),
         (const specfem::compute::assembly &, const type_real &);)))
