#include "kokkos_kernels/impl/compute_material_derivatives.hpp"
#include "kokkos_kernels/impl/compute_material_derivatives.tpp"

#define INSTANTIATION_MACRO(_, DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG)        \
  /** instantiation for NGLL = 5     */                                        \
  template void specfem::kokkos_kernels::impl::compute_material_derivatives<   \
      DIMENSION_TAG, 5, MEDIUM_TAG, PROPERTY_TAG>(                             \
      const specfem::compute::assembly &, const type_real &);                  \
  /** instantiation for NGLL = 8     */                                        \
  template void specfem::kokkos_kernels::impl::compute_material_derivatives<   \
      DIMENSION_TAG, 8, MEDIUM_TAG, PROPERTY_TAG>(                             \
      const specfem::compute::assembly &, const type_real &);

CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS2(
    WHERE(DIMENSION_TAG_DIM2)
        WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ELASTIC_SH, MEDIUM_TAG_ACOUSTIC)
            WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC),
    INSTANTIATION_MACRO)

#undef INSTANTIATION_MACRO
