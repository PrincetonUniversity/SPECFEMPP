#include "kernels/frechet_kernels.hpp"
#include "kernels/frechet_kernels.tpp"

#define INSTANTIATION_MACRO(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG)           \
  template void specfem::kernels::frechet_kernels<DIMENSION_TAG, 5>::          \
      compute_material_derivatives<MEDIUM_TAG, PROPERTY_TAG>(                  \
          const type_real &);                                                  \
  template void specfem::kernels::frechet_kernels<DIMENSION_TAG, 8>::          \
      compute_material_derivatives<MEDIUM_TAG, PROPERTY_TAG>(                  \
          const type_real &);

CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
    INSTANTIATION_MACRO,
    WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC)
        WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC))

#undef INSTANTIATION_MACRO

// Explicit template instantiation
template class specfem::kernels::frechet_kernels<specfem::dimension::type::dim2,
                                                 5>;

template class specfem::kernels::frechet_kernels<specfem::dimension::type::dim2,
                                                 8>;
