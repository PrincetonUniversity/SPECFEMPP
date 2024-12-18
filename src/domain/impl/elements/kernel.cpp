#include "domain/impl/elements/kernel.hpp"
#include "domain/impl/elements/kernel.tpp"
#include "enumerations/material_definitions.hpp"

constexpr static auto dim2 = specfem::dimension::type::dim2;

constexpr static auto forward = specfem::wavefield::simulation_field::forward;
constexpr static auto adjoint = specfem::wavefield::simulation_field::adjoint;
constexpr static auto backward = specfem::wavefield::simulation_field::backward;

#define INSTANTIATION_MACRO(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG,           \
                            BOUNDARY_TAG)                                      \
  template class specfem::domain::impl::kernels::element_kernel_base<          \
      forward, DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, BOUNDARY_TAG, 5>;      \
  template class specfem::domain::impl::kernels::element_kernel_base<          \
      adjoint, DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, BOUNDARY_TAG, 5>;      \
  template class specfem::domain::impl::kernels::element_kernel_base<          \
      backward, DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, BOUNDARY_TAG, 5>;     \
  template class specfem::domain::impl::kernels::element_kernel_base<          \
      forward, DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, BOUNDARY_TAG, 8>;      \
  template class specfem::domain::impl::kernels::element_kernel_base<          \
      adjoint, DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, BOUNDARY_TAG, 8>;      \
  template class specfem::domain::impl::kernels::element_kernel_base<          \
      backward, DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, BOUNDARY_TAG, 8>;     \
  template class specfem::domain::impl::kernels::element_kernel<               \
      forward, DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, BOUNDARY_TAG, 5>;      \
  template class specfem::domain::impl::kernels::element_kernel<               \
      adjoint, DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, BOUNDARY_TAG, 5>;      \
  template class specfem::domain::impl::kernels::element_kernel<               \
      backward, DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, BOUNDARY_TAG, 5>;     \
  template class specfem::domain::impl::kernels::element_kernel<               \
      forward, DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, BOUNDARY_TAG, 8>;      \
  template class specfem::domain::impl::kernels::element_kernel<               \
      adjoint, DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, BOUNDARY_TAG, 8>;      \
  template class specfem::domain::impl::kernels::element_kernel<               \
      backward, DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, BOUNDARY_TAG, 8>;

CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
    INSTANTIATION_MACRO,
    WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC)
        WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC)
            WHERE(BOUNDARY_TAG_NONE, BOUNDARY_TAG_STACEY,
                  BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
                  BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))
