#include "boundary_conditions/composite_stacey_dirichlet/composite_stacey_dirichlet.hpp"
#include "boundary_conditions/composite_stacey_dirichlet/composite_stacey_dirichlet.tpp"
#include "boundary_conditions/stacey/stacey.hpp"
#include "boundary_conditions/stacey/stacey.tpp"
#include "enumerations/dimension.hpp"
#include "enumerations/material_definitions.hpp"
#include "enumerations/medium.hpp"
#include "point/boundary.hpp"
#include "point/field.hpp"
#include "point/properties.hpp"

namespace {
template <specfem::dimension::type DimensionType,
          specfem::element::boundary_tag BoundaryTag, bool UseSIMD>
using PointBoundaryType =
    specfem::point::boundary<BoundaryTag, DimensionType, UseSIMD>;

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD>
using PointPropertyType =
    specfem::point::properties<DimensionType, MediumTag, PropertyTag, UseSIMD>;

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
using PointMassMatrixType =
    specfem::point::field<DimensionType, MediumTag, false, false, false, true,
                          UseSIMD>;

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
using PointVelocityType = specfem::point::field<DimensionType, MediumTag, false,
                                                true, false, false, UseSIMD>;

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
using PointAccelerationType =
    specfem::point::field<DimensionType, MediumTag, false, false, true, false,
                          UseSIMD>;

template <specfem::element::boundary_tag BoundaryTag>
using boundary_type =
    std::integral_constant<specfem::element::boundary_tag, BoundaryTag>;
} // namespace

#define INSTANTIATION_MACRO(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG,           \
                            BOUNDARY_TAG)                                      \
  /** Template instantiation for SIMD=false */                                 \
  template KOKKOS_FUNCTION void                                                \
  specfem::boundary_conditions::impl_compute_mass_matrix_terms<                \
      PointBoundaryType<GET_TAG(DIMENSION_TAG), GET_TAG(BOUNDARY_TAG), false>, \
      PointPropertyType<GET_TAG(DIMENSION_TAG), GET_TAG(MEDIUM_TAG),           \
                        GET_TAG(PROPERTY_TAG), false>,                         \
      PointMassMatrixType<GET_TAG(DIMENSION_TAG), GET_TAG(MEDIUM_TAG),         \
                          false> >(                                            \
      const boundary_type<GET_TAG(BOUNDARY_TAG)> &, const type_real dt,        \
      const PointBoundaryType<GET_TAG(DIMENSION_TAG), GET_TAG(BOUNDARY_TAG),   \
                              false> &,                                        \
      const PointPropertyType<GET_TAG(DIMENSION_TAG), GET_TAG(MEDIUM_TAG),     \
                              GET_TAG(PROPERTY_TAG), false> &,                 \
      PointMassMatrixType<GET_TAG(DIMENSION_TAG), GET_TAG(MEDIUM_TAG), false>  \
          &);                                                                  \
                                                                               \
  /** Template instantiation for SIMD=true */                                  \
  template KOKKOS_FUNCTION void                                                \
  specfem::boundary_conditions::impl_compute_mass_matrix_terms<                \
      PointBoundaryType<GET_TAG(DIMENSION_TAG), GET_TAG(BOUNDARY_TAG), true>,  \
      PointPropertyType<GET_TAG(DIMENSION_TAG), GET_TAG(MEDIUM_TAG),           \
                        GET_TAG(PROPERTY_TAG), true>,                          \
      PointMassMatrixType<GET_TAG(DIMENSION_TAG), GET_TAG(MEDIUM_TAG),         \
                          true> >(                                             \
      const boundary_type<GET_TAG(BOUNDARY_TAG)> &, const type_real dt,        \
      const PointBoundaryType<GET_TAG(DIMENSION_TAG), GET_TAG(BOUNDARY_TAG),   \
                              true> &,                                         \
      const PointPropertyType<GET_TAG(DIMENSION_TAG), GET_TAG(MEDIUM_TAG),     \
                              GET_TAG(PROPERTY_TAG), true> &,                  \
      PointMassMatrixType<GET_TAG(DIMENSION_TAG), GET_TAG(MEDIUM_TAG), true>   \
          &);

CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
    INSTANTIATION_MACRO,
    WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC)
        WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC)
            WHERE(BOUNDARY_TAG_STACEY, BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef INSTANTIATION_MACRO

#define INSTANTIATION_MACRO(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG,           \
                            BOUNDARY_TAG)                                      \
  /** Template instantiation for SIMD=false */                                 \
  template KOKKOS_FUNCTION void                                                \
  specfem::boundary_conditions::impl_apply_boundary_conditions<                \
      PointBoundaryType<GET_TAG(DIMENSION_TAG), GET_TAG(BOUNDARY_TAG), false>, \
      PointPropertyType<GET_TAG(DIMENSION_TAG), GET_TAG(MEDIUM_TAG),           \
                        GET_TAG(PROPERTY_TAG), false>,                         \
      PointVelocityType<GET_TAG(DIMENSION_TAG), GET_TAG(MEDIUM_TAG), false>,   \
      PointAccelerationType<GET_TAG(DIMENSION_TAG), GET_TAG(MEDIUM_TAG),       \
                            false> >(                                          \
      const boundary_type<GET_TAG(BOUNDARY_TAG)> &,                            \
      const PointBoundaryType<GET_TAG(DIMENSION_TAG), GET_TAG(BOUNDARY_TAG),   \
                              false> &,                                        \
      const PointPropertyType<GET_TAG(DIMENSION_TAG), GET_TAG(MEDIUM_TAG),     \
                              GET_TAG(PROPERTY_TAG), false> &,                 \
      const PointVelocityType<GET_TAG(DIMENSION_TAG), GET_TAG(MEDIUM_TAG),     \
                              false> &,                                        \
      PointAccelerationType<GET_TAG(DIMENSION_TAG), GET_TAG(MEDIUM_TAG),       \
                            false> &);                                         \
                                                                               \
  /** Template instantiation for SIMD=true */                                  \
  template KOKKOS_FUNCTION void                                                \
  specfem::boundary_conditions::impl_apply_boundary_conditions<                \
      PointBoundaryType<GET_TAG(DIMENSION_TAG), GET_TAG(BOUNDARY_TAG), true>,  \
      PointPropertyType<GET_TAG(DIMENSION_TAG), GET_TAG(MEDIUM_TAG),           \
                        GET_TAG(PROPERTY_TAG), true>,                          \
      PointVelocityType<GET_TAG(DIMENSION_TAG), GET_TAG(MEDIUM_TAG), true>,    \
      PointAccelerationType<GET_TAG(DIMENSION_TAG), GET_TAG(MEDIUM_TAG),       \
                            true> >(                                           \
      const boundary_type<GET_TAG(BOUNDARY_TAG)> &,                            \
      const PointBoundaryType<GET_TAG(DIMENSION_TAG), GET_TAG(BOUNDARY_TAG),   \
                              true> &,                                         \
      const PointPropertyType<GET_TAG(DIMENSION_TAG), GET_TAG(MEDIUM_TAG),     \
                              GET_TAG(PROPERTY_TAG), true> &,                  \
      const PointVelocityType<GET_TAG(DIMENSION_TAG), GET_TAG(MEDIUM_TAG),     \
                              true> &,                                         \
      PointAccelerationType<GET_TAG(DIMENSION_TAG), GET_TAG(MEDIUM_TAG), true> \
          &);

CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
    INSTANTIATION_MACRO,
    WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC)
        WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC)
            WHERE(BOUNDARY_TAG_STACEY, BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef INSTANTIATION_MACRO
