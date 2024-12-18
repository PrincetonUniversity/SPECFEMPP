#include "domain/impl/boundary_conditions/composite_stacey_dirichlet/composite_stacey_dirichlet.hpp"
#include "domain/impl/boundary_conditions/composite_stacey_dirichlet/composite_stacey_dirichlet.tpp"
#include "domain/impl/boundary_conditions/stacey/stacey.hpp"
#include "domain/impl/boundary_conditions/stacey/stacey.tpp"
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
  template KOKKOS_FUNCTION void                                                \
  specfem::domain::impl::boundary_conditions::impl_compute_mass_matrix_terms<  \
      PointBoundaryType<DIMENSION_TAG, BOUNDARY_TAG, false>,                   \
      PointPropertyType<DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, false>,       \
      PointMassMatrixType<DIMENSION_TAG, MEDIUM_TAG, false> >(                 \
      const boundary_type<BOUNDARY_TAG> &, const type_real dt,                 \
      const PointBoundaryType<DIMENSION_TAG, BOUNDARY_TAG, false> &,           \
      const PointPropertyType<DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, false>  \
          &,                                                                   \
      PointMassMatrixType<DIMENSION_TAG, MEDIUM_TAG, false> &);                \
                                                                               \
  template KOKKOS_FUNCTION void                                                \
  specfem::domain::impl::boundary_conditions::impl_compute_mass_matrix_terms<  \
      PointBoundaryType<DIMENSION_TAG, BOUNDARY_TAG, true>,                    \
      PointPropertyType<DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, true>,        \
      PointMassMatrixType<DIMENSION_TAG, MEDIUM_TAG, true> >(                  \
      const boundary_type<BOUNDARY_TAG> &, const type_real dt,                 \
      const PointBoundaryType<DIMENSION_TAG, BOUNDARY_TAG, true> &,            \
      const PointPropertyType<DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, true>   \
          &,                                                                   \
      PointMassMatrixType<DIMENSION_TAG, MEDIUM_TAG, true> &);

CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
    INSTANTIATION_MACRO,
    WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC)
        WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC)
            WHERE(BOUNDARY_TAG_STACEY, BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef INSTANTIATION_MACRO

#define INSTANTIATION_MACRO(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG,           \
                            BOUNDARY_TAG)                                      \
  template KOKKOS_FUNCTION void                                                \
  specfem::domain::impl::boundary_conditions::impl_apply_boundary_conditions<  \
      PointBoundaryType<DIMENSION_TAG, BOUNDARY_TAG, false>,                   \
      PointPropertyType<DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, false>,       \
      PointVelocityType<DIMENSION_TAG, MEDIUM_TAG, false>,                     \
      PointAccelerationType<DIMENSION_TAG, MEDIUM_TAG, false> >(               \
      const boundary_type<BOUNDARY_TAG> &,                                     \
      const PointBoundaryType<DIMENSION_TAG, BOUNDARY_TAG, false> &,           \
      const PointPropertyType<DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, false>  \
          &,                                                                   \
      const PointVelocityType<DIMENSION_TAG, MEDIUM_TAG, false> &,             \
      PointAccelerationType<DIMENSION_TAG, MEDIUM_TAG, false> &);              \
                                                                               \
  template KOKKOS_FUNCTION void                                                \
  specfem::domain::impl::boundary_conditions::impl_apply_boundary_conditions<  \
      PointBoundaryType<DIMENSION_TAG, BOUNDARY_TAG, true>,                    \
      PointPropertyType<DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, true>,        \
      PointVelocityType<DIMENSION_TAG, MEDIUM_TAG, true>,                      \
      PointAccelerationType<DIMENSION_TAG, MEDIUM_TAG, true> >(                \
      const boundary_type<BOUNDARY_TAG> &,                                     \
      const PointBoundaryType<DIMENSION_TAG, BOUNDARY_TAG, true> &,            \
      const PointPropertyType<DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, true>   \
          &,                                                                   \
      const PointVelocityType<DIMENSION_TAG, MEDIUM_TAG, true> &,              \
      PointAccelerationType<DIMENSION_TAG, MEDIUM_TAG, true> &);

CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
    INSTANTIATION_MACRO,
    WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC)
        WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC)
            WHERE(BOUNDARY_TAG_STACEY, BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef INSTANTIATION_MACRO
