#include "domain/impl/boundary_conditions/stacey/stacey.hpp"
#include "domain/impl/boundary_conditions/stacey/stacey.tpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "point/boundary.hpp"
#include "point/field.hpp"
#include "point/properties.hpp"

namespace {
template <bool UseSIMD, specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag>
using PointVelocityType = specfem::point::field<DimensionType, MediumTag, false,
                                                true, false, false, UseSIMD>;

template <bool UseSIMD, specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag>
using PointAccelerationType =
    specfem::point::field<DimensionType, MediumTag, false, false, true, false,
                          UseSIMD>;

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD>
using PointPropertyType =
    specfem::point::properties<specfem::dimension::type::dim2, MediumTag,
                               PropertyTag, UseSIMD>;

template <bool UseSIMD, specfem::element::boundary_tag BoundaryTag>
using PointBoundaryType =
    specfem::point::boundary<BoundaryTag, specfem::dimension::type::dim2,
                             UseSIMD>;
} // namespace

template KOKKOS_FUNCTION void
specfem::domain::impl::boundary_conditions::impl_apply_boundary_conditions<
    PointBoundaryType<false, specfem::element::boundary_tag::stacey>,
    PointPropertyType<specfem::element::medium_tag::acoustic,
                      specfem::element::property_tag::isotropic, false>,
    PointVelocityType<false, specfem::dimension::type::dim2,
                      specfem::element::medium_tag::acoustic>,
    PointAccelerationType<false, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic> >(
    const specfem::domain::impl::boundary_conditions::stacey_type &,
    const PointBoundaryType<false, specfem::element::boundary_tag::stacey> &,
    const PointPropertyType<specfem::element::medium_tag::acoustic,
                            specfem::element::property_tag::isotropic, false> &,
    const PointVelocityType<false, specfem::dimension::type::dim2,
                            specfem::element::medium_tag::acoustic> &,
    PointAccelerationType<false, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic> &);

template KOKKOS_FUNCTION void
specfem::domain::impl::boundary_conditions::impl_apply_boundary_conditions<
    PointBoundaryType<true, specfem::element::boundary_tag::stacey>,
    PointPropertyType<specfem::element::medium_tag::acoustic,
                      specfem::element::property_tag::isotropic, true>,
    PointVelocityType<true, specfem::dimension::type::dim2,
                      specfem::element::medium_tag::acoustic>,
    PointAccelerationType<true, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic> >(
    const specfem::domain::impl::boundary_conditions::stacey_type &,
    const PointBoundaryType<true, specfem::element::boundary_tag::stacey> &,
    const PointPropertyType<specfem::element::medium_tag::acoustic,
                            specfem::element::property_tag::isotropic, true> &,
    const PointVelocityType<true, specfem::dimension::type::dim2,
                            specfem::element::medium_tag::acoustic> &,
    PointAccelerationType<true, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic> &);

// Elastic Stacey Boundary Conditions
template KOKKOS_FUNCTION void
specfem::domain::impl::boundary_conditions::impl_apply_boundary_conditions<
    PointBoundaryType<false, specfem::element::boundary_tag::stacey>,
    PointPropertyType<specfem::element::medium_tag::elastic,
                      specfem::element::property_tag::isotropic, false>,
    PointVelocityType<false, specfem::dimension::type::dim2,
                      specfem::element::medium_tag::elastic>,
    PointAccelerationType<false, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic> >(
    const specfem::domain::impl::boundary_conditions::stacey_type &,
    const PointBoundaryType<false, specfem::element::boundary_tag::stacey> &,
    const PointPropertyType<specfem::element::medium_tag::elastic,
                            specfem::element::property_tag::isotropic, false> &,
    const PointVelocityType<false, specfem::dimension::type::dim2,
                            specfem::element::medium_tag::elastic> &,
    PointAccelerationType<false, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic> &);

template KOKKOS_FUNCTION void
specfem::domain::impl::boundary_conditions::impl_apply_boundary_conditions<
    PointBoundaryType<true, specfem::element::boundary_tag::stacey>,
    PointPropertyType<specfem::element::medium_tag::elastic,
                      specfem::element::property_tag::isotropic, true>,
    PointVelocityType<true, specfem::dimension::type::dim2,
                      specfem::element::medium_tag::elastic>,
    PointAccelerationType<true, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic> >(
    const specfem::domain::impl::boundary_conditions::stacey_type &,
    const PointBoundaryType<true, specfem::element::boundary_tag::stacey> &,
    const PointPropertyType<specfem::element::medium_tag::elastic,
                            specfem::element::property_tag::isotropic, true> &,
    const PointVelocityType<true, specfem::dimension::type::dim2,
                            specfem::element::medium_tag::elastic> &,
    PointAccelerationType<true, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic> &);

template KOKKOS_FUNCTION void
specfem::domain::impl::boundary_conditions::impl_apply_boundary_conditions<
    PointBoundaryType<false, specfem::element::boundary_tag::stacey>,
    PointPropertyType<specfem::element::medium_tag::elastic,
                      specfem::element::property_tag::anisotropic, false>,
    PointVelocityType<false, specfem::dimension::type::dim2,
                      specfem::element::medium_tag::elastic>,
    PointAccelerationType<false, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic> >(
    const specfem::domain::impl::boundary_conditions::stacey_type &,
    const PointBoundaryType<false, specfem::element::boundary_tag::stacey> &,
    const PointPropertyType<specfem::element::medium_tag::elastic,
                            specfem::element::property_tag::anisotropic, false>
        &,
    const PointVelocityType<false, specfem::dimension::type::dim2,
                            specfem::element::medium_tag::elastic> &,
    PointAccelerationType<false, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic> &);

template KOKKOS_FUNCTION void
specfem::domain::impl::boundary_conditions::impl_apply_boundary_conditions<
    PointBoundaryType<true, specfem::element::boundary_tag::stacey>,
    PointPropertyType<specfem::element::medium_tag::elastic,
                      specfem::element::property_tag::anisotropic, true>,
    PointVelocityType<true, specfem::dimension::type::dim2,
                      specfem::element::medium_tag::elastic>,
    PointAccelerationType<true, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic> >(
    const specfem::domain::impl::boundary_conditions::stacey_type &,
    const PointBoundaryType<true, specfem::element::boundary_tag::stacey> &,
    const PointPropertyType<specfem::element::medium_tag::elastic,
                            specfem::element::property_tag::anisotropic, true>
        &,
    const PointVelocityType<true, specfem::dimension::type::dim2,
                            specfem::element::medium_tag::elastic> &,
    PointAccelerationType<true, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic> &);
