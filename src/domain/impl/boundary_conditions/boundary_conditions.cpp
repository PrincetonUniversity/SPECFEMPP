#include "domain/impl/boundary_conditions/boundary_conditions.hpp"
#include "domain/impl/boundary_conditions/composite_stacey_dirichlet/composite_stacey_dirichlet.hpp"
#include "domain/impl/boundary_conditions/dirichlet/dirichlet.hpp"
#include "domain/impl/boundary_conditions/none/none.hpp"
#include "domain/impl/boundary_conditions/stacey/stacey.hpp"
#include "point/boundary.hpp"
#include "point/field.hpp"

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
    specfem::point::field<specfem::dimension::type::dim2, MediumTag, true,
                          false, false, true, UseSIMD>;

template <bool UseSIMD, specfem::element::boundary_tag BoundaryTag>
using PointBoundaryType =
    specfem::point::boundary<BoundaryTag, specfem::dimension::type::dim2,
                             UseSIMD>;
} // namespace

// None boundary conditions
// -----------------------------------------------------------------------------
template void
specfem::domain::impl::boundary_conditions::apply_boundary_conditions<
    PointBoundaryType<false, specfem::element::boundary_tag::none>,
    PointPropertyType<specfem::element::medium_tag::acoustic,
                      specfem::element::property_tag::isotropic, false>,
    PointVelocityType<false, specfem::dimension::type::dim2,
                      specfem::element::medium_tag::acoustic>,
    PointAccelerationType<false, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic> >(
    const PointBoundaryType<false, specfem::element::boundary_tag::none> &,
    const PointPropertyType<specfem::element::medium_tag::acoustic,
                            specfem::element::property_tag::isotropic, false> &,
    const PointVelocityType<false, specfem::dimension::type::dim2,
                            specfem::element::medium_tag::acoustic> &,
    PointAccelerationType<false, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic> &);

template void
specfem::domain::impl::boundary_conditions::apply_boundary_conditions<
    PointBoundaryType<true, specfem::element::boundary_tag::none>,
    PointPropertyType<specfem::element::medium_tag::acoustic,
                      specfem::element::property_tag::isotropic, true>,
    PointVelocityType<true, specfem::dimension::type::dim2,
                      specfem::element::medium_tag::acoustic>,
    PointAccelerationType<true, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic> >(
    const PointBoundaryType<true, specfem::element::boundary_tag::none> &,
    const PointPropertyType<specfem::element::medium_tag::acoustic,
                            specfem::element::property_tag::isotropic, true> &,
    const PointVelocityType<true, specfem::dimension::type::dim2,
                            specfem::element::medium_tag::acoustic> &,
    PointAccelerationType<true, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic> &);

template void
specfem::domain::impl::boundary_conditions::apply_boundary_conditions<
    PointBoundaryType<false, specfem::element::boundary_tag::none>,
    PointPropertyType<specfem::element::medium_tag::elastic,
                      specfem::element::property_tag::isotropic, false>,
    PointVelocityType<false, specfem::dimension::type::dim2,
                      specfem::element::medium_tag::elastic>,
    PointAccelerationType<false, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic> >(
    const PointBoundaryType<false, specfem::element::boundary_tag::none> &,
    const PointPropertyType<specfem::element::medium_tag::elastic,
                            specfem::element::property_tag::isotropic, false> &,
    const PointVelocityType<false, specfem::dimension::type::dim2,
                            specfem::element::medium_tag::elastic> &,
    PointAccelerationType<false, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic> &);

template void
specfem::domain::impl::boundary_conditions::apply_boundary_conditions<
    PointBoundaryType<true, specfem::element::boundary_tag::none>,
    PointPropertyType<specfem::element::medium_tag::elastic,
                      specfem::element::property_tag::isotropic, true>,
    PointVelocityType<true, specfem::dimension::type::dim2,
                      specfem::element::medium_tag::elastic>,
    PointAccelerationType<true, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic> >(
    const PointBoundaryType<true, specfem::element::boundary_tag::none> &,
    const PointPropertyType<specfem::element::medium_tag::elastic,
                            specfem::element::property_tag::isotropic, true> &,
    const PointVelocityType<true, specfem::dimension::type::dim2,
                            specfem::element::medium_tag::elastic> &,
    PointAccelerationType<true, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic> &);
// -----------------------------------------------------------------------------

// Acoustic free surface boundary conditions
// -----------------------------------------------------------------------------
template void
specfem::domain::impl::boundary_conditions::apply_boundary_conditions<
    PointBoundaryType<false,
                      specfem::element::boundary_tag::acoustic_free_surface>,
    PointPropertyType<specfem::element::medium_tag::acoustic,
                      specfem::element::property_tag::isotropic, false>,
    PointVelocityType<false, specfem::dimension::type::dim2,
                      specfem::element::medium_tag::acoustic>,
    PointAccelerationType<false, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic> >(
    const PointBoundaryType<
        false, specfem::element::boundary_tag::acoustic_free_surface> &,
    const PointPropertyType<specfem::element::medium_tag::acoustic,
                            specfem::element::property_tag::isotropic, false> &,
    const PointVelocityType<false, specfem::dimension::type::dim2,
                            specfem::element::medium_tag::acoustic> &,
    PointAccelerationType<false, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic> &);

template void
specfem::domain::impl::boundary_conditions::apply_boundary_conditions<
    PointBoundaryType<true,
                      specfem::element::boundary_tag::acoustic_free_surface>,
    PointPropertyType<specfem::element::medium_tag::acoustic,
                      specfem::element::property_tag::isotropic, true>,
    PointVelocityType<true, specfem::dimension::type::dim2,
                      specfem::element::medium_tag::acoustic>,
    PointAccelerationType<true, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic> >(
    const PointBoundaryType<
        true, specfem::element::boundary_tag::acoustic_free_surface> &,
    const PointPropertyType<specfem::element::medium_tag::acoustic,
                            specfem::element::property_tag::isotropic, true> &,
    const PointVelocityType<true, specfem::dimension::type::dim2,
                            specfem::element::medium_tag::acoustic> &,
    PointAccelerationType<true, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic> &);
// -----------------------------------------------------------------------------

// Stacey boundary conditions
// -----------------------------------------------------------------------------
template void
specfem::domain::impl::boundary_conditions::apply_boundary_conditions<
    PointBoundaryType<false, specfem::element::boundary_tag::stacey>,
    PointPropertyType<specfem::element::medium_tag::acoustic,
                      specfem::element::property_tag::isotropic, false>,
    PointVelocityType<false, specfem::dimension::type::dim2,
                      specfem::element::medium_tag::acoustic>,
    PointAccelerationType<false, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic> >(
    const PointBoundaryType<false, specfem::element::boundary_tag::stacey> &,
    const PointPropertyType<specfem::element::medium_tag::acoustic,
                            specfem::element::property_tag::isotropic, false> &,
    const PointVelocityType<false, specfem::dimension::type::dim2,
                            specfem::element::medium_tag::acoustic> &,
    PointAccelerationType<false, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic> &);

template void
specfem::domain::impl::boundary_conditions::apply_boundary_conditions<
    PointBoundaryType<true, specfem::element::boundary_tag::stacey>,
    PointPropertyType<specfem::element::medium_tag::acoustic,
                      specfem::element::property_tag::isotropic, true>,
    PointVelocityType<true, specfem::dimension::type::dim2,
                      specfem::element::medium_tag::acoustic>,
    PointAccelerationType<true, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic> >(
    const PointBoundaryType<true, specfem::element::boundary_tag::stacey> &,
    const PointPropertyType<specfem::element::medium_tag::acoustic,
                            specfem::element::property_tag::isotropic, true> &,
    const PointVelocityType<true, specfem::dimension::type::dim2,
                            specfem::element::medium_tag::acoustic> &,
    PointAccelerationType<true, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic> &);

template void
specfem::domain::impl::boundary_conditions::apply_boundary_conditions<
    PointBoundaryType<false, specfem::element::boundary_tag::stacey>,
    PointPropertyType<specfem::element::medium_tag::elastic,
                      specfem::element::property_tag::isotropic, false>,
    PointVelocityType<false, specfem::dimension::type::dim2,
                      specfem::element::medium_tag::elastic>,
    PointAccelerationType<false, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic> >(
    const PointBoundaryType<false, specfem::element::boundary_tag::stacey> &,
    const PointPropertyType<specfem::element::medium_tag::elastic,
                            specfem::element::property_tag::isotropic, false> &,
    const PointVelocityType<false, specfem::dimension::type::dim2,
                            specfem::element::medium_tag::elastic> &,
    PointAccelerationType<false, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic> &);

template void
specfem::domain::impl::boundary_conditions::apply_boundary_conditions<
    PointBoundaryType<true, specfem::element::boundary_tag::stacey>,
    PointPropertyType<specfem::element::medium_tag::elastic,
                      specfem::element::property_tag::isotropic, true>,
    PointVelocityType<true, specfem::dimension::type::dim2,
                      specfem::element::medium_tag::elastic>,
    PointAccelerationType<true, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic> >(
    const PointBoundaryType<true, specfem::element::boundary_tag::stacey> &,
    const PointPropertyType<specfem::element::medium_tag::elastic,
                            specfem::element::property_tag::isotropic, true> &,
    const PointVelocityType<true, specfem::dimension::type::dim2,
                            specfem::element::medium_tag::elastic> &,
    PointAccelerationType<true, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic> &);

// -----------------------------------------------------------------------------

// Composite Stacey Dirichlet boundary conditions
// -----------------------------------------------------------------------------
template void
specfem::domain::impl::boundary_conditions::apply_boundary_conditions<
    PointBoundaryType<
        false, specfem::element::boundary_tag::composite_stacey_dirichlet>,
    PointPropertyType<specfem::element::medium_tag::acoustic,
                      specfem::element::property_tag::isotropic, false>,
    PointVelocityType<false, specfem::dimension::type::dim2,
                      specfem::element::medium_tag::acoustic>,
    PointAccelerationType<false, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic> >(
    const PointBoundaryType<
        false, specfem::element::boundary_tag::composite_stacey_dirichlet> &,
    const PointPropertyType<specfem::element::medium_tag::acoustic,
                            specfem::element::property_tag::isotropic, false> &,
    const PointVelocityType<false, specfem::dimension::type::dim2,
                            specfem::element::medium_tag::acoustic> &,
    PointAccelerationType<false, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic> &);

template void
specfem::domain::impl::boundary_conditions::apply_boundary_conditions<
    PointBoundaryType<
        true, specfem::element::boundary_tag::composite_stacey_dirichlet>,
    PointPropertyType<specfem::element::medium_tag::acoustic,
                      specfem::element::property_tag::isotropic, true>,
    PointVelocityType<true, specfem::dimension::type::dim2,
                      specfem::element::medium_tag::acoustic>,
    PointAccelerationType<true, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic> >(
    const PointBoundaryType<
        true, specfem::element::boundary_tag::composite_stacey_dirichlet> &,
    const PointPropertyType<specfem::element::medium_tag::acoustic,
                            specfem::element::property_tag::isotropic, true> &,
    const PointVelocityType<true, specfem::dimension::type::dim2,
                            specfem::element::medium_tag::acoustic> &,
    PointAccelerationType<true, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic> &);

// -----------------------------------------------------------------------------
