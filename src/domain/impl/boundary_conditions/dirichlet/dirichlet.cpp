#include "domain/impl/boundary_conditions/dirichlet/dirichlet.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
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

template <bool UseSIMD, specfem::element::boundary_tag BoundaryTag>
using PointBoundaryType = specfem::point::boundary<UseSIMD, BoundaryTag>;
} // namespace

template void
specfem::domain::impl::boundary_conditions::impl_apply_boundary_conditions<
    PointBoundaryType<false,
                      specfem::element::boundary_tag::acoustic_free_surface>,
    PointVelocityType<false, specfem::dimension::type::dim2,
                      specfem::element::medium_tag::acoustic>,
    PointAccelerationType<false, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic> >(
    const specfem::domain::impl::boundary_conditions::acoustic_free_surface_type
        &,
    const PointBoundaryType<
        false, specfem::element::boundary_tag::acoustic_free_surface> &,
    const PointVelocityType<false, specfem::dimension::type::dim2,
                            specfem::element::medium_tag::acoustic> &,
    PointAccelerationType<false, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic> &);

template void
specfem::domain::impl::boundary_conditions::impl_apply_boundary_conditions<
    PointBoundaryType<true,
                      specfem::element::boundary_tag::acoustic_free_surface>,
    PointVelocityType<true, specfem::dimension::type::dim2,
                      specfem::element::medium_tag::acoustic>,
    PointAccelerationType<true, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic> >(
    const specfem::domain::impl::boundary_conditions::acoustic_free_surface_type
        &,
    const PointBoundaryType<
        true, specfem::element::boundary_tag::acoustic_free_surface> &,
    const PointVelocityType<true, specfem::dimension::type::dim2,
                            specfem::element::medium_tag::acoustic> &,
    PointAccelerationType<true, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic> &);

// Elastic Acoustic Free Surface Boundary Conditions
template void
specfem::domain::impl::boundary_conditions::impl_apply_boundary_conditions<
    PointBoundaryType<false,
                      specfem::element::boundary_tag::acoustic_free_surface>,
    PointVelocityType<false, specfem::dimension::type::dim2,
                      specfem::element::medium_tag::elastic>,
    PointAccelerationType<false, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic> >(
    const specfem::domain::impl::boundary_conditions::acoustic_free_surface_type
        &,
    const PointBoundaryType<
        false, specfem::element::boundary_tag::acoustic_free_surface> &,
    const PointVelocityType<false, specfem::dimension::type::dim2,
                            specfem::element::medium_tag::elastic> &,
    PointAccelerationType<false, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic> &);

template void
specfem::domain::impl::boundary_conditions::impl_apply_boundary_conditions<
    PointBoundaryType<true,
                      specfem::element::boundary_tag::acoustic_free_surface>,
    PointVelocityType<true, specfem::dimension::type::dim2,
                      specfem::element::medium_tag::elastic>,
    PointAccelerationType<true, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic> >(
    const specfem::domain::impl::boundary_conditions::acoustic_free_surface_type
        &,
    const PointBoundaryType<
        true, specfem::element::boundary_tag::acoustic_free_surface> &,
    const PointVelocityType<true, specfem::dimension::type::dim2,
                            specfem::element::medium_tag::elastic> &,
    PointAccelerationType<true, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic> &);
