#include "domain/impl/boundary_conditions/composite_stacey_dirichlet/composite_stacey_dirichlet.hpp"
#include "domain/impl/boundary_conditions/composite_stacey_dirichlet/composite_stacey_dirichlet.tpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "point/boundary.hpp"
#include "point/field.hpp"
#include "point/properties.hpp"

namespace {
template <specfem::element::boundary_tag BoundaryTag, bool UseSIMD>
using PointBoundaryType = specfem::point::boundary<BoundaryTag, UseSIMD>;

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD>
using PointPropertyType =
    specfem::point::properties<specfem::dimension::type::dim2, MediumTag,
                               PropertyTag, UseSIMD>;

template <specfem::element::medium_tag MediumTag, bool UseSIMD>
using PointMassMatrixType =
    specfem::point::field<specfem::dimension::type::dim2, MediumTag, false,
                          false, false, true, UseSIMD>;
} // namespace

template void
specfem::domain::impl::boundary_conditions::impl_compute_mass_matrix_terms<
    PointBoundaryType<
        specfem::element::boundary_tag::composite_stacey_dirichlet, false>,
    PointPropertyType<specfem::element::medium_tag::acoustic,
                      specfem::element::property_tag::isotropic, false>,
    PointMassMatrixType<specfem::element::medium_tag::acoustic, false> >(
    const specfem::domain::impl::boundary_conditions::
        composite_stacey_dirichlet_type &,
    const type_real dt,
    const PointBoundaryType<
        specfem::element::boundary_tag::composite_stacey_dirichlet, false> &,
    const PointPropertyType<specfem::element::medium_tag::acoustic,
                            specfem::element::property_tag::isotropic, false> &,
    PointMassMatrixType<specfem::element::medium_tag::acoustic, false> &);

template void
specfem::domain::impl::boundary_conditions::impl_compute_mass_matrix_terms<
    PointBoundaryType<
        specfem::element::boundary_tag::composite_stacey_dirichlet, true>,
    PointPropertyType<specfem::element::medium_tag::acoustic,
                      specfem::element::property_tag::isotropic, true>,
    PointMassMatrixType<specfem::element::medium_tag::acoustic, true> >(
    const specfem::domain::impl::boundary_conditions::
        composite_stacey_dirichlet_type &,
    const type_real dt,
    const PointBoundaryType<
        specfem::element::boundary_tag::composite_stacey_dirichlet, true> &,
    const PointPropertyType<specfem::element::medium_tag::acoustic,
                            specfem::element::property_tag::isotropic, true> &,
    PointMassMatrixType<specfem::element::medium_tag::acoustic, true> &);

// Elastic
template void
specfem::domain::impl::boundary_conditions::impl_compute_mass_matrix_terms<
    PointBoundaryType<
        specfem::element::boundary_tag::composite_stacey_dirichlet, false>,
    PointPropertyType<specfem::element::medium_tag::elastic,
                      specfem::element::property_tag::isotropic, false>,
    PointMassMatrixType<specfem::element::medium_tag::elastic, false> >(
    const specfem::domain::impl::boundary_conditions::
        composite_stacey_dirichlet_type &,
    const type_real dt,
    const PointBoundaryType<
        specfem::element::boundary_tag::composite_stacey_dirichlet, false> &,
    const PointPropertyType<specfem::element::medium_tag::elastic,
                            specfem::element::property_tag::isotropic, false> &,
    PointMassMatrixType<specfem::element::medium_tag::elastic, false> &);

template void
specfem::domain::impl::boundary_conditions::impl_compute_mass_matrix_terms<
    PointBoundaryType<
        specfem::element::boundary_tag::composite_stacey_dirichlet, true>,
    PointPropertyType<specfem::element::medium_tag::elastic,
                      specfem::element::property_tag::isotropic, true>,
    PointMassMatrixType<specfem::element::medium_tag::elastic, true> >(
    const specfem::domain::impl::boundary_conditions::
        composite_stacey_dirichlet_type &,
    const type_real dt,
    const PointBoundaryType<
        specfem::element::boundary_tag::composite_stacey_dirichlet, true> &,
    const PointPropertyType<specfem::element::medium_tag::elastic,
                            specfem::element::property_tag::isotropic, true> &,
    PointMassMatrixType<specfem::element::medium_tag::elastic, true> &);
