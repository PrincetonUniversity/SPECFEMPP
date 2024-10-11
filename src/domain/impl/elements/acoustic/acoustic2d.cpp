#include "domain/impl/elements/acoustic/acoustic2d.hpp"
#include "domain/impl/elements/acoustic/acoustic2d.tpp"

// Explicit function template instantiation

template specfem::point::stress_integrand<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    false>
specfem::domain::impl::elements::impl_compute_stress_integrands(
    const specfem::point::partial_derivatives<specfem::dimension::type::dim2,
                                              false, false> &,
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, false> &,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        false> &);

template specfem::point::stress_integrand<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    true>
specfem::domain::impl::elements::impl_compute_stress_integrands(
    const specfem::point::partial_derivatives<specfem::dimension::type::dim2,
                                              false, true> &,
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, true> &,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        true> &);

template specfem::point::field<specfem::dimension::type::dim2,
                               specfem::element::medium_tag::acoustic, false,
                               false, false, true, false>
specfem::domain::impl::elements::impl_mass_matrix_component(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, false> &,
    const specfem::point::partial_derivatives<specfem::dimension::type::dim2,
                                              true, false> &);

template specfem::point::field<specfem::dimension::type::dim2,
                               specfem::element::medium_tag::acoustic, false,
                               false, false, true, true>
specfem::domain::impl::elements::impl_mass_matrix_component(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, true> &,
    const specfem::point::partial_derivatives<specfem::dimension::type::dim2,
                                              true, true> &);
