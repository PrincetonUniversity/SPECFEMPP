#include "domain/impl/elements/elastic/elastic2d.hpp"
#include "domain/impl/elements/elastic/elastic2d.tpp"

// Explicit function template instantiation

template specfem::point::stress_integrand<specfem::dimension::type::dim2,
                                          specfem::element::medium_tag::elastic,
                                          false>
specfem::domain::impl::elements::impl_compute_stress_integrands(
    const specfem::point::partial_derivatives<specfem::dimension::type::dim2,
                                              false, false> &,
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic, false> &,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        false> &);

template specfem::point::stress_integrand<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic, true>
specfem::domain::impl::elements::impl_compute_stress_integrands(
    const specfem::point::partial_derivatives<specfem::dimension::type::dim2,
                                              false, true> &,
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic, true> &,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        true> &);

template specfem::point::field<specfem::dimension::type::dim2,
                               specfem::element::medium_tag::elastic, false,
                               false, false, true, false>
specfem::domain::impl::elements::impl_mass_matrix_component(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic, false> &,
    const specfem::point::partial_derivatives<specfem::dimension::type::dim2,
                                              true, false> &);

template specfem::point::field<specfem::dimension::type::dim2,
                               specfem::element::medium_tag::elastic, false,
                               false, false, true, true>
specfem::domain::impl::elements::impl_mass_matrix_component(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic, true> &,
    const specfem::point::partial_derivatives<specfem::dimension::type::dim2,
                                              true, true> &);

template specfem::point::field<specfem::dimension::type::dim2,
                               specfem::element::medium_tag::elastic, false,
                               false, false, true, false>
specfem::domain::impl::elements::impl_mass_matrix_component(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::anisotropic, false> &,
    const specfem::point::partial_derivatives<specfem::dimension::type::dim2,
                                              true, false> &);

template specfem::point::field<specfem::dimension::type::dim2,
                               specfem::element::medium_tag::elastic, false,
                               false, false, true, true>
specfem::domain::impl::elements::impl_mass_matrix_component(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::anisotropic, true> &,
    const specfem::point::partial_derivatives<specfem::dimension::type::dim2,
                                              true, true> &);
