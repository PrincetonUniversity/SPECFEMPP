#include "domain/impl/elements/element.hpp"
// #include "domain/impl/elements/acoustic/acoustic2d.tpp"
// #include "domain/impl/elements/elastic/elastic2d.tpp"
#include "enumerations/medium.hpp"
#include "point/field_derivatives.hpp"
#include "point/partial_derivatives.hpp"
#include "point/properties.hpp"
#include "point/stress_integrand.hpp"
#include "specfem_setup.hpp"

// Explicit function template instantiation

template specfem::point::stress_integrand<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    false>
specfem::domain::impl::elements::compute_stress_integrands<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::property_tag::isotropic, false>(
    const specfem::point::partial_derivatives2<false, false> &,
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, false> &,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        false> &);

template specfem::point::stress_integrand<specfem::dimension::type::dim2,
                                          specfem::element::medium_tag::elastic,
                                          false>
specfem::domain::impl::elements::compute_stress_integrands<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::property_tag::isotropic, false>(
    const specfem::point::partial_derivatives2<false, false> &,
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic, false> &,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        false> &);

template specfem::point::field<specfem::dimension::type::dim2,
                               specfem::element::medium_tag::acoustic, false,
                               false, false, true, false>
specfem::domain::impl::elements::mass_matrix_component<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::property_tag::isotropic, false>(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, false> &,
    const specfem::point::partial_derivatives2<false, true> &);

template specfem::point::field<specfem::dimension::type::dim2,
                               specfem::element::medium_tag::elastic, false,
                               false, false, true, false>
specfem::domain::impl::elements::mass_matrix_component<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::property_tag::isotropic, false>(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic, false> &,
    const specfem::point::partial_derivatives2<false, true> &);

template specfem::point::stress_integrand<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    true>
specfem::domain::impl::elements::compute_stress_integrands<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::property_tag::isotropic, true>(
    const specfem::point::partial_derivatives2<true, false> &,
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, true> &,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        true> &);

template specfem::point::stress_integrand<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic, true>
specfem::domain::impl::elements::compute_stress_integrands<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::property_tag::isotropic, true>(
    const specfem::point::partial_derivatives2<true, false> &,
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic, true> &,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        true> &);

template specfem::point::field<specfem::dimension::type::dim2,
                               specfem::element::medium_tag::acoustic, false,
                               false, false, true, true>
specfem::domain::impl::elements::mass_matrix_component<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::property_tag::isotropic, true>(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, true> &,
    const specfem::point::partial_derivatives2<true, true> &);

template specfem::point::field<specfem::dimension::type::dim2,
                               specfem::element::medium_tag::elastic, false,
                               false, false, true, true>
specfem::domain::impl::elements::mass_matrix_component<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::property_tag::isotropic, true>(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic, true> &,
    const specfem::point::partial_derivatives2<true, true> &);
