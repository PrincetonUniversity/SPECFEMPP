#include "medium/compute_stress_integrand.hpp"

// This file only contains explicit template instantiations

// dim2, elastic, isotropic, false
template specfem::point::stress_integrand<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_sv,
    false>
specfem::medium::impl_compute_stress_integrands(
    const specfem::point::partial_derivatives<specfem::dimension::type::dim2,
                                              false, false> &,
    const specfem::point::properties<specfem::dimension::type::dim2,
                                     specfem::element::medium_tag::elastic_sv,
                                     specfem::element::property_tag::isotropic,
                                     false> &,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2,
        specfem::element::medium_tag::elastic_sv, false> &);

// dim2, elastic, isotropic, true
template specfem::point::stress_integrand<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_sv,
    true>
specfem::medium::impl_compute_stress_integrands(
    const specfem::point::partial_derivatives<specfem::dimension::type::dim2,
                                              false, true> &,
    const specfem::point::properties<specfem::dimension::type::dim2,
                                     specfem::element::medium_tag::elastic_sv,
                                     specfem::element::property_tag::isotropic,
                                     true> &,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2,
        specfem::element::medium_tag::elastic_sv, true> &);
