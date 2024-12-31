#include "medium/compute_mass_matrix.hpp"

// This file only contains explicit template instantiations

// dim2, elastic, isotropic, false
template specfem::point::field<specfem::dimension::type::dim2,
                               specfem::element::medium_tag::elastic, false,
                               false, false, true, false>
specfem::medium::impl_mass_matrix_component(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic, false> &,
    const specfem::point::partial_derivatives<specfem::dimension::type::dim2,
                                              true, false> &);

// dim2, elastic, isotropic, true
template specfem::point::field<specfem::dimension::type::dim2,
                               specfem::element::medium_tag::elastic, false,
                               false, false, true, true>
specfem::medium::impl_mass_matrix_component(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic, true> &,
    const specfem::point::partial_derivatives<specfem::dimension::type::dim2,
                                              true, true> &);

// dim2, elastic, anisotropic, false
template specfem::point::field<specfem::dimension::type::dim2,
                               specfem::element::medium_tag::elastic, false,
                               false, false, true, false>
specfem::medium::impl_mass_matrix_component(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::anisotropic, false> &,
    const specfem::point::partial_derivatives<specfem::dimension::type::dim2,
                                              true, false> &);

// dim2, elastic, anisotropic, true
template specfem::point::field<specfem::dimension::type::dim2,
                               specfem::element::medium_tag::elastic, false,
                               false, false, true, true>
specfem::medium::impl_mass_matrix_component(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::anisotropic, true> &,
    const specfem::point::partial_derivatives<specfem::dimension::type::dim2,
                                              true, true> &);
