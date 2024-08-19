#include "frechet_derivatives/impl/frechet_element.hpp"
#include "frechet_derivatives/impl/frechet_element.tpp"

// Explicit template instantiation
template class specfem::frechet_derivatives::impl::frechet_elements<
    5, specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::property_tag::isotropic>;

template class specfem::frechet_derivatives::impl::frechet_elements<
    5, specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::property_tag::isotropic>;

template class specfem::frechet_derivatives::impl::frechet_elements<
    8, specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::property_tag::isotropic>;

template class specfem::frechet_derivatives::impl::frechet_elements<
    8, specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::property_tag::isotropic>;
