#include "frechet_derivatives/impl/frechet_element.hpp"
#include "frechet_derivatives/impl/frechet_element.tpp"

// Explicit template instantiation
template class specfem::frechet_derivatives::impl::frechet_elements<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::property_tag::isotropic, 5>;

template class specfem::frechet_derivatives::impl::frechet_elements<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::property_tag::isotropic, 5>;

template class specfem::frechet_derivatives::impl::frechet_elements<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::property_tag::isotropic, 8>;

template class specfem::frechet_derivatives::impl::frechet_elements<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::property_tag::isotropic, 8>;
