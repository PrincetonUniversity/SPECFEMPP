#include "frechet_derivatives/impl/frechet_element.hpp"
#include "frechet_derivatives/impl/frechet_element.tpp"

// Explicit template instantiation

// Elastic isotropic, ngll 5
template class specfem::frechet_derivatives::impl::frechet_elements<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::property_tag::isotropic, 5>;

// Elastic anisotropic, ngll 5
template class specfem::frechet_derivatives::impl::frechet_elements<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::property_tag::anisotropic, 5>;

// Acoustic isotropic, ngll 5
template class specfem::frechet_derivatives::impl::frechet_elements<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::property_tag::isotropic, 5>;

// Elastic isotropic, ngll 8
template class specfem::frechet_derivatives::impl::frechet_elements<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::property_tag::isotropic, 8>;

// Elastic anisotropic, ngll 8
template class specfem::frechet_derivatives::impl::frechet_elements<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::property_tag::anisotropic, 8>;

// Acoustic isotropic, ngll 8
template class specfem::frechet_derivatives::impl::frechet_elements<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::property_tag::isotropic, 8>;
