#include "frechet_derivatives/frechet_derivatives.hpp"

// Explicit template instantiation
// Elastic isotropic, ngll 5
template class specfem::frechet_derivatives::frechet_derivatives<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic, 5>;

template class specfem::frechet_derivatives::frechet_derivatives<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic, 5>;

template class specfem::frechet_derivatives::frechet_derivatives<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic, 8>;

template class specfem::frechet_derivatives::frechet_derivatives<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic, 8>;
