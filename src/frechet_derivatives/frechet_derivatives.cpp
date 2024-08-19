#include "frechet_derivatives/frechet_derivatives.hpp"

// Explicit template instantiation
template class specfem::frechet_derivatives::frechet_derivatives<
    5, specfem::dimension::type::dim2, specfem::element::medium_tag::elastic>;

template class specfem::frechet_derivatives::frechet_derivatives<
    5, specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic>;

template class specfem::frechet_derivatives::frechet_derivatives<
    8, specfem::dimension::type::dim2, specfem::element::medium_tag::elastic>;

template class specfem::frechet_derivatives::frechet_derivatives<
    8, specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic>;
