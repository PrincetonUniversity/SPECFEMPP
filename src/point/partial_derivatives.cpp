
#include "point/partial_derivatives.hpp"
#include "point/partial_derivatives.tpp"
// Explicit template instantiation

template struct specfem::point::partial_derivatives<
    specfem::dimension::type::dim2, false, false>;
template struct specfem::point::partial_derivatives<
    specfem::dimension::type::dim2, true, false>;
template struct specfem::point::partial_derivatives<
    specfem::dimension::type::dim2, false, true>;
template struct specfem::point::partial_derivatives<
    specfem::dimension::type::dim2, true, true>;
