#include "specfem/assembly/fields.hpp"
#include "fields.tpp"
#include "specfem/enums.hpp"

template class specfem::assembly::fields<specfem::element::dimension_tag::dim2>;
template class specfem::assembly::fields<specfem::element::dimension_tag::dim3>;
