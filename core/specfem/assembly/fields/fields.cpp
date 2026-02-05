#include "specfem/assembly/fields.hpp"
#include "enumerations/interface_tags.hpp"
#include "fields.tpp"

template class specfem::assembly::fields<specfem::element::dimension_tag::dim2>;
template class specfem::assembly::fields<specfem::element::dimension_tag::dim3>;
