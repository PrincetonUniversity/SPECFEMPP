#include "specfem/assembly/fields.hpp"
#include "enumerations/interface.hpp"
#include "fields.tpp"

template class specfem::assembly::fields<specfem::dimension::type::dim2>;
template class specfem::assembly::fields<specfem::dimension::type::dim3>;
