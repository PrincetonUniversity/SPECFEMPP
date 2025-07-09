#include "boundary_values.hpp"
#include "boundary_values/boundary_values.tpp"
#include "boundary_values/dim2/impl/boundary_medium_container.tpp"
#include "boundary_values/dim2/impl/boundary_value_container.tpp"

template class specfem::assembly::boundary_values<
    specfem::dimension::type::dim2>;
