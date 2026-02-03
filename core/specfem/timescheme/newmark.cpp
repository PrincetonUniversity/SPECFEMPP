#include "specfem/timescheme/newmark.tpp"
#include "specfem/setup.hpp"
#include <ostream>

template class specfem::time_scheme::newmark<
    specfem::assembly::fields<specfem::dimension::type::dim2>,
    specfem::simulation::type::forward>;
template class specfem::time_scheme::newmark<
    specfem::assembly::fields<specfem::dimension::type::dim2>,
    specfem::simulation::type::combined>;

template class specfem::time_scheme::newmark<
    specfem::assembly::fields<specfem::dimension::type::dim3>,
    specfem::simulation::type::forward>;
template class specfem::time_scheme::newmark<
    specfem::assembly::fields<specfem::dimension::type::dim3>,
    specfem::simulation::type::combined>;
