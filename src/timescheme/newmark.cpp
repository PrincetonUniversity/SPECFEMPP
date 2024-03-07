#include "timescheme/newmark.tpp"
#include "specfem_setup.hpp"
#include "timescheme/interface.hpp"
#include <ostream>

// Explicit template instantiation
template class specfem::time_scheme::newmark<
    specfem::simulation::type::forward>;

template class specfem::time_scheme::newmark<
    specfem::simulation::type::adjoint>;
