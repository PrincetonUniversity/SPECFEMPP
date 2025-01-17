#include "solver/time_marching.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/simulation.hpp"
#include "solver/time_marching.tpp"

// Explcit template instantiation

template class specfem::solver::time_marching<
    specfem::simulation::type::forward, specfem::dimension::type::dim2, 5>;

template class specfem::solver::time_marching<
    specfem::simulation::type::forward, specfem::dimension::type::dim2, 8>;

template class specfem::solver::time_marching<
    specfem::simulation::type::combined, specfem::dimension::type::dim2, 5>;

template class specfem::solver::time_marching<
    specfem::simulation::type::combined, specfem::dimension::type::dim2, 8>;
