#include "specfem/timescheme/newmark.tpp"
#include "specfem/setup.hpp"
#include <ostream>

template class specfem::time_scheme::newmark<
    specfem::assembly::fields<specfem::element::dimension_tag::dim2>,
    specfem::simulation::type::forward>;
template class specfem::time_scheme::newmark<
    specfem::assembly::fields<specfem::element::dimension_tag::dim2>,
    specfem::simulation::type::combined>;
template class specfem::time_scheme::newmark<
    specfem::assembly::fields<specfem::element::dimension_tag::dim2>,
    specfem::simulation::type::combined_undoatt>;

template class specfem::time_scheme::newmark<
    specfem::assembly::fields<specfem::element::dimension_tag::dim3>,
    specfem::simulation::type::forward>;
template class specfem::time_scheme::newmark<
    specfem::assembly::fields<specfem::element::dimension_tag::dim3>,
    specfem::simulation::type::combined>;
template class specfem::time_scheme::newmark<
    specfem::assembly::fields<specfem::element::dimension_tag::dim3>,
    specfem::simulation::type::combined_undoatt>;
