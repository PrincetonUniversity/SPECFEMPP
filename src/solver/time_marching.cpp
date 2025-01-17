#include "solver/time_marching.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/quadrature.hpp"
#include "enumerations/simulation.hpp"
#include "solver/time_marching.tpp"

namespace {
using qp5 = specfem::enums::element::quadrature::static_quadrature_points<5>;
using qp8 = specfem::enums::element::quadrature::static_quadrature_points<8>;
} // namespace

// Explcit template instantiation

template class specfem::solver::time_marching<
    specfem::simulation::type::forward, specfem::dimension::type::dim2, 5>;

template class specfem::solver::time_marching<
    specfem::simulation::type::forward, specfem::dimension::type::dim2, 8>;

template class specfem::solver::time_marching<
    specfem::simulation::type::combined, specfem::dimension::type::dim2, 5>;

template class specfem::solver::time_marching<
    specfem::simulation::type::combined, specfem::dimension::type::dim2, 8>;
